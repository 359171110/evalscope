# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.mixin import CodeExecutionSandboxMixin
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


def _mbpp_exec_target(payload: str, result: Any) -> None:
    """Execute one MBPP completion in a spawned subprocess.

    Must be module-level so ``multiprocessing`` spawn can pickle it.
    """

    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec_globals: Dict[str, Any] = {}
            exec(payload, exec_globals)
        result.append({'status': 'success'})
    except Exception as exc:  # noqa: BLE001
        result.append({'status': 'error', 'error': str(exc)})


def execute_mbpp_locally(code: str, timeout: float) -> Dict[str, Any]:
    """Run MBPP assertions in a subprocess when Docker sandbox is unavailable."""

    import multiprocessing

    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    result = manager.list()
    process = ctx.Process(target=_mbpp_exec_target, args=(code, result))
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
        process.join()
        return {'status': 'timeout', 'error': 'Code execution timed out.'}
    if not result:
        return {'status': 'timeout', 'error': 'Code execution returned no result.'}
    return dict(result[0])

FEWSHOT_PROMPT = """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n[BEGIN]\ndef similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)\n[DONE]
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math\r
def is_not_prime(n):\r
    result = False\r
    for i in range(2,int(math.sqrt(n)) + 1):\r
        if n % i == 0:\r
            result = True\r
    return result
[DONE]
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq\r
def heap_queue_largest(nums,n):\r
  largest_nums = hq.nlargest(n, nums)\r
  return largest_nums
[DONE]
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
[BEGIN]
"""  # noqa: E501

PROMPT = """You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='mbpp',
        pretty_name='MBPP',
        tags=[Tags.CODING],
        description="""
## Overview

MBPP (Mostly Basic Python Problems) is a benchmark consisting of approximately 1,000 crowd-sourced Python programming problems designed for entry-level programmers. It evaluates a model's ability to understand problem descriptions and generate correct Python code.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Natural language task description with test cases
- **Output**: Python function implementation
- **Difficulty**: Entry-level programming problems

## Key Features

- ~1,000 crowd-sourced programming problems
- Covers programming fundamentals and standard library usage
- Each problem includes task description, solution, and 3 test cases
- Problems are designed to be solvable by entry-level programmers
- Automatic evaluation through test case execution

## Evaluation Notes

- Default configuration uses **3-shot** examples
- **Security Warning**: Sandbox environment is required for safe code execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation
- Default timeout is 20 seconds per problem
- Code is extracted from `[BEGIN]...[DONE]` blocks if present
""",
        dataset_id='google-research-datasets/mbpp',
        subset_list=['full'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        train_split='prompt',
        eval_split='test',
        few_shot_num=3,
        prompt_template=PROMPT,
        few_shot_prompt_template=FEWSHOT_PROMPT,
        review_timeout=20,
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
    )
)
class MBPPAdapter(CodeExecutionSandboxMixin, DefaultDataAdapter):
    """
    MBPP adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""

        return Sample(
            input=record['text'],
            target=record['code'],
            metadata={
                'test_list': record['test_list'],
                'task_id': record['task_id'],
                'test_setup_code': record['test_setup_code'],
            }
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return ''

    def format_prompt_template(self, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.prompt_template.format(question=sample.input, tests=tests)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.few_shot_prompt_template.format(question=sample.input, tests=tests)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""

        code = self.postprocess_completion(prediction)
        code = '\n'.join([task_state.metadata['test_setup_code'], code])
        if 'if __name__ ==' in code:
            code = code[:code.index('if __name__ ==')]
        return code

    @classmethod
    def postprocess_completion(cls, completion, stop_words=['\nassert', '\n"""']):
        from evalscope.utils.code_utils import extract_code_from_freeform_completion

        completion = completion.lstrip(' \t')
        if '[DONE]' in completion:
            completion = completion[:completion.index('[DONE]')]
        completion = completion.strip()
        # Some chat models wrap the solution in square brackets: [def foo(): ...]
        if completion.startswith('[') and 'def ' in completion:
            end = completion.rfind(']')
            if end > 0:
                inner = completion[1:end].strip()
                if inner.startswith('def ') or inner.startswith('import '):
                    completion = inner

        code, _ = extract_code_from_freeform_completion(completion, 'python', first_block_only=True)
        if not str(code).strip() and 'def ' in completion:
            code = completion

        for st in stop_words:
            index = code.find(st)
            if index != -1:
                code = code[:index]
        return code

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        problem = task_state.metadata
        completion = filtered_prediction
        for test in problem['test_list']:
            completion += '\n' + test + '\n'

        if self.use_sandbox:
            res = self.execute_code_in_sandbox(code=completion, timeout=self.review_timeout, language='python')
            passed = res.get('status') == 'success'
        else:
            res = execute_mbpp_locally(completion, float(self.review_timeout))
            passed = res.get('status') == 'success'
        score.value = {'acc': passed}
        score.metadata = {'task_id': problem['task_id'], 'timeout': self.review_timeout, 'execution_result': res}
        score.main_score_name = 'acc'

        return score
