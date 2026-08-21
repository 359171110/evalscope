from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
from evalscope.benchmarks.mbpp.mbpp_adapter import MBPPAdapter


def test_humaneval_strips_chat_leading_space_before_def() -> None:
    raw = ' def below_zero(operations: List[int]) -> bool:\n    return False'
    assert HumanevalAdapter._postprocess(raw).startswith('def below_zero')


def test_humaneval_keeps_body_only_indent() -> None:
    raw = '    return False\n'
    assert HumanevalAdapter._postprocess(raw) == raw


def test_mbpp_unwraps_bracketed_def_and_leading_space() -> None:
    raw = ' [def square_perimeter(side_length):\n    return side_length * 4\n]\n[DONE]'
    code = MBPPAdapter.postprocess_completion(raw)
    assert 'def square_perimeter' in code
    assert code.strip().startswith('def square_perimeter')
