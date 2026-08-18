# 静态 MoE 评测协议（可移植版）

这是从 `STATIC_MOE_PRUNING_FRAMEWORK_MANUAL.md` 抽出的评测协议，供在新服务器上继续评测。
本仓库只包含代码和协议配置，**不包含实验结果、profile、校准 cache、导出权重或原始模型**。

完整规范仍以手册为准。新服务器上的正式下游评测请走本目录中的冻结协议，而不是复制旧机器上的绝对路径。

## 1. 仓库里有什么、没有什么

包含：

- EvalScope 本体，以及本地 MMLU / WinoGrande 加载补丁
- 冻结协议：`quick9`、`full6_v1`、`full6_unlimited`
- 结果目录创建、vLLM 评测入口、报告汇总脚本
- 各剪枝方法的**源代码**（`static_moe_prunning`、`WICK`、`PP`、`TENP`、`NAPS`、`NAPS_v2`、`RAMP`、`AIMER`、`reap`）

不包含，必须在新服务器上单独准备：

- `result/` 下的历史实验
- `**/experiments/`、`**/checkpoints/`、`*.pt`、`*.safetensors`
- 基座模型，例如 Qwen3-30B-A3B / Qwen3.6-35B-A3B / Gemma4-26B-A4B
- Quick9 六个数据集的本地副本

## 2. 强制原则

1. 能导出标准 Hugging Face checkpoint 的方法，正式评测必须走 `checkpoint -> vLLM -> EvalScope openai_api`。
2. `static_expert_profile` 只用于异构结构、单元测试和小规模一致性检查。
3. 校准数据只能来自 train split；profile 必须在看 validation/test 指标前冻结。
4. 同一横向比较必须使用相同数据版本、样本顺序、prompt、生成参数和 seed。
5. 新结果只写 `$RESULT_ROOT`（默认 `$ROOT/result`），不要写回方法目录里的历史 `experiments/results`。

## 3. 新服务器初始化

```bash
git clone git@github.com:359171110/evalscope.git
cd evalscope
pip install -e .

cp eval_protocol/env.example.sh eval_protocol/env.sh
# 编辑 env.sh：PYTHON_BIN、VLLM_PYTHON、MODEL_PATH、DATASET_ROOT
source eval_protocol/env.sh
```

六个数据集放到 `$DATASET_ROOT`，或在 `env.sh` 里改成实际路径：

| 环境变量 | Quick9 / full6_v1 用途 |
| --- | --- |
| `ARC_PATH` | ARC-Challenge + ARC-Easy |
| `HELLASWAG_PATH` | HellaSwag |
| `WINOGRANDE_PATH` | `winogrande_1.1.zip` |
| `GSM8K_PATH` | GSM8K，0-shot |
| `MATH_500_PATH` | MATH-500 |
| `MMLU_PATH` | 本地 MMLU CSV 目录 |

启动任务前检查 GPU 和已有进程：

```bash
nvidia-smi
ps -eo pid,ppid,pgid,sid,stat,etime,args |
  grep -E 'vllm|evalscope|run_.*quick9|static_profile' |
  grep -v grep || true
```

## 4. 冻结评测协议

三个协议共用同一数据顺序：

```text
ARC -> HellaSwag -> WinoGrande -> GSM8K -> MATH-500 -> MMLU
```

共同生成约束：`seed=42`，`shuffle=false`，`temperature=0`，`do_sample=false`，
`enable_thinking=false`。vLLM / EvalScope 必须在每个请求里显式传
`extra_body.chat_template_kwargs.enable_thinking=false`。

报告六个数据集分数和六数据集宏平均，不用样本数加权冒充总分。

| 协议 | 用途 | 样本数 | `max_tokens` |
| --- | --- | --- | --- |
| `quick9` | 冻结的快速横向比较 | 600 / 1000 / 400 / 128 / 100 / 570，共 2798 | ARC 2048，HellaSwag 512，WinoGrande 1024，GSM8K 2048，MATH-500 4096，MMLU 2048 |
| `full6_v1` | 正式全量确认，样本数与 Quick9 相同 | 同上 2798 | 同上。与 Quick9 的差别是实验目录里的协议字段，不是另一套抽样 |
| `full6_unlimited` | NAPS 等全 split 评测 | 3548 / 10042 / 1267 / 1319 / 500 / 14042 | 同上。分数不能和 Quick9 / full6_v1 并排当同一协议 |

旧的 Qwen3-30B Quick9 曾用 `64/32/32/1024/4096/1536`。那是另一套 generation 预算；改过 `max_tokens` 的实验必须使用新 work dir，不得复用旧 prediction cache。

机器可读定义：

- `eval_protocol/quick9.json`
- `eval_protocol/full6_v1.json`
- `eval_protocol/full6_unlimited.json`

## 5. 结果目录命名

每次评测只创建一个顶层目录：

```text
目标模型_剪枝率_推理方式_校准集_评测协议_方法名称_时间戳_随机数种子
```

用脚本创建，不要手写：

```bash
export EXPERIMENT_DIR="$($CODE_ROOT/scripts/create_result_dir.sh \
  --model "$MODEL_NAME" \
  --pruning-ratio-label 50 \
  --pruning-ratio-percent 50 \
  --inference vllm \
  --calibration Mixed512x1024 \
  --protocol "$PROTOCOL" \
  --method MyMethod)"
echo "$EXPERIMENT_DIR"
```

内部布局：

```text
result/<严格实验名>/
  experiment_manifest.json
  checkpoints/<method>/
  server_logs/<method>.log
  <method>/<dataset>/{configs,logs,predictions,reviews,reports}/
```

方法名只允许字母、数字和连字符，不能有空格或下划线。

## 6. 推荐评测路径

能导出统一宽度 checkpoint 时：

```bash
export METHOD=MyMethod
export MODEL_ID="${MODEL_NAME}-50-$METHOD"
export PORT="${VLLM_PORT:-18080}"
export CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints/$METHOD"

CUDA_VISIBLE_DEVICES=0 "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT_DIR" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --seed 42 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  >"$EXPERIMENT_DIR/server_logs/$METHOD.log" 2>&1 &

PROTOCOL="$PROTOCOL" bash eval_protocol/run_vllm_protocol.sh \
  "$MODEL_ID" \
  "http://127.0.0.1:$PORT" \
  "$METHOD" \
  "$EXPERIMENT_DIR"
```

无法被标准 vLLM 表达的异构宽度，才允许 `transformer` profile runtime，并在 manifest 里写明原因。

## 7. 汇总

```bash
WATCH_SECONDS=0 bash scripts/watch_eval_reports.sh
```

默认扫描 `$RESULT_ROOT`。需要指定实验时，编辑 `scripts/watch_eval_reports.sh` 顶部的 `EXPERIMENT_PATHS`。

## 8. 与手册其余章节的关系

方法注册、校准 cache、checkpoint 导出/归档仍见：

```text
STATIC_MOE_PRUNING_FRAMEWORK_MANUAL.md
```
