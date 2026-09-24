# EvalScope · 静态 / 结构化 MoE 剪枝研究分叉

**Private research fork** — static / structured Mixture-of-Experts pruning, with EvalScope as the unified capability-eval layer.

本仓库面向 **静态 / 结构化 MoE 剪枝** 实验：在 routed expert 内部做等宽或分层 channel 剪枝（或对照 whole-expert 删除），用冻结的 EvalScope 协议做下游能力比较。EvalScope 本体仍在 `evalscope/`；剪枝方法、协议 overlay 和结果说明在仓库根目录。

## 上游与许可证

- 分叉自 [`modelscope/evalscope`](https://github.com/modelscope/evalscope)，许可证 **Apache-2.0**。
- 本仓库按设计应为 **私有**，已相对上游分叉演进（方法目录、协议与手册远超社区 EvalScope）。**不要当成 drop-in community fork。**
- 不定期回并上游产品功能。上游文档站点描述的是通用评测框架，不是本分叉的 MoE 实验协议。

## 本分叉增加了什么

相对上游 EvalScope，这里主要多了四类内容。

### 1. 评测协议 overlay

可移植的 Quick9 / full6 / full8 协议，正式路径是 `checkpoint → vLLM → EvalScope openai_api`：

| 路径 | 作用 |
| --- | --- |
| [`eval_protocol/`](eval_protocol/README.md) | 冻结 JSON：`quick9.json`、`full6_v1.json`、`full8_v1.json`；`run_vllm_protocol.sh` |
| [`测评协议.md`](测评协议.md) | MoE 公平预算、harness 分层、校准与 SHA256 审计 |
| [`STATIC_MOE_PRUNING_FRAMEWORK_MANUAL.md`](STATIC_MOE_PRUNING_FRAMEWORK_MANUAL.md) | 校准、导出、实验目录命名、结果落盘规范 |
| [`scripts/watch_eval_reports.sh`](scripts/watch_eval_reports.sh) | 协议校验后的报告汇总（不要绕过它手拼分数） |

本地加载补丁仍在 EvalScope adapter 内：MMLU 可读本地 CSV 目录，WinoGrande 可读 `winogrande_1.1.zip`。

### 2. 根目录剪枝方法

根目录每个文件夹是一套独立方法或对照实现。**没有名为 `HSP/` 的顶层目录**；HSP-Hetero / AHSP 在 `CSP/` 内。另有 `RAMP/`（可重构性感知剪枝），不在早期清单里，但树中存在。

详见下一节导航表。

### 3. Serving 环境与启动器

[`eval_protocol/envs/gemma4-vllm-cu128/`](eval_protocol/envs/gemma4-vllm-cu128/README.md) 冻结统一 serving 环境（Python 3.10、torch `2.11.0+cu128`、vLLM `0.23.1.dev0` + CUDA 12.8 补丁）。覆盖：

- Qwen3-30B-A3B-Instruct-2507
- Qwen3.6-35B-A3B
- Gemma4-26B-A4B-it
- DeepSeek-V2-Lite-Chat

用 `setup_gemma4_vllm_cu128.sh` 在新机器重建；**不要**把 conda env 或 vLLM 源码树提交进 git。

### 4. 结果文档与自定义评测

- [`Results.md`](Results.md) / [`Results_analysis.md`](Results_analysis.md)：下游表与诊断笔记。由 `scripts/compile_results_md.py` 生成；不要把大表贴进本 README。
- [`custom_eval/`](custom_eval/)：额外文本 / 多模态自定义评测样例。

## 评测立场

来自 `测评协议.md` 与 `eval_protocol/README.md` 的硬约束：

1. **主能力表只用固定 EvalScope。** 同一横向比较锁定 commit、TaskConfig、prompt、few-shot、decoding、answer extraction。当前冻结生成约束：`seed=42`、`shuffle=false`、`temperature=0`、`do_sample=false`、`enable_thinking=false`。
2. **MoE 公平性审计单独做。** matched routed-expert FFN 预算、shared expert 隔离（不剪、不计 routed budget）、64-channel block / width、profile 与 checkpoint SHA256。这些字段不是能力分数，不要塞进主表同一列。
3. **不要混 harness。** OpenCompass、`lm-evaluation-harness`、LightEval 若跑了，只能作为论文协议复核的次列，不得与 EvalScope 主表混算。
4. **能导出标准 HF checkpoint 的方法，正式评测必须走 vLLM。** `static_expert_profile` 只用于标准 vLLM 表达不了的异构结构、单测和小规模一致性检查。
5. **校准只用 train split。** profile 必须在看 validation / test 指标前冻结。

| 协议 | 用途 | 规模 |
| --- | --- | ---: |
| `quick9` | 冻结的快速横向比较 | 6 数据集子集，共 2798 条 |
| `full6_v1` | 正式全量确认 | 同上 6 个全 split，共 30718 条 |
| `full8_v1` | 全量 + 代码 | full6 + HumanEval 164 + MBPP 500 |

数据集顺序固定：ARC → HellaSwag → WinoGrande → GSM8K → MATH-500 → MMLU（full8 再加 HumanEval / MBPP）。**不要把 Quick9 分数和 full6/full8 并排当成同一协议。** 报告六（或八）数据集分数与宏平均，不用样本数加权冒充总分。

## 方法导航

从文件夹 README / DESIGN 进入，不要从根 README 猜超参。

| 目录 | 是什么 | 从哪里开始 |
| --- | --- | --- |
| [`AIMER/`](AIMER/src/calib_free_prune.py) | 原版 calibration-free AIMER（whole-expert 风格，复用 REAP 脚手架） | `AIMER/scripts/command_calibfree.sh`、`AIMER/src/calib_free_prune.py` |
| [`AIMER_Channel/`](AIMER_Channel/readme.md) | data-free 等宽 channel：concat gate/up/down 的 inverse-AIMER | `AIMER_Channel/readme.md` |
| [`AIMER_Mix/`](AIMER_Mix/readme.md) | AIMER × 几何能量的 rank 混合；CalibrationFree | `AIMER_Mix/readme.md`、`run_calibration_free_full8.sh` |
| [`AIMER_MIX_PLUS/`](AIMER_MIX_PLUS/readme.md) | Mix 骨干 + PP / PRP / LayerProp 伪源救援 | `AIMER_MIX_PLUS/readme.md` |
| [`AIMER_UNIFY/`](AIMER_UNIFY/readme.md) | 同一套 Mix / LayerProp / PRP 融合，不按模型名分支 | `AIMER_UNIFY/readme.md` |
| [`CSP/`](CSP/readme.md) | Canonical Structural Participation；**HSP-Hetero** 与 **AHSP** 也在此目录 | `CSP/readme.md`；HSP：`run_one_model_hsp_full8.sh` |
| [`HARP/`](HARP/readme.md) | 分层结构剪枝（Layer-SP → Expert-SP → Channel-SP）；含 RankAdaptive | `HARP/readme.md`、[`HARP_RankAdaptive.md`](HARP/HARP_RankAdaptive.md) |
| [`Wanda/`](Wanda/readme.md) | WikiText128×2048 校准的 structured Wanda | `Wanda/readme.md`、`run_wikitext128x2048_full8.sh` |
| [`Magnitude/`](Magnitude/readme.md) | data-free 联合 L2 magnitude 基线 | `Magnitude/readme.md` |
| [`Random/`](Random/readme.md) | 每 expert 随机 channel 排列基线 | `Random/readme.md` |
| [`Product/`](Product/readme.md) | gate×up L2 乘积基线（down 不进 ranking） | `Product/readme.md` |
| [`Geom/`](Geom/readme.md) | 三投影 L2 几何均值基线 | `Geom/readme.md` |
| [`TENP/`](TENP/readme.md) | **uniform-width ENP-COS**（不是完整 TENP 论文流程） | `TENP/readme.md`、`run_wikitext128x2048_full8.sh` |
| [`WICK/`](WICK/README.md) | data-free：几何均值 rank + router 伪探针保护 | `WICK/README.md` |
| [`PP/`](PP/README.md) | Pure-Pseudo：只用 router 伪探针分数，无 weight-only 保护 | `PP/README.md` |
| [`ROUTER_LAYERPROP/`](ROUTER_LAYERPROP/README.md) | data-free、router-conditioned LayerProp 与伪 token 传播 | `ROUTER_LAYERPROP/README.md` |
| [`ROUTING_AWARE_HETEROGENEOUS/`](ROUTING_AWARE_HETEROGENEOUS/README.md) | 自校准、routing-aware 异构宽度计划 | `ROUTING_AWARE_HETEROGENEOUS/README.md` |
| [`NAPS/`](NAPS/NAPS_DESIGN.md) | Native-Route AIMER Protection and Selection | `NAPS/NAPS_DESIGN.md`、`run_experiment.sh` |
| [`NAPS_v2/`](NAPS_v2/NAPS_v2_DESIGN.md) | NAPS-v2：选择与输出补偿分离 | `NAPS_v2/NAPS_v2_DESIGN.md`、`run_experiment.sh` |
| [`RAMP/`](RAMP/ramp_design.md) | Reconstructability-Aware 异构 channel 剪枝 | `RAMP/ramp_design.md` |
| [`static_moe_prunning/`](static_moe_prunning/README.md) | V4 静态专家剪枝（prefix block、Route×Tail / Tail-Risk 等） | `static_moe_prunning/README.md` |
| [`reap/`](reap/README.md) | 第三方 **REAP** 官方实现，作 whole-expert 对照 | `reap/README.md`；公平比较约束见 `测评协议.md` |
| [`calibration_method/`](calibration_method/self%20moe%20calibration/readme.md) | Checkpoint-Native MoE Self-Calibration（模型自生成校准 token） | `calibration_method/self moe calibration/readme.md` |

目标模型因方法而异；当前协议冻结的 Dense 目标见 `eval_protocol/README.md`。CSP / HARP 另支持 Mixtral-8x7B 与 OLMoE-1B-7B 的构建入口。

## 故意不进 git 的内容

实验产物、权重和结果 dump **不在仓库里**。`.gitignore` 已排除：

- `/result/`、`/results/`
- `**/experiments/`、`**/checkpoints/`
- `*.pt`、`*.pth`、`outputs/`
- 部分方法本地笔记（如 `NAPS/EXP_RESULTS.md`、`NAPS_v2/ITERATION_LOG.md`）

新服务器需要自行准备：基座 checkpoint、六个（或八个）数据集本地副本、校准 cache、导出权重。见 [`eval_protocol/README.md`](eval_protocol/README.md) 与 `eval_protocol/env.example.sh`。

## 最短上手

```bash
pip install -e .
cp eval_protocol/env.example.sh eval_protocol/env.sh   # 填 PYTHON_BIN / MODEL_PATH / DATASET_ROOT
source eval_protocol/env.sh
bash eval_protocol/envs/gemma4-vllm-cu128/setup_gemma4_vllm_cu128.sh   # 新机器重建 serving env

# 某方法导出 HF checkpoint 后：
PROTOCOL=quick9 bash eval_protocol/run_vllm_protocol.sh <MODEL_ID> <API_BASE> <METHOD> <EXPERIMENT_DIR>
WATCH_SECONDS=0 bash scripts/watch_eval_reports.sh
```

方法侧的 profile / export 命令写在各目录 README，不在这里重复。

## 警告

- 含 **未发表方法**、内部实验设计与本机路径笔记。
- 手册 / 脚本里仍有 `/data01/home/xinpei.gao/...` 一类绝对路径；移植时以 `eval_protocol/env.example.sh` 为准，不要照抄旧机器路径。
- 仓库按设计为私有。不要把方法细节、未发表分数或内部手册同步到公开 fork / 社交媒体。

## 上游 EvalScope

- 上游：https://github.com/modelscope/evalscope
- 通用用法、benchmark 列表与 Web dashboard 以 upstream 为准。本分叉在 MoE 方法与协议上 **ahead**，同时 **不跟踪** 上游全部产品功能（可能 behind 新 benchmark / service）。
- 跑本仓库的剪枝实验：`pip install -e .`，再按 `eval_protocol/` 配环境。不要按上游营销 README 的 `pip install evalscope` 工作流，那不会带上根目录方法。

中文指针：本文件即为主文档。[`README_zh.md`](README_zh.md) 只保留短说明，不再镜像上游 EvalScope 中文介绍。
