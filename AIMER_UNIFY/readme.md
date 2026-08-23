# AIMER-Unify

AIMER-Unify 是一套 **data-free / training-free** 的 routed-MoE **等宽 channel 剪枝** 排序方法。
它只用一份融合规则，同时覆盖 Qwen3、Qwen3.6 和 Gemma4，**不按模型名分支**，也 **不锁 Mix core**。

论文里不能写成「Qwen 保留 Mix，Gemma 丢掉 Mix」。Unify 用 keep-set overlap 当 Mix 的门控：
Mix 与 FFN 证据一致时跟 Mix（Qwen 的典型情况），不一致时把预算交给 LayerProp + PRP（Gemma 的典型情况）。

Method token：`AIMERUnify`。Ranking cache `purpose`：`aimer_unify_ranking`。

---

## 要解决的问题

MixPlus 的 data-free 结果把同一套「Mix core + 伪源救援」拆成了两条故事：

| 模型 | Mix 的角色 | 25% 上最好的 data-free | 50% |
|---|---|---|---|
| Qwen3 / Qwen3.6 | Mix **必须留**。丢掉 Mix（`LPnb` = LayerProp only）25% 掉约 2 点，50% 掉 10–13 点 | MixPlus v2 或 Mix+LP | Mix 仍是主体 |
| Gemma4 | Mix **有害**。带 Mix 的 50% 崩到约 9% | ignore-Mix 的 LPx（LP+PRP）84.7% | adaptive LPxa 44.7%；固定 LPx 在 Wino 上更好 |
| DeepSeek-V2-Lite | MixPlus v2 在 25% 最好 | 50% 全部塌；LayerProp HF load 仍坏 | 暂不跑 Unify |

Gemma 上还有一个 complementary 现象：LP-only 81.2% 与 PRP-only 77.9% 合在一起变成 LPx 84.7%。
因此 FFN 侧应该是 **LayerProp 与 PRP 的自适应混合**，而不是二选一。

Unify 把这两条线收成同一个公式，而不是 `if gemma: drop Mix`。

---

## 证据源（只用 FFN 空间）

每个 routed expert 有三条完整的 channel 排列，全部来自已有 cache，Unify **不重新扫权重、不重建伪源**：

| 源 | 符号 | 含义 | 为什么用 |
|---|---|---|---|
| AIMER-Mix | \(S_{\mathrm{Mix}}\) | 权重侧 AIMER + geom/L2 的 rank 混合 | Qwen 上是稳定先验 |
| LayerProp | \(S_{\mathrm{LP}}\) | 无数据 decoder hidden-state 传到 expert FFN 的排序 | FFN 输入证据；Gemma 上比 Mix 更准 |
| PRP | \(S_{\mathrm{PRP}}\) | 上一层 routed expert `down_proj` 写入方向，经当前 router Top-K 过滤 | 与 LP complementary 的 FFN 通道 |

**明确不用 PP。** PP 是当前层 router-row probe，落在 router 空间，不是 expert FFN 输入。
Gemma4 上 router RMS/scale 不能当 FFN 证据；Unify 在 `unify_core` 里遇到 `pp` 直接拒绝。

---

## 融合公式

对 **每一个** `(layer, expert)`，在目标宽度 \(K\)（retained channels）下独立融合。
所有分数先变成「越大越重要」的 rank percentile，再线性混合。

### 1. FFN 伪源：覆盖门控的 LP+PRP

\[
S_{\mathrm{pseudo}}
= \lambda\, S_{\mathrm{LP}} + (1-\lambda)\, S_{\mathrm{PRP}},
\qquad
\lambda_e = \frac{N_e}{N_e+\tau},\quad \tau=8
\]

- \(N_e\) 是 LayerProp 对该 expert 的 **hit count**（该 expert 被伪 hidden 命中的次数）。
- \(N_e=0\)（未覆盖）\(\Rightarrow \lambda=0\) \(\Rightarrow\) 完全用 PRP。
- \(N_e \to \infty\) \(\Rightarrow \lambda \to 1\) \(\Rightarrow\) 完全用 LayerProp。
- \(\tau=8\) 与 MixPlus LPxa 相同：热 expert 仍偏 LP，冷 expert 把权重交给 PRP。

旧 LayerProp cache 没有 `hit_counts` 时的回退（与 MixPlus 一致）：

- `coverage <= 0` → \(N=0\)，\(\lambda=0\)
- `coverage ∈ (0, 1]`（二值覆盖）→ \(N=\infty\)，\(\lambda=1\)
- `coverage > 1` → 把 coverage 本身当 \(N\)

当前 cache：Gemma4 用 `aimer_mix_plus_lpxa`（有 `hit_counts`）；Qwen3 / Qwen3.6 用 `aimer_mix_plus_lp`（只有 coverage，覆盖到的 expert \(\lambda\to 1\)）。

### 2. Mix 门控：keep-set overlap

\[
\alpha_e
= \frac{\bigl|\mathrm{Top}K(S_{\mathrm{Mix}}) \cap \mathrm{Top}K(S_{\mathrm{pseudo}})\bigr|}{K}
\in [0,1]
\]

\(\alpha\) 不是超参，是 **这一档宽度 \(K\) 上两条 keep-set 的 Jaccard 式重叠率**（交集大小 / \(K\)）。
因此 ranking **随宽度变化**：25% 与 50% 必须各建一份 cache。

### 3. 最终分数

\[
S = \alpha\, S_{\mathrm{Mix}} + (1-\alpha)\, S_{\mathrm{pseudo}}
\]

然后对 \(S\) 做稳定降序，取前 \(K\) 个 channel。所有 routed expert **等宽** \(K\)，不按分数改 width，不跨 expert 做 global Top-K。

直观行为：

- \(\alpha \to 1\)：Mix 与 FFN keep-set 一致 → 输出跟 Mix（Qwen）
- \(\alpha \to 0\)：两套 keep-set 几乎不相交 → 输出跟 LP+PRP（Gemma）
- 中间值：在 Mix 先验和 FFN 证据之间做凸组合，而不是硬切 core/boundary

没有「Mix 前缀不可替换」的 locked core，也没有 MixPlus 那套 `boundary_fraction` / `base_boundary_weight` / `pseudo_weight`。

---

## 单 expert 伪代码

```text
mix_rank  ← descending unit ranks of Mix scores
lp_rank   ← rank percentiles from LayerProp order   (if present)
prp_rank  ← rank percentiles from PRP order         (if present)
N         ← LayerProp hit count (coverage fallback if missing)

if lp and prp:
    λ = N / (N + τ)          # τ = 8; N=0 → λ=0; N=inf → λ=1
    S_pseudo = λ * lp + (1-λ) * prp
elif lp only:
    S_pseudo = lp
elif prp only:
    S_pseudo = prp
else:
    return Mix order         # α = 1

α = |TopK(mix_rank) ∩ TopK(S_pseudo)| / K
S = α * mix_rank + (1-α) * S_pseudo
return argsort(S, descending, stable)
```

构建入口要求 **同时提供 PRP 和 LayerProp**（`build_unify_artifacts.py`）。
缺 PP 是故意的；缺其中一条 FFN 源则拒绝构建，避免静默退化成「又一个 MixPlus 变体」。

---

## 和 MixPlus 的差别

| | MixPlus | Unify |
|---|---|---|
| Mix 的角色 | 锁一段 core，只在 boundary 用伪源救援 | 全程可替换；权重是 overlap |
| 融合超参 | 按模型/宽度手调 `pp/prp/lp` 权重、boundary 比例 | 只有 \(\tau=8\)；\(\alpha,\lambda\) 都由数据（cache）算出 |
| PP | Qwen 默认开 | **禁止** |
| 模型分支 | Gemma 关 PP、开 LP；Qwen 相反 | **无** `if gemma` |
| 宽度 | Mix 排列可共享，boundary 仍 width-specific | Mix 门控本身 width-specific |
| ignore-Mix | LPx / LPxa 把 Mix 权重打成 0 | Mix 仍在公式里，只是 \(\alpha=0\) 时等价于忽略 |

Unify 不是 LPxa 再加一点 Mix，也不是 MixPlus v2 换一套权重。
它是 **agreement gate**：先在 FFN 空间形成 \(S_{\mathrm{pseudo}}\)，再问 Mix 是否同意这套 keep-set。

---

## 结构化剪枝

删 channel \(c\) 时同步切：

- `W_gate[c, :]`
- `W_up[c, :]`
- `W_down[:, c]`

不剪 router、shared expert、dense MLP、多模态或 MTP。等宽导出，HF `moe_intermediate_size` 改成 \(K\)。

---

## 协议宽度

alignment 块大小决定合法 \(K\)。当前四模型协议（Unify 先跑前三个）：

| Arg | 模型 | Align | 25% \(K\) | 50% \(K\) | LayerProp cache |
|---|---|---|---|---|---|
| `qwen3` | Qwen3-30B-A3B-Instruct-2507 | 64 | 576 | 384 | `aimer_mix_plus_lp/qwen3` |
| `gemma4` | Gemma4-26B-A4B-it | 32 | 512 | 352 | `aimer_mix_plus_lpxa/gemma4`（含 hit_counts） |
| `qwen36` | Qwen3.6-35B-A3B | 64 | 384 | 256 | `aimer_mix_plus_lp/qwen36` |
| `deepseek` | DeepSeek-V2-Lite-Chat | 32 | 1056 | 704 | 暂缓（LayerProp load 仍坏） |

每个模型先 50% 再 25%。结果目录：

```text
{NAME}_{ratio}_vllm_CalibrationFree_full8_v1_AIMERUnify_{TIMESTAMP}_{SEED}
```

---

## 目录

| 文件 | 作用 |
|---|---|
| `unify_core.py` | 融合：overlap 门控、\(\lambda\)、诊断 |
| `build_unify_artifacts.py` | 读 Mix/LP/PRP cache，写出 width-specific ranking + profile |
| `export_unify_checkpoint.py` | 按 ranking 切片 routed expert，写出 HF checkpoint |
| `run_one_model_full8.sh` | 复用 cache → build → export → full8 eval |
| `wait_and_launch.sh` | 等 GPU 0/1/2 空闲后分别拉起 Qwen3 / Gemma4 / Qwen3.6 |
| `tests/test_unify_core.py` | overlap 0/1、跟 Mix、跟 FFN、未覆盖 LP→PRP、拒绝 PP |

复用的 cache（不重建）：

```text
/data/xinpeigao/evalscope_results/_artifacts/aimer_mix/$MODEL/aimer_mix_rankings.pt
/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus/$MODEL/prp.pt
LayerProp: 见上表
```

Unify 产物：

```text
/data/xinpeigao/evalscope_results/_artifacts/aimer_unify/$MODEL/
  prp.pt  layerprop.pt
  aimer_unify_{25,50}pct_rankings.pt
  aimer_unify_{25,50}pct_per_layer.pt
  checkpoint_{25,50}/
```

---

## 运行

单测：

```bash
PYTHONPATH=/home/xinpeigao/evalscope \
  /data/xinpeigao/conda_envs/gemma4-vllm-cu128/bin/pytest -q \
  AIMER_UNIFY/tests/test_unify_core.py
```

单模型（需要空闲 GPU；不要抢 3–7）：

```bash
RESULT_ROOT=/home/xinpeigao/evalscope/results \
TIMESTAMP=YYYYMMDDHHMM \
METHOD_TOKEN=AIMERUnify \
  bash AIMER_UNIFY/run_one_model_full8.sh qwen3|gemma4|qwen36 GPU PORT
```

等 GPU 0/1/2 显存 &lt; 3072 MiB 后自动开三路：

```bash
RESULT_ROOT=/home/xinpeigao/evalscope/results \
TIMESTAMP=YYYYMMDDHHMM \
  bash AIMER_UNIFY/wait_and_launch.sh
```

默认端口：GPU0→19890，GPU1→19891，GPU2→19892。  
Qwen3.6 会 source `qwen36.env`（FlashInfer GDN TMA workaround）。

手工 build / export：

```bash
python -m AIMER_UNIFY.build_unify_artifacts \
  --model-path ... \
  --aimer-mix-cache .../aimer_mix_rankings.pt \
  --source prp=.../prp.pt \
  --source layerprop=.../layerprop.pt \
  --output-channel-cache .../aimer_unify_25pct_rankings.pt \
  --output-profile .../aimer_unify_25pct_per_layer.pt \
  --retained-channels 576 \
  --layerprop-tau 8

python -m AIMER_UNIFY.export_unify_checkpoint \
  --model-path ... \
  --profile .../aimer_unify_25pct_per_layer.pt \
  --channel-cache .../aimer_unify_25pct_rankings.pt \
  --output-dir .../checkpoint_25
```

诊断里会记 `mix_alpha_{mean,min,max}` 和 `layerprop_lambda_{mean,min,max}`。
Qwen 上预期 \(\bar\alpha\) 偏高，Gemma 上预期 \(\bar\alpha\) 偏低——这是方法在工作，不是又写了模型开关。
