# CSP：Canonical Structural Participation

本目录实现一个完全独立的、data-free、training-free、forward-free 的
SwiGLU structured channel pruning 方法。CSP 不从已有 magnitude 或 AIMER
score 出发，而是在 channel signature 上定义结构参与度。Up/down 的
minimum-energy canonicalization 是可选项，默认关闭；默认 score 使用 raw
signature，只有显式传入 `--canonicalize` 才进入 canonical gauge。

## 1. 问题定义

一个 SwiGLU expert 为

$$
E(x)=W_d[\operatorname{SiLU}(W_gx)\odot W_ux].
$$

第 $c$ 个 channel 由完整三元组

$$
g_c=W_g[c,:],\qquad u_c=W_u[c,:],\qquad d_c=W_d[:,c]
$$

构成。剪枝时同步删除 gate/up 的第 $c$ 行和 down 的第 $c$ 列。每个
expert 独立排序，但所有 routed experts 保留相同的 $K$，得到 kernel-friendly
的等宽结构化 checkpoint。

## 2. Function-equivalence 与 canonicalization

对任意非零标量，$(u_c,d_c)$ 与 $(\alpha u_c,d_c/\alpha)$ 表示相同的
channel function。启用 canonicalization 时，CSP 在这个 gauge orbit 上选择
minimum-energy representative：

$$
\alpha_c^\star=\sqrt{\frac{\lVert d_c\rVert_2}{\lVert u_c\rVert_2}},\qquad
\widetilde u_c=\alpha_c^\star u_c,\qquad
\widetilde d_c=d_c/\alpha_c^\star.
$$

其满足

$$
\lVert\widetilde u_c\rVert_2=\lVert\widetilde d_c\rVert_2
=\sqrt{\lVert u_c\rVert_2\lVert d_c\rVert_2},
$$

并且是 $\lVert u'\rVert_2^2+\lVert d'\rVert_2^2$ 在等价类上的唯一正解。

## 3. Structural participation score

启用 canonicalization 时，规范 channel signature 为

$$
\widetilde\theta_c=[g_c;\widetilde u_c;\widetilde d_c],\qquad N=3d.
$$

令 $p_{c,j}=|\widetilde\theta_{c,j}|/\lVert\widetilde\theta_c\rVert_1$，
则 effective participation 为

$$
N_{\mathrm{eff}}(c)=\frac{\lVert\widetilde\theta_c\rVert_1^2}
{\lVert\widetilde\theta_c\rVert_2^2}.
$$

CSP saliency 定义为相对于完全 diffuse distribution $U_N$ 的 Rényi-2 divergence：

$$
S_c^{\mathrm{CSP}}
=D_2(p_c\Vert U_N)
=\log\left(N\frac{\lVert\widetilde\theta_c\rVert_2^2}
{\lVert\widetilde\theta_c\rVert_1^2}\right).
$$

高分 channel 保留。该分数只描述 canonical structural localization；“localized
channel 更不可替代”是需要实验验证的经验假设，不是定理。

## 4. Closed form

无需构造 signature 或概率分布。canonicalization 开启时令

$$
L_{1,c}=\lVert g_c\rVert_1+\alpha_c^\star\lVert u_c\rVert_1
+\frac{\lVert d_c\rVert_1}{\alpha_c^\star},
$$

$$
L_{2,c}^2=\lVert g_c\rVert_2^2+2\lVert u_c\rVert_2\lVert d_c\rVert_2.
$$

则；canonicalization 关闭时，直接使用
$L_1=\lVert g_c\rVert_1+\lVert u_c\rVert_1+\lVert d_c\rVert_1$ 和
$L_2^2=\lVert g_c\rVert_2^2+\lVert u_c\rVert_2^2+\lVert d_c\rVert_2^2$。
两种模式都使用：

$$
\boxed{S_c^{\mathrm{CSP}}=\log(3d)+\log L_{2,c}^2-2\log L_{1,c}.}
$$

实现使用 FP32 reductions 和 log-domain，避免大模型权重乘积的溢出。对
functionally dead 或数值退化 channel 使用 viability gate：

$$
V_c=\lVert g_c\rVert_2\lVert u_c\rVert_2\lVert d_c\rVert_2.
$$

$V_c\le\tau$ 的 channel 得分为 $-\infty$。$V_c$ 对 up/down gauge 严格不变。

## 5. Architecture-aware coordinate

如果 expert 前存在固定坐标变换 $z=A_ex$，则在固定线性部分上使用
$g^{\mathrm{eff}}=A_e^\top g$、$u^{\mathrm{eff}}=A_e^\top u$；input-dependent
RMSNorm 本身不被错误地吸收到权重中。当前实现对 Gemma4 自动使用
`pre_feedforward_layernorm_2.weight` 作为固定 learned scale，Qwen3、Qwen3.6
和 DeepSeek-V2-Lite 使用 raw expert coordinates。

## 6. 实现与模型支持

| 模型 | checkpoint type | expert layout | alignment | 25% 保留 | 50% 保留 |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3 | `qwen3_moe` | separate | 64 | 576 | 384 |
| Qwen3.6 | `qwen3_5_moe_text` | packed | 64 | 384 | 256 |
| Gemma4 | `gemma4_text` | packed | 32 | 512 | 352 |
| DeepSeek-V2-Lite | `deepseek_v2` | separate | 32 | 1056 | 704 |
| OLMoE-1B-7B | `olmoe` | separate | 64 | 768 | 512 |

主要入口：

* [csp_core.py](csp_core.py)：canonicalization、CSP score、stable ranking；
* [model_adapter.py](model_adapter.py)：四类模型的 checkpoint layout adapter；
* [build_csp_artifacts.py](build_csp_artifacts.py)：weight-only ranking/profile 构建；
* [export_csp_checkpoint.py](export_csp_checkpoint.py)：routed expert 结构化导出；
* [run_prepare.sh](run_prepare.sh)：只构建和导出，不启动推理实验。

构建并导出一个模型（默认不启用 canonicalization）：

```bash
cd /path/to/evalscope
MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507 \
  bash CSP/run_prepare.sh qwen3 384 /path/to/csp-artifacts
```

如需启用 canonicalization：

```bash
CSP_CANONICALIZE=1 MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507 \
  bash CSP/run_prepare.sh qwen3 384 /path/to/csp-artifacts-canonical
```

等价的 Python 入口为：

```bash
python -m CSP.build_csp_artifacts \
  --model-path /path/to/model \
  --output-channel-cache /path/to/csp_rankings.pt \
  --output-profile /path/to/csp_profile.pt \
  --retained-channels 384
```

显式启用 canonicalization：

```bash
python -m CSP.build_csp_artifacts \\
  --model-path /path/to/model \\
  --output-channel-cache /path/to/csp_rankings.pt \\
  --output-profile /path/to/csp_profile.pt \\
  --retained-channels 384 \\
  --canonicalize
```

构建过程不使用 tokenizer、calibration samples、activation、gradient、router
statistics 或 downstream metrics。DeepSeek 的 dense 前置层和 shared experts
不剪，导出时额外保持 fused shared expert width。

HSP-Hetero 使用模型专属的、以 50% 保留宽度为中心的三档配置：

$$
\mathcal K=\{k_0-64,k_0,k_0+64\},\qquad
(n_{k_0+64},n_{k_0},n_{k_0-64})=(E/4,E/2,E/4),
$$

其中 $k_0$ 为模型的 50% 基准宽度。对每个模型，
$k_0+64$ 和 $k_0-64$ 两档分别分配给 $E/4$ 个 experts，
从而保持总容量 $E\times k_0$ 不变。高分 expert 分配高档宽度，
中间 50% 分配基准宽度，低分 expert 分配低档宽度；expert 内部的
排序仍由 CSP score 决定。

模型专属的 HSP-Hetero 宽度三档如下：

| 模型 | 低档 | 基准档 | 高档 |
| --- | ---: | ---: | ---: |
| Qwen3 | 320 | 384 | 448 |
| Qwen3.6 | 192 | 256 | 320 |
| Gemma4 | 288 | 352 | 416 |
| DeepSeek-V2-Lite | 640 | 704 | 768 |
| OLMoE-1B-7B | 448 | 512 | 576 |
| Mixtral-8x7B | 7104 | 7168 | 7232 |

上述数值用于文档说明与离线配置记录，不需要运行 build 即可确认。

HSP-Hetero 构建时可使用：

```bash
bash CSP/run_prepare.sh qwen3 "" /path/to/hsp-artifacts
```

其他模型的默认三档为：Qwen3=`320/384/448`、Qwen3.6=`192/256/320`、
Gemma4=`288/352/416`、DeepSeek-V2-Lite=`640/704/768`、OLMoE=`448/512/576`、Mixtral=`7104/7168/7232`；分别通过
`CSP_HETEROGENEOUS_WIDTHS` 与 `CSP_BUDGET_WIDTH` 传入。标准 HF/vLLM 导出统一
zero-pad 到高档物理宽度，不能据此宣称已经获得真实异构 kernel 加速。

## 6.1 Adaptive HSP（AHSP）

AHSP 是 HSP 的自适应升级版，保留现有 CSP/HSP 的入口和 profile 格式，不改变
旧方法的默认行为。AHSP 不再仅根据 Expert-SP 的排序分配三档固定宽度，而是为
每个 expert 建立 compression risk curve。

对 expert $e$，先计算 Expert-SP；对其 channel 按 Channel-SP 降序排列并形成
aligned blocks。保留前 $n$ 个 block 时，channel tail risk 为：

$$
R^C_e(n)=1-
\frac{\sum_{j=1}^{n}\exp(S^C_{e,j})}
{\sum_{j=1}^{M}\exp(S^C_{e,j})}.
$$

AHSP 将 expert 的 structural fragility 作为权重：

$$
R_e(n)=
\frac{\exp(S^E_e)}{\operatorname{mean}_e[\exp(S^E_e)]}R^C_e(n).
$$

随后从最小允许宽度开始，反复把一个 block 分配给能获得最大边际风险下降的
expert，直到达到每层固定总预算。候选分配同时包含 AHSP greedy allocation、旧
HSP 的 25/50/25 分位数分配和等宽 CSP 分配。三者使用同一个 risk objective 比较，
最终选择 residual risk 最小者；因此 risk curve 不支持自适应分配时可以退化为
旧 HSP 或 CSP。

AHSP 使用模型参数，不需要 calibration、forward、activation、gradient、router
statistics 或下游指标。每个 expert 的最终 channel 仍取 Channel-SP 排序的前
$K_e$ 个，并同步裁剪 gate/up 行和 down 列。packed 模型的 gate/up 两个半区分别
padding 后再拼接，以保持 HF/vLLM 的物理布局。

示例（Qwen3）：

```bash
python -m CSP.build_csp_artifacts \
  --model-path /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output-channel-cache /path/to/ahsp/csp_rankings.pt \
  --output-profile /path/to/ahsp/ahsp_profile.pt \
  --ahsp --ahsp-min-width 320 --budget-width 384 --ahsp-max-width 448
```

AHSP profile 可直接交给现有 `CSP.export_csp_checkpoint` 导出；旧的 uniform CSP
和 HSP-Hetero 命令保持不变。当前 risk curve 是结构性 proxy，是否优于 HSP 仍需
通过独立的 loss/accuracy 实验验证。

## 7. 理论边界

严格成立的性质包括：function-equivalence、minimum-energy canonical gauge、
up/down canonical norm equality、CSP gauge invariance、Rényi effective
participation 定义、equal-support 时 $N_{\mathrm{eff}}=k$，以及 closed-form score。

“较高 structural localization 的 channel 更值得保留”以及跨 Qwen3、Qwen3.6、
Gemma4、DeepSeek architecture 的泛化，属于 Structural Specialization Hypothesis，
必须通过独立实验验证。
