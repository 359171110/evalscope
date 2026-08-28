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

## 7. 理论边界

严格成立的性质包括：function-equivalence、minimum-energy canonical gauge、
up/down canonical norm equality、CSP gauge invariance、Rényi effective
participation 定义、equal-support 时 $N_{\mathrm{eff}}=k$，以及 closed-form score。

“较高 structural localization 的 channel 更值得保留”以及跨 Qwen3、Qwen3.6、
Gemma4、DeepSeek architecture 的泛化，属于 Structural Specialization Hypothesis，
必须通过独立实验验证。
