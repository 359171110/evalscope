# NAPS-v2: Native-Anchor Protection and Expert-Output Compensation

## 0. Method Positioning

NAPS-v2 is a **data-free, training-free, fixed-width structured channel-pruning method for MoE experts**. It preserves the validated Stable-AIMER ranking backbone, but revises the two parts of NAPS-v1 that were too conservative:

1. NAPS-v1 allowed PP replacement only when a hard effective-evidence gate was passed and accepted each swap only after a pseudo-output reconstruction test. This reduced Qwen3 to only 20 swaps over 6,144 experts, despite earlier experiments showing stable gains when approximately 5% of each expert's channels were replaced by the highest-ranked PP channels.
2. NAPS-v1 merge compensation covered only AIMER-tail channels displaced by a swap. It did not compensate the output contribution of the complete pruned set, so its additional effect was necessarily small under 50% pruning.

NAPS-v2 separates channel selection from output compensation:

$$
\boxed{
\mathrm{Effective\mbox{-}zero\ handling}
\rightarrow
\mathrm{Stable\mbox{-}AIMER\ baseline}
\rightarrow
\mathrm{dynamic\ PP\ swap}
\rightarrow
\mathrm{PP\mbox{-}output\ coverage}
\rightarrow
\mathrm{expert\mbox{-}level\ residual\ compensation}
}
$$

The central distinction is:

$$
\boxed{
\mathrm{Selection\ score}
\neq
\mathrm{Compensation\ coverage\ score}
}
$$

In particular, the `down_proj` column norm is not added to the Stable-AIMER or PP channel-selection ranking. It is used only to estimate how much expert output is covered by the retained channel set.

NAPS-v2 defines two independently exportable variants:

$$
\boxed{
\mathrm{NAPS\text{-}v2\text{-}Mask}
\quad\text{and}\quad
\mathrm{NAPS\text{-}v2\text{-}ExpertComp}
}
$$

Both variants must use exactly the same retained channel set. `ExpertComp` may only modify retained `down_proj` columns; it must not alter the mask, router, gate projection, up projection, or structural expert width.

---

## 1. Empirical Motivation

### 1.1 Stable-AIMER remains the global ranking backbone

Existing experiments established that Stable Concat-AIMER is the strongest reliable global ranking among the tested AIMER variants. NAPS-v2 therefore does not replace or rescale its active-channel score.

For channel $c$ of expert $e$, define the aligned parameter tuple:

$$
w_{e,c}=[g_{e,c};u_{e,c};d_{e,c}],
$$

where $g_{e,c}$ and $u_{e,c}$ are rows of `gate_proj` and `up_proj`, and $d_{e,c}$ is a column of `down_proj`.

The Stable-AIMER score is:

$$
s^A_{e,c}
=
\frac{
\sqrt{\operatorname{mean}(w_{e,c}^2)}
}{
\operatorname{mean}(|w_{e,c}|)+\epsilon_A
}.
$$

### 1.2 Fixed PP replacement has already shown stable gains

Earlier experiments established the following empirical pattern when replacing the Stable-AIMER retained tail with the highest-ranked PP channels:

- approximately 5% of the original expert width gives the strongest stable gain;
- gains diminish beyond 5%;
- near 10%, the gain saturates and may decline slightly.

These percentages refer to the original expert channel count $C$, not to the retained width $K$ or prune-set size.

NAPS-v2 therefore treats PP replacement as a preregistered ranking operation rather than a sequence of pseudo-output reconstruction decisions. The replacement fraction is dynamically restricted to the conservative interval:

$$
\rho_e^{\mathrm{swap}}\in[0.03,0.08].
$$

### 1.3 Router-derived probes are structural probes, not token centroids

For a layer router matrix $W_R\in\mathbb R^{E\times H}$, NAPS constructs one router-derived pseudo token per expert:

$$
q_i=\operatorname{RMSNorm}(W_R[i,:]).
$$

The RMSNorm is applied exactly once. Native routing then computes:

$$
z_{i,j}=q_i^\top W_R[j,:]
$$

and selects the native Top-$k$ experts. The router stage does not apply a second RMSNorm.

The router row is a discriminative direction, not necessarily the centroid of real hidden states routed to the expert. Consequently, $q_i$ is not guaranteed to route to expert $i$.

The audited routing statistics are:

| Model | Expert records | Self probe enters own Top-8 | Self-route failures | Zero native probes |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 6,144 | 6,144 (100%) | 0 | 0 |
| Qwen3.6 | 10,240 | 9,014 (88.03%) | 1,226 | 522 |

For experts receiving exactly one native pseudo token:

| Model | Single-token experts | Unique token is self | Unique token is non-self |
| --- | ---: | ---: | ---: |
| Qwen3 | 1,094 | 1,094 | 0 |
| Qwen3.6 | 1,254 | 1,113 | 141 |

NAPS-v2 therefore assigns two distinct meanings to the probe bank:

- **native routed probes** preserve the actual router competition geometry;
- **self anchors** guarantee that every expert has at least one structural probe for PP scoring and output-coverage estimation.

No forced self assignment may be reported as native routing.

### 1.4 NAPS-v1 merge coverage was structurally too narrow

NAPS-v1 supplied only swap-displaced channels to bounded merge:

$$
\mathcal D_e
=
\mathcal K_e^A\setminus\mathcal K_e^{\mathrm{NAPS}}.
$$

It did not compensate the complete final prune set:

$$
\mathcal P_e
=
\{1,\ldots,C\}\setminus\mathcal K_e^{\mathrm{NAPS}}.
$$

This distinction is substantial. The completed NAPS-v1 artifacts contained:

| Model | Budget | Accepted swaps | Accepted merge pairs |
| --- | --- | ---: | ---: |
| Qwen3 | B9 | 20 | 0 |
| Qwen3 | B6 | 20 | 10 |
| Qwen3.6 | B9 | 17,017 | 2,151 |
| Qwen3.6 | B6 | 17,606 | 556 |

At B6, each expert loses half of its physical channels, but NAPS-v1 attempted compensation for only a small swap-derived subset. NAPS-v2 instead estimates retained expert-output coverage over the full active channel set and compensates selected residual output from the final full prune set.

---

## 2. Notation and Fixed-Width Constraint

For expert $e$ in layer $l$:

- $C$: original intermediate width;
- $K$: retained intermediate width;
- $\mathcal A_e$: active, non-effective-zero channels;
- $\mathcal Z_e$: effective-zero channels;
- $\mathcal K_e^A$: Stable-AIMER baseline keep set;
- $\mathcal K_e$: final NAPS-v2 keep set;
- $\mathcal P_e$: final prune set;
- $Q_e^N$: native routed pseudo probes for expert $e$;
- $q_e$: expert $e$'s router-derived self anchor;
- $Q_e^C$: coverage probe set;
- $H_e(Q)$: SwiGLU channel-response matrix on probe set $Q$;
- $D_e$: full `down_proj` matrix, with channel-aligned columns.

Current model widths are:

| Model | $C$ | B9 retained $K$ | B6 retained $K$ |
| --- | ---: | ---: | ---: |
| Qwen3 | 768 | 576 | 384 |
| Qwen3.6 | 512 | 384 | 256 |

All scores, responses, regressions, coverage values, and validation losses must be calculated in FP32. Exported tensors are converted back to the checkpoint dtype.

---

## 3. Stage A: Effective-Zero Handling

Define:

$$
z_{e,c}
=
\max\left(
\|g_{e,c}\|_\infty,
\|u_{e,c}\|_\infty,
\|d_{e,c}\|_\infty
\right).
$$

With $\tau_0=10^{-12}$:

$$
\mathcal Z_e=\{c:z_{e,c}<\tau_0\}.
$$

Effective-zero channels:

- receive Stable-AIMER score $-\infty$;
- do not enter the PP swap-in candidate set;
- do not enter PP-output coverage denominators;
- cannot be residual-compensation targets;
- cannot contribute to a compensation update.

When $|\mathcal Z_e|>C-K$, fixed width requires retaining some zero fillers. The exact count is:

$$
F_e^0=|\mathcal Z_e|-(C-K).
$$

Zero fillers are selected deterministically by original channel index after all active channels. They are structural padding only and must not be counted as PP-retained output coverage.

---

## 4. Stage B: Stable-AIMER Baseline

Set:

$$
\widetilde s^A_{e,c}
=
\begin{cases}
-\infty,&c\in\mathcal Z_e,\\
s^A_{e,c},&c\in\mathcal A_e.
\end{cases}
$$

Stable descending sort with original channel index as the final tie-breaker gives the baseline order $\pi_e^A$.

The first $K$ channels form:

$$
\mathcal K_e^A=\pi_e^A[:K].
$$

The remaining channels form:

$$
\mathcal P_e^A=\pi_e^A[K:].
$$

NAPS-v2 does not globally rerank Stable-AIMER. It replaces a bounded suffix of $\mathcal K_e^A$ with a bounded prefix of an independently computed PP ranking over $\mathcal P_e^A$.

---

## 5. Stage C: Dual-Semantics Pseudo-Token Bank

### 5.1 Probe construction

For each layer, construct one pseudo token for each router row:

$$
q_i=\operatorname{RMSNorm}_{\gamma,\epsilon}(W_R[i,:]).
$$

This is the only RMSNorm applied to the pseudo token.

Native router logits are:

$$
Z=QW_R^\top.
$$

Let $T_i^N$ be the native Top-$k$ expert indices for pseudo token $q_i$. Native Top-$k$ weights are:

$$
\alpha_{i,j}
=
\operatorname{softmax}_{j\in T_i^N}(Z_{i,j}).
$$

### 5.2 Native routed probe set

For expert $e$:

$$
Q_e^N=\{q_i:e\in T_i^N\}.
$$

The native routed count is:

$$
M_e=|Q_e^N|.
$$

This count is used to set the dynamic swap fraction. It is not interpreted as a real-data token frequency.

### 5.3 Self anchor and coverage probe set

The self anchor for expert $e$ is $q_e$, whether or not $e\in T_e^N$.

Define:

$$
Q_e^C=Q_e^N\cup\{q_e\}.
$$

If $q_e\in Q_e^N$, it appears once. If it is not naturally routed to $e$, it is added only as a structural anchor.

For an added anchor, NAPS-v2 must not invent a native router probability. Coverage and PP scoring use one of the following frozen anchor-weight policies:

1. **Default v2 policy:** uniform averaging over $Q_e^C$;
2. **Required ablation:** native weights for $Q_e^N$ plus anchor weight equal to the median native weight in that expert, followed by normalization.

The default method uses uniform averaging because it avoids assigning a fabricated routing confidence to a non-native anchor.

Diagnostics must record:

- `native_probe_count`;
- `self_naturally_routed`;
- `anchor_added`;
- `coverage_probe_count`;
- native self rank and Top-$k$ margin;
- routed expert IDs for each pseudo token in a separate layer-level audit artifact.

---

## 6. Stage D: PP Channel Ranking

For probes $Q_e^C$, compute the SwiGLU response of channel $c$:

$$
h_{e,c}(q)
=
\operatorname{SiLU}(g_{e,c}^\top q)
(u_{e,c}^\top q).
$$

The default PP selection score deliberately excludes the `down_proj` norm:

$$
s^{PP}_{e,c}
=
\operatorname{Mean}_{q\in Q_e^C}
|h_{e,c}(q)|.
$$

This score answers:

> Which AIMER-pruned active channels respond most strongly on the expert's structural probes?

It does not attempt to estimate complete hidden-space output quality.

Only active baseline-pruned channels are eligible for rescue:
 
$$
\mathcal R_e
=
\mathcal P_e^A\cap\mathcal A_e.
$$

Sort $\mathcal R_e$ by descending $s^{PP}_{e,c}$, then by descending Stable-AIMER score, then by original channel index. The resulting order is $\pi_e^{PP}$.

---

## 7. Stage E: Dynamic 3%-8% Deterministic Swap

### 7.1 Dynamic fraction

NAPS-v2 maps native routed pseudo-token count $M_e$ to a fraction of the original width $C$:

$$
\rho_e^{\mathrm{swap}}
=
\begin{cases}
0.03,&0\le M_e\le2,\\
0.04,&3\le M_e\le4,\\
0.05,&5\le M_e\le8,\\
0.06,&9\le M_e\le16,\\
0.07,&17\le M_e\le32,\\
0.08,&M_e>32.
\end{cases}
$$

This mapping is centered on the previously validated 5% operating point. Most experts remain in the 3%-5% range, while only the routed long tail reaches 7%-8%.

The requested number of swaps is:

$$
B_e^{\mathrm{req}}
=
\operatorname{round}(\rho_e^{\mathrm{swap}}C).
$$

The feasible number is:

$$
B_e
=
\min\left(
B_e^{\mathrm{req}},
|\mathcal R_e|,
|\mathcal K_e^A\cap\mathcal A_e|
\right).
$$

For current widths this gives:

| Fraction | Qwen3 $C=768$ | Qwen3.6 $C=512$ |
| ---: | ---: | ---: |
| 3% | 23 | 15 |
| 4% | 31 | 20 |
| 5% | 38 | 26 |
| 6% | 46 | 31 |
| 7% | 54 | 36 |
| 8% | 61 | 41 |

### 7.2 Swap operation

Take the highest-ranked $B_e$ rescue channels:

$$
\mathcal S_e^{\mathrm{in}}=\pi_e^{PP}[:B_e].
$$

Take the lowest-ranked $B_e$ active channels from the Stable-AIMER keep set:

$$
\mathcal S_e^{\mathrm{out}}
=
\operatorname{Tail}_{B_e}
(\mathcal K_e^A\cap\mathcal A_e).
$$

The final keep set is:

$$
\boxed{
\mathcal K_e
=
(\mathcal K_e^A\setminus\mathcal S_e^{\mathrm{out}})
\cup
\mathcal S_e^{\mathrm{in}}
}
$$

NAPS-v2 does **not** greedily test each swap with pseudo-output reconstruction loss. Earlier direct replacement experiments provide the empirical prior, while the 3%-8% bound controls intervention size.

The operation must be deterministic. No PP channel may be duplicated, no zero channel may be swapped in, and exactly $K$ channels must remain.

### 7.3 Required swap diagnostics

Each expert record must include:

- `swap_fraction`;
- `requested_swaps`;
- `feasible_swaps`;
- `actual_swaps`;
- `swap_in_channels`;
- `swap_out_channels`;
- PP and Stable-AIMER scores for both sets;
- overlap between $\mathcal K_e^A$ and $\mathcal K_e$;
- whether any requested swap was reduced by capacity or zero handling.

---

## 8. Stage F: PP-Output Coverage

PP-output coverage is not a channel-selection score. It estimates how much of an expert's structural output magnitude remains after applying the final mask.

For active channel $c$ define:

$$
r_{e,c}
=
\operatorname{Mean}_{q\in Q_e^C}
|h_{e,c}(q)|
\cdot
\|d_{e,c}\|_2.
$$

Define total active output mass:

$$
R_e^{\mathrm{all}}
=
\sum_{c\in\mathcal A_e}r_{e,c}.
$$

Retained output mass is:

$$
R_e^{\mathrm{keep}}
=
\sum_{c\in\mathcal K_e\cap\mathcal A_e}r_{e,c}.
$$

The PP-output coverage is:

$$
\boxed{
c_e
=
\frac{R_e^{\mathrm{keep}}}
{R_e^{\mathrm{all}}+\epsilon_C}
}
$$

and the uncovered mass is:

$$
u_e=1-c_e.
$$

Effective-zero fillers are excluded from both numerator and denominator.

Coverage answers:

> Approximately how much output-bearing channel mass remains in the final retained set?

It does not claim to equal real-data explained variance or downstream quality.

Required diagnostics include:

- `output_coverage` $c_e$;
- `uncovered_output_mass` $u_e$;
- coverage before and after PP swap;
- retained and pruned output-mass sums;
- output-mass quantiles within retained and pruned sets;
- native probe count and anchor status used to interpret confidence.

---

## 9. Stage G: Coverage-Guided Expert Residual Compensation

### 9.1 Purpose

NAPS-v2 compensation targets the output contribution of selected channels from the **complete final active prune set**:

$$
\mathcal P_e^{\mathrm{active}}
=
\mathcal A_e\setminus\mathcal K_e.
$$

It is not limited to swap-displaced channels.

The compensation mechanism fits pruned channel responses in the subspace spanned by retained channel responses, then writes the corresponding output contribution into retained `down_proj` columns.

### 9.2 Compensation target selection

Rank final pruned active channels by $r_{e,c}$ descending. Let:

$$
\mathcal T_e(P)
=
\operatorname{TopP}_{c\in\mathcal P_e^{\mathrm{active}}}(r_{e,c})
$$

be the smallest prefix whose cumulative output mass reaches fraction $P$ of total pruned output mass, subject to a hard channel cap $K_{\max}^{\mathrm{comp}}$.

The preregistered initial policy is:

$$
P=0.80,
$$

with:

$$
K_{\max}^{\mathrm{comp}}
=
\begin{cases}
32,&\text{B9},\\
64,&\text{B6}.
\end{cases}
$$

This policy compensates the dominant pruned output mass without forcing every low-impact pruned channel into the regression.

Required ablations are $P\in\{0.50,0.80,0.95\}$ and channel caps $\{16,32,64\}$ where structurally feasible.

### 9.3 Weighted ridge projection

Let $Q_e^C$ contain $m$ coverage probes. Construct:

$$
H_R\in\mathbb R^{m\times|\mathcal K_e\cap\mathcal A_e|}
$$

from retained active channel responses, and:

$$
H_P\in\mathbb R^{m\times|\mathcal T_e|}
$$

from selected pruned target responses.

Use a diagonal probe-weight matrix $W_e$. The default policy uses uniform weights:

$$
W_e=\frac{1}{m}I.
$$

Solve the ridge objective:

$$
A_e^\star
=
\arg\min_A
\left\|
W_e^{1/2}(H_P-H_RA)
\right\|_F^2
+
\lambda_e\|A\|_F^2.
$$

The closed-form solution is:

$$
A_e^\star
=
(H_R^\top W_eH_R+\lambda_e I)^{-1}
H_R^\top W_eH_P.
$$

The ridge strength is scale-adaptive:

$$
\lambda_e
=
\lambda_0
\frac{\operatorname{tr}(H_R^\top W_eH_R)}
{\max(1,|\mathcal K_e\cap\mathcal A_e|)}.
$$

The initial default is:

$$
\lambda_0=10^{-3}.
$$

Implementation must use a stable linear solve, not an explicit matrix inverse. If the solve is non-finite or fails, the expert falls back to NAPS-v2-Mask.

### 9.4 Sparse write-back

A fully dense $A_e^\star$ could modify every retained `down_proj` column. NAPS-v2 therefore sparsifies each target column of $A_e^\star$ by keeping the largest coefficients by absolute value.

The initial default is:

$$
s=2
$$

representatives per compensated pruned channel. Required ablations use $s\in\{1,2,4\}$.

Let $\widehat A_e$ be the sparsified coefficient matrix. If $D_P$ contains `down_proj` columns for target channels, update retained active columns by:

$$
\boxed{
D_R'
=
D_R+D_P\widehat A_e^\top
}
$$

Only `down_proj` is modified. Gate and up weights of retained channels remain unchanged.

### 9.5 Coverage and confidence gating

Expert compensation is attempted only when all conditions hold:

1. $\mathcal T_e$ is non-empty;
2. uncovered output mass exceeds a minimum:

   $$
   u_e\ge u_{\min}=0.05;
   $$

3. coverage probe count is non-zero, which is guaranteed by the self anchor;
4. the ridge system and update are finite.

The number of native probes does not hard-disable compensation. It controls the maximum compensation strength:

$$
\delta_{\max,e}
=
\begin{cases}
0.02,&M_e\le2,\\
0.03,&3\le M_e\le8,\\
0.04,&9\le M_e\le16,\\
0.05,&M_e>16.
\end{cases}
$$

where the expert update ratio is:

$$
\delta_e
=
\frac{\|D_R'-D_R\|_F}
{\|D_R\|_F+\epsilon_D}.
$$

If $\delta_e>\delta_{\max,e}$, scale the update rather than immediately reject it:

$$
D_R'
\leftarrow
D_R
+
\frac{\delta_{\max,e}}{\delta_e}
(D_R'-D_R).
$$

This preserves the update direction while enforcing a probe-count-dependent trust region.

### 9.6 Expert-level validation and fallback

NAPS-v2 must report both compensated and mask pseudo-space diagnostics, but pseudo reconstruction is not used to accept or reject individual swaps.

For the complete expert compensation update, compute:

- mask and compensated uniform output loss over $Q_e^C$;
- mask and compensated native-only output loss over $Q_e^N$ when non-empty;
- update ratio before and after trust-region scaling;
- recovered selected-pruned output mass.

The default v2 export applies any finite update that satisfies the trust region. A strict validation ablation additionally requires compensated coverage-probe output loss not to exceed mask loss. This distinction must be explicit in artifact metadata:

- `expert_comp_policy=trust_region` for the default;
- `expert_comp_policy=strict_proxy_nonregression` for the ablation.

If the solve fails, produces non-finite values, or violates structural invariants, the entire expert falls back to NAPS-v2-Mask.

---

## 10. Exported Variants

### 10.0 PuzzleMoE-inspired pairwise compensation (experimental)

PuzzleMoE contributes a principle rather than a drop-in replacement for NAPS-v2:

$$
\boxed{
	ext{merge only magnitude-similar elements; preserve expert-specific salient elements}
}
$$

Its activation-weight saliency mask requires a small unlabeled calibration set. Router-derived pseudo tokens are useful for structural ranking, but are not a substitute for real hidden-state statistics when deciding whether two experts share a parameter.

The experimental `PuzzleComp` prototype applies this principle to B6-equivalent packed storage without changing the default Mask or ExpertComp artifacts. For a pair of experts with source width $C$ and retained width $K$, choose a small reserve $R$ from the low-priority end of each fixed AIMER ranking:

- retain $K-R$ protected core channels independently for each expert;
- select $2R$ shared original channel positions outside both protected cores;
- compute activation-weight saliency separately for gate, up, and down weights;
- store one shared magnitude tensor plus two masks and sign patterns for the $2R$ shared positions;
- reconstruct each expert with its own mask/sign pattern;
- accept the pair only when both coverage-probe and native-probe output loss are non-increasing.

The pair storage accounting is:

$$
2(K-R)+2R=2K.
$$

Thus the packed pair has the same logical channel-storage budget as two independent K-width experts. Each shared slot reconstructs one channel for each member of the pair, so a standard-vLLM materialization has width $K+R$ per expert. This intentionally increases physical compute in the correctness-first quality evaluation; a custom decode-GEMM path would be required to realize the packed storage/runtime benefit.

This differs from the existing ExpertComp path. ExpertComp fits the complete pruned output into the same expert's retained `down_proj` columns. The output-recoverability analysis shows that this residual is poorly represented by that local subspace (`R^2` about `4.29%` for Qwen3 and `2.82%` for Qwen3.6), so increasing its regression rank or trust region is not the intended PuzzleComp direction.

The prototype is calibration-required and experimental. It must not be used with arbitrary adjacent expert pairing: pairing is a separate layer-level search problem, and effective-zero channels must be excluded before residual matching. The default NAPS-v2 builder remains unchanged.

### 10.1 NAPS-v2-Mask

Uses:

- effective-zero handling;
- Stable-AIMER baseline;
- dual-semantics probe bank;
- dynamic 3%-8% deterministic PP swap.

It exports the final fixed-width channel set without compensation.

### 10.2 NAPS-v2-ExpertComp

Uses the identical mask and additionally applies:

- PP-output coverage;
- full-prune-set target ranking;
- Top-P/capped target selection;
- sparse weighted ridge projection;
- probe-count-dependent expert trust region.

Checkpoint identity must be independently verified. Results may be reused between Mask and ExpertComp only if all exported model shards are byte-identical.

---

## 11. Determinism and Audit Requirements

All ranking and selection operations must be deterministic:

- stable descending sort;
- original channel index as final tie-breaker;
- fixed pseudo-token ordering by expert index;
- no stochastic probe generation;
- FP32 construction;
- explicit seed recorded even though construction is deterministic.

Each expert diagnostic record must contain at least:

```text
layer_id
expert_id
source_width
retained_width
effective_zero_count
forced_zero_retained
native_probe_count
self_naturally_routed
self_native_rank
anchor_added
coverage_probe_count
swap_fraction
requested_swaps
actual_swaps
swap_in_channels
swap_out_channels
coverage_before_swap
coverage_after_swap
pruned_output_mass
compensation_target_count
compensation_target_mass_fraction
ridge_lambda
representatives_per_target
update_ratio_raw
update_ratio_final
mask_uniform_loss
compensated_uniform_loss
mask_native_loss
compensated_native_loss
fallback_reason
```

Layer-level artifacts must preserve native pseudo-token Top-$k$ expert IDs and weights so self-routing and expert coverage can be independently recomputed.

---

## 12. Required Invariants and Tests

### 12.1 Probe invariants

1. Exactly one RMSNorm is applied when constructing each pseudo token.
2. Native routing is calculated without a second normalization.
3. `native_probe_count` matches the number of native Top-$k$ assignments to the expert.
4. `coverage_probe_count >= 1` for every expert.
5. Added self anchors are never counted as native routes.

### 12.2 Selection invariants

1. Final width is exactly $K$ for every expert.
2. No effective-zero channel is swapped in.
3. `actual_swaps` equals both the number of swap-in and swap-out channels.
4. Swap-in channels come only from the Stable-AIMER baseline prune set.
5. Swap-out channels come only from the active suffix of the Stable-AIMER baseline keep set.
6. NAPS-v2-Mask and NAPS-v2-ExpertComp use identical retained indices.

### 12.3 Compensation invariants

1. Compensation targets come from the complete final active prune set.
2. Effective-zero channels never contribute to compensation.
3. Only retained `down_proj` columns are modified.
4. Gate/up tensors and router tensors are byte-identical to NAPS-v2-Mask except for structural channel slicing already implied by the mask.
5. Final update ratio does not exceed $\delta_{\max,e}$.
6. Any non-finite solve or update falls back to Mask.
7. Exported tensor shapes and checkpoint index remain valid.

### 12.4 Synthetic tests

At minimum, tests must cover:

- self naturally routed and self not naturally routed;
- zero native probes with an added self anchor;
- all dynamic swap-count buckets;
- insufficient active rescue channels;
- effective-zero overflow with forced fillers;
- deterministic PP ties;
- ridge solve on exact, noisy, rank-deficient, and empty residual systems;
- Top-P target selection and channel cap;
- sparsification with $s=1,2,4$;
- trust-region scaling;
- complete expert fallback on non-finite values.

---

## 13. Preregistered Ablation Matrix

NAPS-v2 must separate gains from selection and compensation.

| ID | Swap | Coverage probes | Compensation | Purpose |
| --- | --- | --- | --- | --- |
| V1 | NAPS-v1 evidence-gated | Native only | Displaced-only pair merge | Historical baseline |
| V2-M | Dynamic 3%-8% | Native + self anchor | None | Isolate swap gain |
| V2-C1 | Dynamic 3%-8% | Native + self anchor | Top-P sparse expert compensation | Main v2 method |
| V2-C2 | Dynamic 3%-8% | Native + self anchor | Strict proxy-nonregression compensation | Validation-policy ablation |
| V2-F5 | Fixed 5% | Native + self anchor | None | Compare dynamic mapping with prior optimum |
| V2-N | Dynamic 3%-8% | Native only where available | None | Measure anchor effect on PP selection |

The first implementation round should prioritize:

1. Qwen3 B6 V2-M versus current NAPS-Mask;
2. Qwen3 B6 V2-C1 versus V2-M;
3. Qwen3 B9 V2-M and V2-C1;
4. Qwen3.6 B6 V2-M and V2-C1;
5. Qwen3.6 B9 only after the B6 mechanism is understood.

No B9/B6 result may be inferred from another width. Mask and ExpertComp require separate evaluation unless byte identity is proven.

---

## 14. Evaluation Protocol

All downstream comparisons use the frozen `full6_v1` protocol:

| Dataset | Exact samples |
| --- | ---: |
| ARC | 3,548 |
| HellaSwag | 10,042 |
| WinoGrande | 1,267 |
| GSM8K | 1,319 |
| MATH-500 | 500 |
| MMLU | 14,042 |

Common settings:

- seed 42;
- temperature 0;
- `do_sample=false`;
- thinking disabled;
- evaluation batch size 16;
- independent result directories per exported checkpoint.

The Full6 Macro is the unweighted arithmetic mean of six scores and may be reported only after every report JSON passes exact-count validation.

In addition to downstream scores, each experiment report must include:

- total and per-expert swap counts;
- swap-fraction distribution;
- coverage before and after swap;
- compensation target counts and output-mass fractions;
- expert update-ratio distribution;
- fallback counts and reasons;
- number of affected layers and experts;
- exact checkpoint shard differences relative to Mask.

---

## 15. Interpretation Boundaries

NAPS-v2 makes the following limited claims:

1. Router weights provide useful calibration-free structural probes.
2. They are not assumed to be real-token centroids.
3. Native routing preserves router competition geometry but does not guarantee expert coverage.
4. Self anchors guarantee structural coverage without being relabeled as native routes.
5. Stable-AIMER and PP scores decide channel selection without `down_proj` output norms.
6. PP-output coverage uses `down_proj` norms only to estimate retained hidden-space output mass.
7. Expert compensation approximates selected pruned responses in the retained response subspace; it does not claim exact recovery on real data.
8. Pseudo-space metrics are diagnostics, not substitutes for downstream evaluation.

The main NAPS-v2 hypothesis is:

> Stable-AIMER provides a reliable global mask; bounded dynamic PP replacement recovers channels emphasized by router-derived structural probes; PP-output coverage identifies experts with meaningful pruned output mass; and sparse trust-region expert compensation can recover part of that mass without changing fixed width or coupling output norms back into channel selection.

---

## 16. Implementation Plan

NAPS-v2 should be implemented alongside NAPS-v1, not by silently changing existing artifacts.

Recommended files:

```text
NAPS_v2/naps_v2_core.py
NAPS_v2/model_adapter.py
NAPS_v2/build_naps_v2_artifacts.py
NAPS_v2/export_naps_v2_checkpoint.py
NAPS_v2/test_naps_v2_core.py
NAPS_v2/test_naps_v2_export.py
```

Artifact directories and checkpoint names must include `naps_v2` and the exact pruning ratio:

```text
NAPS_v2/experiments/qwen3_b6_naps_v2
NAPS_v2/experiments/qwen3_b9_naps_v2
NAPS_v2/experiments/qwen36_b6_naps_v2
NAPS_v2/experiments/qwen36_b9_naps_v2

NAPS_v2/checkpoints/qwen3_b6_naps_v2_mask
NAPS_v2/checkpoints/qwen3_b6_naps_v2_expertcomp
```

Run labels must use the actual structural pruning ratio:

- B9 uses `25`;
- B6 uses `50`.

The implementation is complete only after:

1. focused synthetic tests pass;
2. Qwen3 and Qwen3.6 real-tensor layer smoke tests pass;
3. all artifacts satisfy deterministic and structural invariants;
4. exported checkpoints pass index, shard, manifest, tensor-shape, and generation smoke validation;
5. Mask and ExpertComp are evaluated independently under `full6_v1`.
