# Results analysis

This note records the questions raised after `Results.md` was compiled, the checks run on the checkpoints and generations, and the conclusions.

Random and Magnitude are `full8_v1` CalibrationFree runs:

- Random: `*_Random_202608191559_42` (DeepSeek 50%/25% uses `*_202608191826_42`)
- Magnitude: `*_Magnitude_202608200100_42`

Wanda and AIMERChannel scores in `Results.md` use the same export/eval path (uniform routed-expert channel slice → vLLM → EvalScope). Wanda is `WikiText128x2048`; AIMERChannel is CalibrationFree. DeepSeek writes `moe_intermediate_size=K` and `shared_expert_intermediate_size=2816`; vLLM on this box reads the latter via `deepseek_shared_width.patch`.

---

## Questions

1. Is the Magnitude implementation wrong? Why is it generally worse than Random at the same sparsity?
2. DeepSeek-V2-Lite-Chat at **50%** is near zero for **both** Random and Magnitude. Is that a loading/export bug?
3. DeepSeek at **25% Magnitude** is also near zero (mean retained 9.0%), while **25% Random** still holds 63.5%. That gap is too large to ignore.
4. DeepSeek **Wanda 25%** also collapses (mean retained 3.5% on the finished datasets), even though Wanda uses a frozen WikiText-2 train cache. Why does calibration not save it?

---

## What the table actually shows

Mean retained % vs Dense:

| Model | 25% Random | 25% Magnitude | 50% Random | 50% Magnitude |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B-Instruct-2507 | 87.8 | 77.5 | 46.1 | 26.6 |
| Gemma4-26B-A4B-it | 79.1 | 71.0 | 30.2 | 33.7 |
| Qwen3.6-35B-A3B | 92.0 | 89.9 | 57.5 | 55.5 |
| DeepSeek-V2-Lite-Chat | 63.5 | **9.0** | **0.1** | **6.0** |

On Qwen3 / Gemma / Qwen3.6 the drop is concentrated on **GSM8K, MATH-500, HumanEval, MBPP**. Multiple-choice (ARC, MMLU, sometimes HellaSwag) stays much closer. Magnitude is not uniformly worse: Gemma 50% WinoGrande is 0.545 for Magnitude vs 0.050 for Random.

DeepSeek 25% Magnitude and both 50% runs are not “a bit worse”. They are collapse-level scores.

---

## Implementation checks (not a silent export bug)

The Magnitude exporter is the same slice logic as Random (`index_select` on coupled gate/up/down or packed `[gate; up]`). The following were verified on the **DeepSeek Magnitude 25%** checkpoint against the source model and `magnitude_rankings.pt`:

- Layout: `gate/up = [C, H]`, `down = [H, C]`. Exported widths are 1056 (25%) and 704 (50%).
- Shared expert `gate_proj` remains `[2816, 2048]` and equals the source tensor. Config has `shared_expert_intermediate_size=2816`.
- Dense layer 0 is not pruned. Changed routed tensors: `26 layers × 64 experts × 3 = 4992`.
- Ranking is **descending** L2. Recomputing coupled FP32 magnitude from source weights matches the cache (max abs error 0).
- Exported expert 0 equals `source.index_select(kept_indices)` on gate/up/down.
- Gemma packed layout matches vLLM: `gate_up_proj` is `[E, 2I, H]`, first `I` gate, second `I` up.

So these are **not** the failure mode from the earlier DeepSeek Random run (shared width tied to `moe_intermediate_size * n_shared_experts`). Random 25% on the same vLLM stack produces coherent English and ~1284/1319 GSM8K answers with `\boxed{}`, which also shows the DeepSeek load path works.

---

## Score distributions (why Magnitude ≠ Random)

Coupled channel L2 on layer 0 (DeepSeek: first MoE layer 1):

| Model | L2 shape | Energy dropped at 25% prune | Implication |
| --- | --- | ---: | --- |
| Qwen3.6 | Many channels are **exactly 0** | ~1.5% | Magnitude mostly deletes dead channels → close to Random |
| Qwen3 | A tail near 0 (`p01 ≈ 0.01`) | ~17% | Some unused channels, 25% still usable, 50% hurts more |
| Gemma4 | Fairly flat (`mean 2.90 ± 0.30`) | ~22% | Weak ranking signal |
| DeepSeek | **Almost uniform** (`mean 2.16 ± 0.11`, `p01/p99 ≈ 1.94/2.52`) | ~23% (≈ channel fraction) | Top-K L2 is not pruning “junk”; it is a slightly biased subset |

Qwen3.6 is the positive control: when dead channels exist, Magnitude does drop them. DeepSeek has no such tail, so “keep largest L2” is a poor importance proxy.

---

## Generation evidence (collapse vs degradation)

GSM8K predictions:

- **DeepSeek Random 25%**: fluent English, ~1284 boxed answers. Model is alive.
- **DeepSeek Magnitude 25%**: CJK / replacement characters / garbage; ~25 boxed, ~81 empty. Tokenizer-level collapse, not a scoring-script artifact.
- **DeepSeek Random 50%**: many empty or newline-only completions. Collapse.
- **DeepSeek Magnitude 50%**: truncated junk, markdown hashes. Collapse.
- **Qwen3 Magnitude 50%**: still English; ~168 boxed vs Random ~859. Degraded, not DeepSeek-style garbage.

Dense DeepSeek in `Results.md` is healthy (e.g. GSM8K 0.7058). Collapse appears only after pruning.

---

## Analysis by question

### 1. Why is Magnitude often worse than Random?

Weight-only coupled L2 is not task importance. The protocol also **forces the same K on every routed expert**, so Magnitude cannot steal width from unused experts.

That matches the dataset split: knowledge/choice holds up; math and code generation fall first. It is the expected weakness of this baseline (same family of results as calibration-free magnitude vs random / Wanda-style methods), not inverted ranking. Ranking direction, cache, and slices were checked.

Gemma 50% shows the other side: Random can also fail a dataset (WinoGrande 0.05) while Magnitude does not. Do not treat Random as a strict upper bound on every cell.

### 2. DeepSeek 50% both methods

Lite-Chat is the smallest of the four (routed width 1408, 64 experts, 16B-class). Cutting routed FFN to 704 destroys generation for **both** selectors. Random 25% already proved export + vLLM + shared-width split can serve this model. Treat 50% DeepSeek scores as **collapse**, not as a 0.1% vs 6.0% method ranking.

### 3. DeepSeek 25% Magnitude vs Random

Same K=1056, same exporter, same server stack. Random lives, Magnitude dies. After the slice audit, the remaining difference is **which 352 channels are removed** (and the order they are packed).

Because DeepSeek L2 is nearly flat, Magnitude 25% is “drop the slightly smaller 352 channels”. Those channels are still necessary for this model: Dense keeps both sides and works; dropping the small-L2 side unbalances expert outputs and the residual stream, which shows up as CJK/garbage. That is a **selection-policy failure on this small, non-sparse-magnitude model**, not a wrong `moe_intermediate_size` or shared-expert cut.

### 4. DeepSeek Wanda 25% vs calibration

Wanda 25% on the other three models is healthy (mean retained 88.9% / 85.7% / 90.9%). On DeepSeek it is collapse: ARC 0.044, HellaSwag 0.028, WinoGrande 0.022, GSM8K 0.014, MATH-500 0.002 (MMLU / HumanEval / MBPP were still running when this was written). AIMERChannel 25% on the same model holds 65.0% with **no** calibration data. Random 25% holds 63.5%. So the DeepSeek load path and the 25% width cut are survivable; Wanda’s **kept-channel set** is not.

The Wanda artifact is not a missed cache. `wanda_25pct_per_layer.json` records WikiText-2 train 128×2048, native router `mass` weighting, `unseen_experts: 0` (all 64 routed experts were observed), skip dense layer 0, shared experts unpruned, `K=1056`. 25% is cloned from the 50% ranking (same permutation, shorter prefix). Qwen3 Wanda still works with 81 unseen experts falling back to weight-L2; DeepSeek has better coverage and still dies. Coverage is not the failure.

Structured Wanda scores a coupled SwiGLU channel as

```
S(c) = sqrt(
    ||W_gate[c,:] ⊙ RMS_p(x)||_2^2
  + ||W_up[c,:]   ⊙ RMS_p(x)||_2^2
  + RMS_p(z_c)^2 ||W_down[:,c]||_2^2
)
```

`RMS_p(x)` is the expert’s input second moment over hidden dim. It is **shared by every channel of that expert**. After RMSNorm it is usually flat, so the gate/up terms collapse toward weight L2. The only per-channel activation is `RMS_p(z_c)` on the gated intermediate. If that tracks weight size, Wanda **reinforces** Magnitude instead of correcting it.

Magnitude / Random shards had already been deleted, so the Magnitude keep-set was recomputed as coupled FP32 L2 Top-K on the source tensors (the ranking the Magnitude exporter uses). Wanda and AIMER keep-sets were recovered by matching exported `gate_proj` rows to the source (layers 1, 8, 13, 20, 26 × experts 0, 16, 32, 48, 63). AIMER recovered keep-sets match the theoretical `RMS(w)/MeanAbs(w)` Top-K at 100%.

Kept-set overlap at K=1056 (independent 75% subsets would overlap at 0.75; Jaccard chance ≈ 0.60):

| Pair | Mean overlap | Min overlap | Jaccard |
| --- | ---: | ---: | ---: |
| Wanda ∩ Magnitude L2 | **0.910** | 0.806 | 0.838 |
| Wanda ∩ AIMER | 0.746 | 0.732 | 0.595 |
| Magnitude L2 ∩ AIMER | 0.747 | 0.723 | 0.596 |

Mean coupled L2 of kept vs dropped channels, and mean L2-rank of kept channels (0 = largest L2; random 75% keep ≈ 703.5; Magnitude Top-K = 527.5):

| Method | Mean L2 kept | Mean L2 dropped | Mean L2-rank of kept |
| --- | ---: | ---: | ---: |
| Magnitude L2 | 2.321 | 2.169 | 527.5 |
| Wanda | 2.318 | 2.180 | **558.1** |
| AIMER | 2.284 | 2.281 | **703.4** |

AIMER is orthogonal to L2 (rank matches random). Wanda is a slight perturbation of Magnitude. Layer-1 expert-0 L2 is still almost uniform (`mean 2.06 ± 0.06`, `cv ≈ 3%`): no dead-channel tail for either criterion to harvest.

Generation matches Magnitude, not Random: Wanda 25% ARC mean output length 1706 tokens, median **2048** (`max_tokens`), ~75% of samples hit the cap. Completions are blank lines, `</html>`, and garbage. Dense DeepSeek ARC is median 5 tokens / 0.44 s; Qwen3 Random 25% ARC is mean 5.1 tokens. The slow Wanda eval is **runaway generation**, not MLA kernel cost. Per-token rate is fine (~130 tok/s); requests emit ~300× more tokens.

WikiText vs chat domain mismatch is **not** the main story. A domain-shifted Wanda ranking would diverge from weight-only L2. It does not. Calibration ran; it did not add a ranking signal. The protocol also **forces the same K on every expert**, so router mass cannot reallocate width — only within-expert channel order can change, and that order stayed Magnitude-like.

---

## Conclusions

1. **Magnitude code is consistent with the spec.** Descending coupled L2, uniform K, packed/separate layouts, DeepSeek shared width split, and skip-dense-layer-0 are all behaving as intended.
2. **Magnitude underperforming Random on Qwen3/Gemma (especially math/code) is a method result**, not a hidden invert. Qwen3.6 staying close to Random is explained by true zero-L2 channels.
3. **DeepSeek 50% (Random and Magnitude) is model collapse** at this sparsity. Do not over-interpret those rows. Wanda 50% (7.5%) and AIMERChannel 50% (13.9%) are the same collapse band.
4. **DeepSeek 25% Magnitude is also collapse**, while 25% Random is not. Loading is fine; keeping high-L2 channels is actively harmful when L2 has no dead-channel tail.
5. **DeepSeek Wanda 25% is the same selection failure, not a broken calibrator.** All 64 experts were seen; the keep-set overlaps Magnitude L2 at 91% and is chance-level vs AIMER. Calibration-free AIMER 25% and Random 25% both live. Wanda 25% on Qwen3 / Gemma / Qwen3.6 living shows the Wanda pipeline is fine on models that have an energy/activation tail.
6. **Calibration does not imply a better structured ranking.** On this small, nearly uniform-width expert, `RMS(x)` is shared across channels and `RMS(z)` does not reorder vs L2, so Wanda ≈ keep-largest-L2. AIMER’s `RMS/MeanAbs` is L2-orthogonal and survives without data.
7. **No checkpoint re-export is required** to explain the table. A useful optional ablation (not a bugfix) would be DeepSeek 25% with the **smallest-K** prefix of the Magnitude or Wanda ranking, or a random subset of that Top-K, to test whether large-L2 channels are the harmful set.

---

## Result paths

```
/home/xinpeigao/evalscope/results/
  <Model>_{25,50}_vllm_CalibrationFree_full8_v1_Random_<ts>_42
  <Model>_{25,50}_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42
  <Model>_{25,50}_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42
  <Model>_{25,50}_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42

/data/xinpeigao/evalscope_results/_artifacts/{random,magnitude,aimer_channel,wanda}/<qwen3|gemma4|qwen36|deepseek>/
  checkpoint_25  checkpoint_50
```
