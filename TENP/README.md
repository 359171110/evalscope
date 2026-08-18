# ENP / TENP Reproduction

This directory contains a from-paper reproduction of the two algorithms in
*TENP: Trapezoidal Expert Neuron Pruning for Mixture-of-Experts*:

- **ENP** ranks each routed expert's SwiGLU channels by the paper's signed
  output projection and keeps a common width in every routed expert.
- **TENP** additionally ranks routed experts by final Top-k gate weight,
  output magnitude, and input-output direction change. A deterministic linear
  trapezoid keeps more full-width experts in deeper layers; all remaining
  experts stay routable but use narrower ENP-ranked channel prefixes.

The standalone equations and small-tensor reference functions live in
`enp_tenp_reference.py`. Framework integration lives in:

```text
static_moe_prunning/code/src/enp_tenp.py
static_moe_prunning/code/scripts/build_enp_tenp_profiles.py
```

The integration reuses the existing `static_expert_profile` EvalScope ModelAPI.
It does not delete experts, alter Router parameters, or mask Router candidates.
The current runtime performs physical channel slicing during expert execution;
it does not export a compact Hugging Face checkpoint.

## Reproduction Protocol

The default profile calibration uses the frozen train-only 512 x 1024 mixed
token cache from the static pruning framework. The paper-level defaults are:

```text
routed expert parameter retention: 0.60
TENP full expert ratio: 0.30
trapezoid shallow/deep weights: 1.0 / 2.0
runtime channel block size: 64
```

The 524,288-token calibration run keeps `min_tokens_per_expert=32` as an audit
threshold. Nonzero experts below that threshold are accepted only through the
launcher's explicit override and their exact route counts are stored in both
statistics and channel artifacts. Experts with zero routed tokens are kept full
width through an explicit policy and listed in the artifacts; TENP counts them
inside its fixed full-expert and total block budgets.

The evaluation protocol follows the corrected quick9 dataset selection:

| Dataset | Limit | Total |
| --- | ---: | ---: |
| ARC-Challenge + ARC-Easy | 300 per subset | 600 |
| HellaSwag | 1000 | 1000 |
| MMLU | 10 per subject | 570 |
| WinoGrande | 400 | 400 |
| GSM8K | 128 | 128 |
| MATH-500 | 20 per Level | 100 |

Generation uses `max_tokens=4096`, greedy decoding, seed 42, batch size 1,
and thinking disabled.

## Commands

Inspect the frozen paths and protocol without loading the model:

```bash
bash TENP/run_qwen3_enp_tenp_reproduction.sh dry-run
```

Build the shared statistics, signed-projection channel cache, and Dense/ENP/TENP
profiles. The Dense profile is emitted by the shared builder but is not evaluated:

```bash
BUILD_GPU=1 bash TENP/run_qwen3_enp_tenp_reproduction.sh build
```

Run artifact/model preflight on the available GPUs, then launch all six datasets
for ENP and TENP in parallel by method. Dense is omitted because its baseline
results already exist:

```bash
GPUS_CSV=1,3 bash TENP/run_qwen3_enp_tenp_reproduction.sh preflight
GPUS_CSV=1,3 bash TENP/run_qwen3_enp_tenp_reproduction.sh eval
```

Only use GPUs that are actually idle. The launcher accepts any non-negative
physical GPU index so the run can use whichever devices are currently free.

## Paper Ambiguities

The paper does not publish its per-layer trapezoid counts. This reproduction
uses the design document's linear depth weights and largest-remainder integer
allocation. At 64-channel runtime granularity, an indivisible remaining budget
is distributed by stable expert-score order, so narrow experts can differ by at
most one block while the total routed-expert budget remains exact.