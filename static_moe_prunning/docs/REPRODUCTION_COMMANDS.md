# Static pruning reproduction commands

All commands below expose only physical GPUs 4–7. GPU0–3 are forbidden.

Exception: the user-authorized downstream screening matrix launched by
`code/scripts/run_downstream_matrix.sh` accepts physical GPUs 0–5. This exception
applies only to that matrix launcher; the formal PPL commands below retain the
4–7 protocol unless an experiment explicitly changes it.

## Prerequisites

Run from a clean clone and keep all model/data artifacts outside Git:

```bash
cd /path/to/moe_prune_v4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export PYTHON_BIN="$(command -v python)"
```

The completed experiments used Python 3.10 and PyTorch `2.10.0+cu128`.
Install the CUDA-specific PyTorch wheel appropriate for the target server before
the remaining requirements when the default package index does not provide it.

Model weights, dataset shards, token caches, channel-calibration tensors,
profiles, logs, and results are external/generated artifacts and are
intentionally absent from Git. Formal runs must preserve the recorded split,
hashes, offsets, token counts, and frozen-before-evaluation metadata.

The repository includes four frozen AMP/AIMER prior pairs under
`experiments/calibration/`. Verify them before use:

```bash
cd experiments/calibration
sha256sum -c FROZEN_PRIORS.sha256
cd ../..
```

| Checkpoint identity | Shape per table | AMP cache | AIMER cache |
|---|---:|---|---|
| Qwen1.5-MoE-A2.7B | 24 x 60 | `qwen15_moe_priors_20260728/amp_scores.pt` | `qwen15_moe_priors_20260728/aimer_scores.pt` |
| Qwen3-30B-A3B | 48 x 128 | `qwen3_base_priors_20260728/amp_scores.pt` | `qwen3_base_priors_20260728/aimer_scores.pt` |
| Qwen3-30B-A3B-Instruct-2507 | 48 x 128 | `static_expert_priors_20260728/amp_scores.pt` | `static_expert_priors_20260728/aimer_scores.pt` |
| Qwen3.5-35B-A3B | 40 x 256 | `qwen35_prospective_priors_20260728/amp_scores.pt` | `qwen35_prospective_priors_20260728/aimer_scores.pt` |

These tables are frozen method inputs, not experiment results or model weights.
Their exact AMP/AIMER formulas and builders are not included in this snapshot,
so the tables cannot yet be regenerated from the base checkpoints. Downstream
profile builders can consume them with `--amp-score-cache` and
`--aimer-score-cache`.

Model loading, physical expert-prefix runtime, AMP/AIMER prior generation, PPL
evaluation, and the EvalScope ModelAPI are self-contained in this repository.
The publication audit rejects machine-specific home paths and reports any
reintroduced historical compatibility imports.

## Current best method: calibration artifacts → profile → full PPL

```bash
MODEL_PATH="$MODEL_PATH" GPU=4 PYTHON_BIN="$PYTHON_BIN" \
	bash scripts/run_our_best_end_to_end.sh
```

This rebuilds Frontier Committee Regret from frozen train-only calibration
artifacts, performs exact structural and train-routed-compute allocation, then
runs the complete WikiText-2 test protocol: 114 windows, 233,368 tokens,
sequence length 2048. It does not use validation/test metrics to build the
profile.

To evaluate the already frozen, hash-audited best profile directly:

```bash
MODEL_PATH="$MODEL_PATH" GPU=4 PYTHON_BIN="$PYTHON_BIN" \
	bash scripts/run_frontier_committee_regret_full.sh
```

## EvalScope capability evaluation

Use a frozen profile and its exact channel cache. Record their SHA256 values in
the preregistration before any evaluation:

```bash
export PROFILE=/path/to/frozen_profile.pt
export CHANNEL_CACHE=/path/to/frozen_channel_cache.pt
export PROFILE_SHA256=<preregistered-64-character-sha256>
export CHANNEL_SHA256=<preregistered-64-character-sha256>
export EVALSCOPE_ROOT=/path/to/evalscope
export DATA_ROOT=/path/to/evalscope_benchmarks
export EVAL_WORK_DIR=/path/to/eval_outputs/qwen3_static_50pct
```

Run artifact and source-identity preflight first. This validates the externally
supplied hashes, train-only profile provenance, exact per-layer metadata when
present, channel topology, physical GPU policy, and writes an auditable
manifest without loading model weights:

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH="$PWD/code:$EVALSCOPE_ROOT" \
"$PYTHON_BIN" code/scripts/run_evalscope_static_profile.py \
	--model-path "$MODEL_PATH" \
	--model-id qwen3-static-50pct \
	--model-family qwen3 \
	--profile "$PROFILE" \
	--channel-cache "$CHANNEL_CACHE" \
	--expected-profile-file-sha256 "$PROFILE_SHA256" \
	--expected-channel-file-sha256 "$CHANNEL_SHA256" \
	--work-dir "$EVAL_WORK_DIR" \
	--datasets gsm8k math_500 arc hellaswag \
	--dataset-args "{\"gsm8k\":{\"local_path\":\"$DATA_ROOT/gsm8k\"},\"math_500\":{\"local_path\":\"$DATA_ROOT/math_500\"},\"arc\":{\"local_path\":\"$DATA_ROOT/arc\"},\"hellaswag\":{\"local_path\":\"$DATA_ROOT/hellaswag\"}}" \
	--seed 42 \
	--no-enable-thinking \
	--no-timestamp \
	--preflight-only
```

Review `evalscope_static_profile_manifest.json`, freeze its source-tree hashes
and task configuration, then rerun the same command without
`--preflight-only`. Do not reuse the preflight work directory for a different
profile, cache, dataset configuration, seed, or source-tree hash.

The local four-task command above is a runnable smoke/core subset. Additional
formal tasks such as MMLU-Pro and IFEval require their own frozen local dataset
copies and explicit `dataset_args`; they must not be downloaded or substituted
silently during a formal run.

## Official REAP matched-profile comparison

The official REAP checkout is used read-only. Freeze its clean commit before
calibration; the current audited checkout is
`1970473c51ca3caeb98c10392f15b3a08a672974`. The bridge calls the official
`LayerwiseMoEObserver` and official `reap` saliency implementation directly,
with `renormalize_router_weights=true`, but does not import the repository's
benchmark entrypoint or install its separate vLLM stack.

First create one train-only token artifact shared by Official REAP,
Route×Tail, and Tail-Risk:

```bash
"$PYTHON_BIN" code/scripts/build_shared_calibration_token_cache.py \
	--model-path "$MODEL_PATH" \
	--output-cache /path/to/c1_wikitext_train_128x2048.pt \
	--dataset wikitext \
	--config wikitext-2-raw-v1 \
	--split train \
	--text-field text \
	--sequence-length 2048 \
	--calibration-sequences 128 \
	--token-offset 0 \
	--protocol-name c1_wikitext_train_128x2048_v1
```

Pass that exact file through `--calibration-token-cache` when running
`calibrate_hessian_channels.py` and `collect_dynamic_regret_teacher.py`. Build
the official 50% Qwen3 REAP profile with:

```bash
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$MODEL_PATH" \
OFFICIAL_REAP_ROOT=/path/to/reap \
OFFICIAL_REAP_COMMIT=1970473c51ca3caeb98c10392f15b3a08a672974 \
CALIBRATION_CACHE=/path/to/c1_wikitext_train_128x2048.pt \
CHANNEL_CACHE=/path/to/matched_channel_cache.pt \
OUTPUT_OBSERVER=/path/to/reap_observer.pt \
OUTPUT_PROFILE=/path/to/reap_50pct.pt \
EXPERTS_TO_PRUNE_PER_LAYER=64 \
GPU=4 \
bash scripts/run_reap_matched.sh
```

During method iteration, REAP is evaluated on the base checkpoint with a
retained-expert router mask and full/zero expert widths. A unit-level
equivalence test verifies this produces the same expert mixture as physically
deleting the same experts and router rows. This avoids generating a checkpoint
per seed/budget. Only frozen final candidates are physically exported with the
official `prune()` path for tensor-byte, disk, reload, memory, and deployment
measurements.

Before paired evaluation, run `validate_reap_profile_pair.py`. The
`method_native` group requires equal total blocks and shared calibration and
evaluation artifacts; `per_layer_controlled` additionally requires exact
per-layer block equality. Both profiles then use the same
`run_evalscope_static_profile.py` command, dataset config, decoding, and answer
extraction.

REAP normally has zero routed-channel pruning because each token still executes
top-k full-width retained experts. That is an expected result, not a runtime
error. Report it separately from REAP's routed-expert parameter pruning.

## Reproduced results

- Frontier Committee Regret: PPL `8.6589252813`, 60.000271% structural pruning.

External baseline implementations are outside this repository's publication
scope. Their historical measurements remain in the experiment ledger only.
