#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

failed=0

if grep -RIn --include='*.py' -E '^[[:space:]]*(from|import)[[:space:]]+moe_prune_v[23]' code; then
  echo "WARNING: historical compatibility imports remain; model-level clean-clone execution is not yet self-contained." >&2
fi

if grep -RIn -E '/data01/home/|/home/[^/]+/' \
  README.md docs/REPRODUCTION_COMMANDS.md scripts code \
  --exclude='audit_publication.sh'; then
  echo "ERROR: machine-specific home paths remain in runtime documentation or code." >&2
  failed=1
fi

while IFS= read -r -d '' path; do
  echo "ERROR: publishable file exceeds 100 MiB: $path" >&2
  failed=1
done < <(git ls-files --cached --others --exclude-standard -z | while IFS= read -r -d '' path; do
  [[ -f "$path" ]] || continue
  (( $(stat -c '%s' "$path") > 104857600 )) && printf '%s\0' "$path"
done)

if git ls-files --cached --others --exclude-standard | \
  grep -E '(^|/)\.git(/|$)|\.(safetensors|ckpt|pth|npy|npz)$|pytorch_model.*\.bin$'; then
  echo "ERROR: nested Git metadata, model weights, or disallowed tensor artifacts are publishable." >&2
  failed=1
fi

allowed_prior_pattern='^experiments/calibration/(qwen15_moe_priors_20260728|qwen3_base_priors_20260728|static_expert_priors_20260728|qwen35_prospective_priors_20260728)/(amp_scores|aimer_scores)\.pt$'
publishable_priors="$(git ls-files --cached --others --exclude-standard '*.pt')"
unexpected_pt="$(printf '%s\n' "$publishable_priors" | grep -Ev "$allowed_prior_pattern" || true)"
if [[ -n "$unexpected_pt" ]]; then
  printf '%s\n' "$unexpected_pt"
  echo "ERROR: only the allowlisted frozen AMP/AIMER tables may be published." >&2
  failed=1
fi
if [[ "$(printf '%s\n' "$publishable_priors" | grep -Ec "$allowed_prior_pattern")" -ne 8 ]]; then
  printf '%s\n' "$publishable_priors"
  echo "ERROR: the publication must contain exactly eight frozen AMP/AIMER tables." >&2
  failed=1
fi

(
  cd experiments/calibration
  sha256sum -c FROZEN_PRIORS.sha256
)

bash -n scripts/*.sh

if (( failed != 0 )); then
  exit 1
fi

echo "publication_audit=passed"