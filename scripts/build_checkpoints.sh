#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model_dir="models/Qwen3.5-2B"
if [[ ! -d "$model_dir" ]]; then
  printf 'Missing local base model: %s\n' "$model_dir" >&2
  exit 1
fi

for cycles in 2 3 5 8; do
  uv run python -m qfs.agent "$model_dir" \
    "build/babyagi-2b-ledger-pointer-b256-m${cycles}" \
    --cycles "$cycles" \
    --protocol babyagi_ledger_pointer \
    --think-tokens 256 \
    --device mps
done
