#!/usr/bin/env bash
# Usage: scripts/build_checkpoints.sh [2b|4b]    (default 2b)
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

size="${1:-2b}"
case "$size" in
  2b) model_dir="models/Qwen3.5-2B" ;;
  4b) model_dir="models/Qwen3.5-4B" ;;
  *) printf 'Usage: %s [2b|4b]\n' "$0" >&2; exit 2 ;;
esac
if [[ ! -d "$model_dir" ]]; then
  printf 'Missing local base model: %s\n' "$model_dir" >&2
  exit 1
fi

for cycles in 2 3 5 8; do
  uv run python -m qfs.agent "$model_dir" \
    "build/babyagi-${size}-ledger-pointer-b256-m${cycles}" \
    --cycles "$cycles" \
    --protocol babyagi_ledger_pointer \
    --think-tokens 256 \
    --device mps
done
