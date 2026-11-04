#!/usr/bin/env bash
set -euo pipefail

limit_bytes=50000000
failed=0

while IFS= read -r -d '' path; do
  [[ -f "$path" ]] || continue
  bytes=$(wc -c < "$path" | tr -d ' ')
  if (( bytes > limit_bytes )); then
    printf 'Tracked file exceeds 50 MB: %s (%s bytes)\n' "$path" "$bytes" >&2
    failed=1
  fi
done < <(git ls-files -z)

exit "$failed"
