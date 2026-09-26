#!/usr/bin/env bash
# Prefix each stdin line with an ISO timestamp. Translate tqdm CR updates to newlines.
set -euo pipefail
stdbuf -oL tr '\r' '\n' | while IFS= read -r line; do
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${line}"
done
