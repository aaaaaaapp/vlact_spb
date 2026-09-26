#!/usr/bin/env bash
# 兼容入口 → 高并行 tosutil 下载
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download_vlact_pi_from_tos_fast.sh" "$@"
