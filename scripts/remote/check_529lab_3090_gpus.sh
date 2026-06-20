#!/usr/bin/env bash
set -euo pipefail

host="${1:-529_Lab-3090}"

ssh "$host" 'nvidia-smi; echo; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
