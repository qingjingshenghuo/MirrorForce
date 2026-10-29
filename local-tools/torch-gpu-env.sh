#!/usr/bin/env bash
set -euo pipefail
# 独立运行CUDA PyTorch：不覆盖CPU测试包，不修改冠军JAX环境。
root=${MF_ROOT:-/opt/mirrorforce}
repo=${MF_REPO:-/mnt/f/YGOagent}
vendor="$root/venv/lib/python3.11/site-packages/nvidia"
vendor_libs=$(find "$root/distill-gpu-deps/nvidia" "$vendor" -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="${vendor_libs}/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$root/distill-gpu-deps:$root/distill-testdeps:$repo/mirrorforce${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec "$root/venv/bin/python" "$@"
