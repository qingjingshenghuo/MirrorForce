#!/usr/bin/env bash
set -euo pipefail
# 只加载发布策略进行推理，不执行训练更新。
root=/opt/mirrorforce
bundle="$root/assets"
checkpoint=dd71087ce1bf827d9b8360c310ae4be02714b50b9535f4959544d24372d28561
socket=${1:-/tmp/mirrorforce-policy.sock}
output=${2:-$root/runs/service-$(date +%Y%m%d-%H%M%S)}
echo "$checkpoint  $bundle/checkpoint/$checkpoint.ckpt" | sha256sum -c -
export MF_DUEL_NATIVE="$root/native-build-v2/duel_native.cpython-311-x86_64-linux-gnu.so"
export MF_EFFECTINFO_LIB="$root/native-build-v2/libmfcore.so"
export MF_YGOPRO_RUN="$bundle"
export MF_SCRIPTED_RUN="$bundle"
export MF_TEST_ASSETS="$bundle"
export PYTHONPATH="$root/repo/mirrorforce"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="$root/cache/jax"
export OMP_NUM_THREADS=2
cd "$root/repo/mirrorforce"
exec "$root/venv/bin/python" -u -m tools.mf_runtime_policy_service \
    --checkpoint "$bundle/checkpoint/$checkpoint.ckpt" --weights iterate --selection greedy \
    --cards-db "$bundle/cards.cdb" --code-list "$bundle/code_list.txt" --script-root "$bundle" \
    --announce-tables "$bundle/announce-tables-3c7d93af1db79dc64048fca21cc6ba09a0b94f75066b0f9aad6f12ca7d939c20.json" \
    --semantic-file "$bundle/frozen_semantics.npz" \
    --card-tables "$bundle/card-tables-f261202ace3caccf77245737a6b8ef9d67948958753f3add69f86fa2c480de56.npz" \
    --warmup-deck "$bundle/decks/SkyStriker.ydk" --opponent-mode mirror \
    --socket "$socket" --out "$output"
