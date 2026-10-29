#!/usr/bin/env bash
set -euo pipefail
# 用独立日志目录启动冻结教师，参数依次为源码根、资源根、原生目录、Python、socket、日志父目录。
repo=$1
bundle=$2
native=$3
python=$4
socket=$5
logs=$6
export PYTHONPATH="$repo/mirrorforce${PYTHONPATH:+:$PYTHONPATH}"
export MF_DUEL_NATIVE="$native/duel_native.cpython-311-x86_64-linux-gnu.so"
export MF_EFFECTINFO_LIB="$native/libmfcore.so"
export MF_YGOPRO_RUN="$bundle"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=2
checkpoint=dd71087ce1bf827d9b8360c310ae4be02714b50b9535f4959544d24372d28561
announce=("$bundle"/announce-tables-*.json)
tables=("$bundle"/card-tables-*.npz)
[[ ${#announce[@]} == 1 && ${#tables[@]} == 1 ]]
mkdir -p "$logs"
cd "$repo/mirrorforce"
exec "$python" -u -m tools.mf_runtime_policy_service \
  --checkpoint "$bundle/checkpoint/$checkpoint.ckpt" --weights iterate --selection greedy \
  --cards-db "$bundle/cards.cdb" --code-list "$bundle/code_list.txt" --script-root "$bundle" \
  --announce-tables "${announce[0]}" --semantic-file "$bundle/frozen_semantics.npz" \
  --card-tables "${tables[0]}" --warmup-deck "$bundle/decks/SkyStriker.ydk" --opponent-mode mirror \
  --socket "$socket" --out "$logs/service-$(date +%s)-$$"
