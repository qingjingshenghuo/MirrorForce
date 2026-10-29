#!/usr/bin/env bash
set -euo pipefail
# 使用真实发布模型进行成对对局；只推理，不训练。
root=/opt/mirrorforce
export PYTHONPATH="$root/repo/mirrorforce"
export MF_DUEL_NATIVE="$root/native-build-v2/duel_native.cpython-311-x86_64-linux-gnu.so"
export MF_EFFECTINFO_LIB="$root/native-build-v2/libmfcore.so"
export MF_YGOPRO_RUN="$root/assets"
output=${1:-$root/runs/selfplay-$(date +%Y%m%d-%H%M%S)}
cd "$root/repo/mirrorforce"
exec "$root/venv/bin/python" -u -m tools.mf_runtime_local_match \
    --a /tmp/mirrorforce-policy.sock --b /tmp/mirrorforce-policy.sock \
    --deck "$root/assets/decks/SkyStriker.ydk" --cards-db "$root/assets/cards.cdb" \
    --script-root "$root/assets" --native "$MF_DUEL_NATIVE" \
    --games 2 --seed 20261028 --game-seconds 300 --timeout 180 --out "$output"
