#!/usr/bin/env bash
# GPU-Auswahl als eigener Concern: nimmt die erste brauchbare GPU aus der
# Präferenzliste und gibt eval-bare Exports aus — inklusive CLOUD_TYPE, denn
# nicht jede GPU gibt es in jeder Cloud (RTX PRO 4000: nur Secure).
#
# "available" allein ist unzuverlässig (RTX 4000 Ada meldet available bei
# stockStatus Low, deploy schlägt trotzdem fehl) — daher werden GPUs mit
# Low-Stock übersprungen, solange es eine bessere Option gibt.
#
# Usage:
#   eval "$(./scripts/runpod_gpu.sh)"           # setzt GPU_ID + CLOUD_TYPE
#   GPU_PREFS="NVIDIA RTX A4000" ./scripts/runpod_gpu.sh
set -euo pipefail

# Reihenfolge ≈ Preis aufsteigend; A40 (Ampere, 48 GB) ist als vorletzte
# Generation günstiger als die RTX PRO 4000 (Blackwell)
#GPU_PREFS="${GPU_PREFS:-NVIDIA RTX 4000 Ada Generation,NVIDIA A40,NVIDIA RTX PRO 4000 Blackwell,NVIDIA RTX A4000}"
GPU_PREFS="NVIDIA RTX PRO 6000 Blackwell Server Edition"

gpus=$(runpodctl gpu list 2>/dev/null)

pick() {  # $1: jq-Zusatzfilter
  IFS=',' read -ra prefs <<<"$GPU_PREFS"
  for gpu in "${prefs[@]}"; do
    cloud=$(echo "$gpus" | jq -r --arg id "$gpu" \
      ".[] | select(.gpuId == \$id and .available and ($1)) |
       if .communityCloud then \"COMMUNITY\" else \"SECURE\" end")
    if [ -n "$cloud" ]; then
      echo "export GPU_ID=\"$gpu\""
      echo "export CLOUD_TYPE=\"$cloud\""
      mem=$(echo "$gpus" | jq -r --arg id "$gpu" '.[] | select(.gpuId == $id).memoryInGb')
      echo "GPU: $gpu (${mem} GB, $cloud)" >&2
      return 0
    fi
  done
  return 1
}

pick '.stockStatus != "Low"' && exit 0
echo "Nur Low-Stock-GPUs verfügbar — Deploy kann fehlschlagen." >&2
pick 'true' && exit 0
echo "Keine GPU aus der Präferenzliste verfügbar: $GPU_PREFS" >&2
exit 1
