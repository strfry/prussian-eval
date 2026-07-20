#!/usr/bin/env bash
# Standard-Modell über das öffentliche vLLM-Image auf RunPod deployen —
# zum Testen der Eval-Pipeline ohne eigenes Docker-Image.
#
# Modell-spezifischer Wrapper wie runpod_deploy_bonsai.sh: setzt nur die
# vllm-serve-Argumente, den Pod-Lebenszyklus übernimmt runpod_pod.sh.
#
# Usage:
#   export RUNPOD_API_KEY=...   # https://www.runpod.io/console/user/settings
#   ./scripts/runpod_deploy_vllm.sh
#
# Env: MODEL (HF-Id), SERVED_NAME, MAX_MODEL_LEN, IMAGE, POD_NAME
set -euo pipefail

# Qwen3-30B-A3B-Instruct-2507: 262k Token nativ, MoE (3B aktiv → schnell),
# gutes Tool-Calling. FP8-Gewichte (~31 GB) laufen auf Ampere (A40) als
# Weight-only via Marlin. Der volle 262k-Kontext passt auf 48 GB nur mit
# FP8-KV-Cache (48 statt 96 KiB/Token); mit BF16-KV auf ~128k gehen.
# KV-Dtype "fp8" = e4m3 — e5m2 lehnt vLLM bei FP8-Checkpoints ab.
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
# Schützt den öffentlichen RunPod-Proxy-Endpoint (vLLM liest VLLM_API_KEY)
VLLM_API_KEY="${VLLM_API_KEY:-$(openssl rand -hex 16)}"

# 
MAX_MODEL_LEN=1280
MODEL=CohereLabs/c4ai-command-r-plus-4bit
SERVED_NAME=$(basename "$MODEL")
VOLUME_GB=100

# Entrypoint des Images ist ["vllm", "serve"] → Modell als Positionsargument.
# Tool-Calling-Parser nötig für den inspect-ai-Agent-Loop (submit()-Tool).
docker_args="$MODEL \
  --hf-token $HF_TOKEN \
  --served-model-name $SERVED_NAME \
  --max-model-len $MAX_MODEL_LEN \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes"

# HF_HOME auf dem Volume: die Gewichte (~31 GB, mehr als die Container-Disk)
# landen dort und überleben Pod-Restarts
pod_id=$(POD_NAME="${POD_NAME:-vllm-eval}" PORT=8000 VOLUME_GB="${VOLUME_GB:-60}" \
  "$(dirname "$0")/runpod_pod.sh" create "$IMAGE" \
    --env "$(jq -n --arg key "$VLLM_API_KEY" '{VLLM_API_KEY: $key, HF_HOME: "/workspace/hf"}')" \
    --docker-args "$docker_args")

base_url="https://${pod_id}-8000.proxy.runpod.net/v1"

cat <<EOF

In env.runpod.sh eintragen, dann 'source env.runpod.sh':
  export OPENAI_BASE_URL="$base_url"
  export OPENAI_API_KEY="$VLLM_API_KEY"
  export RUNPOD_MODEL="$SERVED_NAME"
  export INSPECT_EVAL_MODEL="openai-api/openai/\$RUNPOD_MODEL"

Erster Start lädt Image + Modellgewichte — ein paar Minuten warten.
Health-Check:
  curl -H "Authorization: Bearer $VLLM_API_KEY" $base_url/models
Stoppen (Abrechnung läuft sonst weiter!):
  ./scripts/runpod_pod.sh delete $pod_id
EOF
