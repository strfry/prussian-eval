#!/usr/bin/env bash
# Ternary-Bonsai-27B auf RunPod deployen.
#
# Modell-spezifischer Wrapper: setzt nur die llama-server-Konfiguration
# (LLAMA_ARG_*-Env-Vars) und überlässt den Pod-Lebenszyklus runpod_pod.sh.
# Das Image vorher einmalig bauen und pushen: siehe docker/llamacpp-prism/Dockerfile
#
# Usage:
#   export RUNPOD_API_KEY=...   # https://www.runpod.io/console/user/settings
#   ./scripts/runpod_deploy.sh
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/strfry/llamacpp-prism:prism-b9594-server-cuda}"
# Schützt den öffentlichen RunPod-Proxy-Endpoint
LLAMACPP_API_KEY="${LLAMACPP_API_KEY:-$(openssl rand -hex 16)}"

# llama-server liest LLAMA_ARG_* — kein Startup-Kommando nötig.
# LLAMA_CACHE auf dem /workspace-Volume: das GGUF (7.2 GB) überlebt Restarts.
env_json=$(jq -n --arg key "$LLAMACPP_API_KEY" '{
  LLAMA_ARG_HF_REPO:      "prism-ml/Ternary-Bonsai-27B-gguf:Q2_0",
  LLAMA_ARG_ALIAS:        "Ternary-Bonsai-27B",
  LLAMA_ARG_CTX_SIZE:     "8192",
  LLAMA_ARG_N_GPU_LAYERS: "99",
  LLAMA_ARG_JINJA:        "1",
  LLAMA_ARG_PORT:         "8080",
  LLAMA_ARG_API_KEY:      $key,
  LLAMA_CACHE:            "/workspace/models"
}')

pod_id=$(POD_NAME="${POD_NAME:-bonsai-llamacpp}" MIN_CUDA="${MIN_CUDA:-12.8}" \
  "$(dirname "$0")/runpod_pod.sh" create "$IMAGE" --env "$env_json")

base_url="https://${pod_id}-8080.proxy.runpod.net/v1"

cat <<EOF

In env.runpod.sh eintragen, dann 'source env.runpod.sh':
  export LLAMACPP_BASE_URL="$base_url"
  export LLAMACPP_API_KEY="$LLAMACPP_API_KEY"
  export RUNPOD_MODEL="llamacpp/Ternary-Bonsai-27B"

Erster Start lädt das Modell (7.2 GB) von HF — ein paar Minuten warten.
Health-Check:
  curl -H "Authorization: Bearer $LLAMACPP_API_KEY" $base_url/models
Stoppen (Abrechnung läuft sonst weiter!):
  ./scripts/runpod_pod.sh stop $pod_id
EOF
