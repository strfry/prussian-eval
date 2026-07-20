#!/usr/bin/env bash
# Generischer RunPod-Pod-Lebenszyklus (modell-agnostisch, dünner Wrapper um
# runpodctl v2). Modell-/Server-Konfiguration kommt von außen über --env etc.
#
# Usage:
#   ./scripts/runpod_pod.sh create <image> [weitere runpodctl-Flags…]
#   ./scripts/runpod_pod.sh url <pod-id>
#   ./scripts/runpod_pod.sh list | status|start|stop|delete <pod-id>
#
# Env (create): POD_NAME, GPU_ID, CLOUD_TYPE, PORT, VOLUME_GB, MIN_CUDA
#
# create gibt nur die Pod-ID auf stdout aus (für Skript-Verwendung),
# alles andere auf stderr.
set -euo pipefail

#: "${RUNPOD_API_KEY:?RUNPOD_API_KEY setzen (runpod.io/console/user/settings)}"

PORT="${PORT:-8080}"
cmd="${1:?Kommando fehlt (create|list|url|status|start|stop|delete)}"
shift || true

case "$cmd" in
  create)
    image="${1:?Image fehlt}"
    shift
    # GPU-Auswahl ist ein eigener Concern — setzt auch den passenden CLOUD_TYPE
    [ -n "${GPU_ID:-}" ] || eval "$("$(dirname "$0")/runpod_gpu.sh")"
    CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
    args=(
      --name "${POD_NAME:-eval-pod}"
      --image "$image"
      --gpu-id "$GPU_ID"
      --cloud-type "$CLOUD_TYPE"
      --ports "${PORT}/http"
      --volume-in-gb "${VOLUME_GB:-25}"
    )
    [ -n "${MIN_CUDA:-}" ] && args+=(--min-cuda-version "$MIN_CUDA")
    echo "Erzeuge Pod '${POD_NAME:-eval-pod}' ($GPU_ID, $CLOUD_TYPE)…" >&2
    created=$(runpodctl pod create "${args[@]}" "$@" -o json)
    pod_id=$(echo "$created" | jq -r '.id // .pod.id // empty')
    if [ -z "$pod_id" ]; then
      echo "Konnte Pod-ID nicht aus der Antwort lesen:" >&2
      echo "$created" >&2
      exit 1
    fi
    echo "Pod: $pod_id" >&2
    echo "Endpoint: https://${pod_id}-${PORT}.proxy.runpod.net" >&2
    echo "$pod_id"
    ;;
  url)
    pod_id="${1:?Pod-ID fehlt}"
    echo "https://${pod_id}-${PORT}.proxy.runpod.net"
    ;;
  list)
    runpodctl pod list "$@"
    ;;
  status)
    runpodctl pod get "${1:?Pod-ID fehlt}"
    ;;
  start|stop|delete)
    runpodctl pod "$cmd" "${1:?Pod-ID fehlt}"
    ;;
  *)
    echo "Unbekanntes Kommando: $cmd" >&2
    exit 1
    ;;
esac
