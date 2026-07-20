# prussian-eval

inspect-ai reconstruction eval suite for [prussian-mcp](https://github.com/strfry/prussian-mcp).
Feeds an English gloss to the `prussian-agent` toolset and scores the Old
Prussian output against the gold subtitle corpus.

## Layout

| Path | Role |
|---|---|
| `evals/reconstruction.py` | The inspect-ai `Task` — prompt-surface knob (`instruct`), scorer, `submit()`-based agent loop. |
| `evals/corpus_dataset.py` | Dataset loader — joins the YouTube corpus with the ConLL-U silver parse for the focus-token recovery metric. |
| `scripts/ragas_analysis.py` | Post-processes inspect-ai eval logs with RAGAS-style metrics (context recall, faithfulness). |
| `run_multi_model_eval.sh` | Runs the reconstruction task across multiple model/embedding configs. |
| `logs/` | Eval run output (gitignored). |

## Setup

Expects to live as a sibling of `mcp/`, `fst/`, `corpus/` under the
[prussian binder repo](https://github.com/strfry/prussian) (`../mcp`,
`../fst`, `../corpus` — see `pyproject.toml` `[tool.uv.sources]` and
`evals/corpus_dataset.py`). Prompts (`prompts/`) live in `../mcp/` too.

```bash
cd ../mcp && make -C ../fst cg3-sets && make -C ../fst cg3-check && make -C ../fst conllu
cd ../eval && uv sync
```

Env files (`OPENAI_API_KEY` / embedding config) are the same ones used by
`prussian-mcp` — copy or symlink them from `../mcp/`.

## Running

```bash
source ../mcp/env.hf-voyage.sh
uv run inspect eval evals/reconstruction.py --model openai/$OPENAI_MODEL -T limit=3
uv run inspect eval evals/reconstruction.py -T instruct=basevocab -T pos=ADV
uv run inspect view

./run_multi_model_eval.sh --limit 10 --instruct basevocab
uv run python scripts/ragas_analysis.py logs/
```

## RunPod (self-hosted inference)

Für Evaluations auf selbst-gehosteten Modellen über einen RunPod-Server.
Drei getrennte Concerns:

- **GPU-Auswahl**: `scripts/runpod_gpu.sh` — nimmt die erste brauchbare GPU
  aus der Präferenzliste (`GPU_PREFS`), überspringt Low-Stock und setzt den
  passenden `CLOUD_TYPE` (z.B. RTX PRO 4000: nur Secure Cloud).
- **Pod-Lebenszyklus**: `scripts/runpod_pod.sh`
  (`create|list|status|start|stop|delete`).
- **Modell-Deployment**: dünne Wrapper darüber (unten), drucken am Ende
  Pod-URL und API-Key zum Eintragen in `env.runpod.sh`.

### Variante 1: Standard-Modell via vLLM (Pipeline-Test)

Öffentliches `vllm/vllm-openai`-Image, kein eigenes Docker-Image nötig:

```bash
export RUNPOD_API_KEY=...         # runpod.io/console/user/settings
./scripts/runpod_deploy_vllm.sh   # Default: Qwen3-30B-A3B-Instruct-2507-FP8, 262k Kontext
```

### Variante 2: Ternary-Bonsai-27B (eigenes Image nötig)

Der Q2_0_g128-Quant braucht die Hybrid-Attention-Kernels des
[PrismML-Forks](https://github.com/PrismML-Eng/llama.cpp) — das offizielle
llama.cpp-Server-Image lädt das GGUF nicht. `docker/llamacpp-prism/Dockerfile`
baut den Fork (multi-stage, statisches `llama-server`-Binary, Konfiguration
über `LLAMA_ARG_*`-Env-Vars) und muss in eine für RunPod erreichbare Registry:

```bash
docker build -t ghcr.io/strfry/llamacpp-prism:prism-b9594-server-cuda docker/llamacpp-prism
docker push ghcr.io/strfry/llamacpp-prism:prism-b9594-server-cuda

./scripts/runpod_deploy_bonsai.sh
```

Das GGUF landet im `/workspace`-Volume (`LLAMA_CACHE`) und überlebt
Pod-Restarts.

### inspect eval gegen RunPod

```bash
source env.runpod.sh   # LLAMACPP_BASE_URL + OPENAI_BASE_URL setzen

uv run inspect eval evals/reconstruction.py \
  --model openai-api/$RUNPOD_MODEL \
  -T limit=3

# oder mit Inspect-Env-Variable:
uv run inspect eval evals/reconstruction.py -T instruct=basevocab
```

### Kosten

- RTX 4000 Ada: ~$0.34/hr (Community Cloud); RTX PRO 4000: nur Secure Cloud, teurer — `runpodctl gpu list` / Console prüfen
- Pod bleibt laufen bis gestoppt — für längere Eval-Sessions
- Storage: $0.10/GB/mo (Volume)
