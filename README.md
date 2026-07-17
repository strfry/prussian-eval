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
