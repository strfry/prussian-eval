"""Dataset loader for the corpus-reconstruction eval (pure, no LLM).

Builds inspect-ai ``Sample``s from ENG↔Old-Prussian sentence pairs:

* the **English** gloss (``sources[].translation`` in the YouTube corpus) is
  the agent input,
* the Old-Prussian subtitle (``text_clean`` / normalized ``text_norm``) is the
  gold target,
* the gold **CoNLL-U parse** (``prussian_silver.conllu``) supplies the part-of-
  speech filter and the *focus tokens* (words of the filtered POS) that the
  recovery probe looks for.

The join between the two corpora is the exact sentence text
(silver ``# text`` == corpus ``text_clean``).

``make_dataset`` is a single configurable filter: ``pos`` (default ``"ADV"`` —
the adverb example) is just one knob alongside ``min_words`` / ``max_words`` /
``lang``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

# Sibling checkouts: mcp/, corpus/, fst/ live next to each other under the
# prussian binder repo root (see pyproject [tool.uv.sources]).
_PROJECTS = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = _PROJECTS / "corpus" / "parsed" / "youtube_corpus_sentences.json"
DEFAULT_SILVER = _PROJECTS / "fst" / "data" / "prussian_silver.conllu"


def _focus_tokens_by_text(silver_path: Path, pos: str | None) -> dict[str, list[dict]]:
    """Map gold sentence text -> list of {form, lemma} tokens with UPOS==pos.

    Streams the CoNLL-U file.  When ``pos`` is None every sentence maps to an
    empty focus list (the POS filter is disabled, but the text set is still the
    set of parsed gold sentences).
    """
    out: dict[str, list[dict]] = {}
    text: str | None = None
    focus: list[dict] = []
    with silver_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
                focus = []
            elif line.strip() == "":
                if text is not None:
                    out[text] = focus
                text, focus = None, []
            elif text is not None and not line.startswith("#"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 4 and (pos is None or cols[3] == pos):
                    if pos is not None:
                        focus.append({"form": cols[1], "lemma": cols[2]})
        if text is not None:
            out[text] = focus
    return out


def _translation_by_text(corpus_path: Path, lang: str) -> dict[str, str]:
    """Map gold sentence text -> first available <lang> translation gloss."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in data:
        text = entry.get("text_clean")
        norm = entry.get("text_norm")
        if not text:
            continue
        for src in entry.get("sources", []):
            if src.get("sub_lang") == lang and src.get("translation"):
                out.setdefault(text, {"translation": src["translation"], "norm": norm})
                break
    return out


def make_dataset(
    pos: str | None = "ADV",
    min_words: int = 1,
    max_words: int | None = None,
    lang: str = "en",
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    corpus_path: str | Path = DEFAULT_CORPUS,
    silver_path: str | Path = DEFAULT_SILVER,
) -> MemoryDataset:
    """Build the reconstruction dataset.

    Args:
        pos: keep only sentences whose gold parse contains a token with this
            UPOS (e.g. ``"ADV"``, ``"VERB"``).  ``None`` disables the POS
            filter (and the focus-recovery probe).
        min_words / max_words: word-count bounds on the gold sentence
            (``max_words=None`` = no upper bound).
        lang: source-translation language used as the agent input (only
            ``"en"`` has real glosses in the current corpus).
        limit: keep at most this many samples (after optional shuffle).
        shuffle / seed: deterministic shuffle before applying ``limit``.
        corpus_path / silver_path: override the sibling-repo defaults.

    Returns:
        A ``MemoryDataset`` of ``Sample(input=<gloss>, target=<text_norm>,
        metadata={gold, pos, focus, n_words})``.
    """
    focus_by_text = _focus_tokens_by_text(Path(silver_path), pos)
    trans_by_text = _translation_by_text(Path(corpus_path), lang)

    samples: list[Sample] = []
    for text, focus in focus_by_text.items():
        if pos is not None and not focus:
            continue
        tr = trans_by_text.get(text)
        if not tr:
            continue
        n_words = len(text.split())
        if n_words < min_words or (max_words is not None and n_words > max_words):
            continue
        samples.append(
            Sample(
                input=tr["translation"],
                target=tr["norm"] or text,
                metadata={
                    "gold": text,
                    "pos": pos,
                    "focus": focus,
                    "n_words": n_words,
                },
            )
        )

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return MemoryDataset(samples)
