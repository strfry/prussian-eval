"""Dataset loader for the corpus-reconstruction eval (pure, no LLM).

Builds inspect-ai ``Sample``s from English↔Old-Prussian sentence pairs:

* the **English** gloss (the most-frequent translation across all subtitle
  tracks in the YouTube corpus) is the agent input,
* the Old-Prussian subtitle (``text_clean`` / normalized ``text_norm``) is the
  gold target,
* the gold **CoNLL-U parse** (``prussian_silver.conllu``) supplies the part-of-
  speech filter and the *focus tokens* (words of the filtered POS) that the
  recovery probe looks for.

The join between the two corpora is the exact sentence text
(silver ``# text`` == corpus ``text_clean``).

``make_dataset`` is a single configurable filter: ``pos`` (default ``"ADV"`` —
the adverb example) is just one knob alongside ``min_words`` / ``max_words``.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

# Sibling checkouts: mcp/, corpus/, fst/ live next to each other under the
# prussian binder repo root (see pyproject [tool.uv.sources]).
_PROJECTS = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = _PROJECTS / "corpus" / "parsed" / "youtube_corpus_sentences.json"
DEFAULT_SILVER = _PROJECTS / "fst" / "data" / "prussian_silver.conllu"

# ---------------------------------------------------------------------------
# CoNLL-U parser
# ---------------------------------------------------------------------------

def parse_silver(path: Path) -> dict[str, dict]:
    """Parse CoNLL-U into ``{text: {sent_id, tokens, full_lookup, fully_disambiguated}}``.

    Tokens without PUNCT are kept.  ``full_lookup`` is True when every content
    token has a real lemma and UPOS is not ``X`` or ``_``.
    ``fully_disambiguated`` is True when no token carries an ``Ambig=`` feature.
    """
    PUNCT = {"PUNCT"}
    sents: dict[str, dict] = {}
    current: dict | None = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# sent_id ="):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
                current = {
                    "sent_id": sent_id,
                    "text": text,
                    "tokens": [],
                    "_n_content": 0,
                    "_bad_lookup": False,
                    "_has_ambig": False,
                }
            elif line.strip() == "" and current is not None:
                fl = current["_n_content"] > 0 and not current["_bad_lookup"]
                fd = current["_n_content"] > 0 and not current["_has_ambig"]
                sents[current["text"]] = {
                    "sent_id": current["sent_id"],
                    "tokens": current["tokens"],
                    "full_lookup": fl,
                    "fully_disambiguated": fd,
                }
                current = None
            elif current is not None and not line.startswith("#"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 10:
                    continue
                form, lemma, upos, xpos, feats, misc = (
                    cols[1], cols[2], cols[3], cols[4], cols[5], cols[9]
                )
                if upos in PUNCT:
                    continue
                current["_n_content"] += 1
                if lemma == "_" or upos in ("X", "_"):
                    current["_bad_lookup"] = True
                if "Ambig=" in misc:
                    current["_has_ambig"] = True
                current["tokens"].append({
                    "form": form,
                    "lemma": lemma,
                    "upos": upos,
                    "tags": xpos,
                    "feats": feats,
                })
    # flush last
    if current is not None:
        fl = current["_n_content"] > 0 and not current["_bad_lookup"]
        fd = current["_n_content"] > 0 and not current["_has_ambig"]
        sents[current["text"]] = {
            "sent_id": current["sent_id"],
            "tokens": current["tokens"],
            "full_lookup": fl,
            "fully_disambiguated": fd,
        }
    return sents

# ---------------------------------------------------------------------------
# EN translation join + heuristic checks
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[.*?\]")
_MUSIC_RE = re.compile(r"[♪♫]")
_ELLIPSIS = re.compile(r"^(\.\.\.|…)|(\.\.\.|…)$")


def check_translation(trans: str) -> list[str]:
    """Return list of heuristic flags (empty = ok)."""
    flags: list[str] = []
    if _BRACKET_RE.search(trans):
        flags.append("bracket_annotation")
    if _MUSIC_RE.search(trans):
        flags.append("music_symbol")
    if _ELLIPSIS.search(trans):
        flags.append("truncated")
    stripped = trans.strip()
    if not stripped:
        flags.append("empty")
    elif all(c in ".,!?;:—–-… " for c in stripped):
        flags.append("punctuation_only")
    if stripped and stripped[0].islower():
        flags.append("starts_lowercase")
    return flags


def load_translations(path: Path) -> dict[str, dict]:
    """Map ``text_clean`` -> ``{translation_en, text_norm, frequency, translation_flags, translation_ok}``.

    Translations are always English, regardless of the subtitle-track slot
    (``sub_lang``): YouTube has no Prussian language option, so uploaders put
    the Prussian track in an arbitrary slot (en/lt/lv) with the English gloss
    as the second line of the same subtitle block.  We therefore use the
    frequency-sorted ``translations`` aggregate, not the per-source language.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for entry in data:
        text = entry.get("text_clean")
        if not text:
            continue
        norm = entry.get("text_norm", "")
        freq = entry.get("frequency", 0)
        translations = entry.get("translations") or []
        trans = translations[0]["text"] if translations else None
        if not trans:
            continue
        flags = check_translation(trans)
        out[text] = {
            "translation_en": trans,
            "text_norm": norm,
            "frequency": freq,
            "translation_flags": flags,
            "translation_ok": len(flags) == 0,
        }
    return out

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def make_dataset(
    pos: str | None = "ADV",
    min_words: int = 1,
    max_words: int | None = None,
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
        limit: keep at most this many samples (after optional shuffle).
        shuffle / seed: deterministic shuffle before applying ``limit``.
        corpus_path / silver_path: override the sibling-repo defaults.

    Returns:
        A ``MemoryDataset`` of ``Sample(input=<gloss>, target=<text_norm>,
        metadata={gold, pos, focus, n_words})``.
    """
    sents = parse_silver(Path(silver_path))
    trans_by_text = load_translations(Path(corpus_path))

    samples: list[Sample] = []
    for text, sinfo in sents.items():
        tokens = sinfo["tokens"]
        focus = [{"form": t["form"], "lemma": t["lemma"]} for t in tokens if t["upos"] == pos]
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
                input=tr["translation_en"],
                target=tr["text_norm"] or text,
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
