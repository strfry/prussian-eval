"""Dataset loader for the corpus-reconstruction eval (pure, no LLM).

Builds inspect-ai ``Sample``s from the curated ``data/quasi_gold.jsonl``
(606 sentences): English gloss as input, Old-Prussian sentence as gold target,
with *focus tokens* (words of the filtered POS) for the recovery probe.

The ``pos`` parameter (default ``"ADV"``) controls which POS tokens are
flagged as focus — sentences without focus tokens for the chosen POS are
skipped when ``pos`` is not ``None``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

_EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUASI_GOLD = _EVAL_DIR / "data" / "quasi_gold.jsonl"

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
    path: str | Path = DEFAULT_QUASI_GOLD,
) -> MemoryDataset:
    """Build the reconstruction dataset from the curated quasi-gold JSONL.

    Args:
        pos: keep only sentences whose token list contains a word with this
            UPOS (e.g. ``"ADV"``, ``"VERB"``).  ``None`` disables the POS
            filter (and the focus-recovery probe).
        path: path to ``quasi_gold.jsonl``.

    Returns:
        A ``MemoryDataset`` of ``Sample(input=<gloss>, target=<text_norm>,
        metadata={gold, pos, focus, n_words, sent_id})``.
    """
    samples: list[Sample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get("tokens", [])
            focus = [{"form": t["form"], "lemma": t["lemma"]} for t in tokens if t["upos"] == pos]
            if pos is not None and not focus:
                continue
            samples.append(
                Sample(
                    input=row["translation_en"],
                    target=row.get("text_norm") or row["text_clean"],
                    metadata={
                        "gold": row["text_clean"],
                        "pos": pos,
                        "focus": focus,
                        "n_words": row.get("n_words", len(row["text_clean"].split())),
                        "sent_id": row.get("sent_id", ""),
                    },
                )
            )
    return MemoryDataset(samples)
