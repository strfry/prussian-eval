"""Build a quasi-gold dataset from EN↔Prussian sentence pairs.

Joins three data sources:
  1. prussian_silver.conllu  – POS tags, lemmas, features, disambiguation flags
  2. youtube_corpus_sentences.json – English translations, frequency
  3. eval/logs/*.eval        – model-evaluation scores (inspect-ai ZIPs)

Outputs (into eval/data/):
  • sentences_scored.jsonl – all sentences with flags & scores
  • quasi_gold.jsonl       – gold subset: full_lookup ∧ fully_disambiguated ∧ translation_ok
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

from evals.corpus_dataset import load_translations, parse_silver

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
_SILVER = _ROOT / "fst" / "data" / "prussian_silver.conllu"
_CORPUS = _ROOT / "corpus" / "parsed" / "youtube_corpus_sentences.json"
_EVAL_DIR = _ROOT / "eval" / "logs"
_OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Eval signal aggregation
# ---------------------------------------------------------------------------

def _aggregate_eval_signals(eval_dir: Path) -> dict[str, dict]:
    """Map gold text -> aggregated eval metrics across all runs.

    The eval ZIPs use ZSTD compression (type 93) which requires Python >= 3.13.
    We shell out to the system Python to read them when the current interpreter
    does not support the compression method.
    """
    if not eval_dir.is_dir():
        return {}

    # Try native zipfile first
    try:
        return _aggregate_eval_native(eval_dir)
    except NotImplementedError:
        pass

    # Fallback: shell out to system python3 (>= 3.13)
    print("    (zipfile needs ZSTD support – falling back to system python3)")
    script = (
        "import json, os, sys, zipfile\n"
        "from collections import defaultdict\n"
        "eval_dir = sys.argv[1]\n"
        "agg = {}\n"
        "for f in sorted(os.listdir(eval_dir)):\n"
        "    if not f.endswith('.eval'): continue\n"
        "    try:\n"
        "        zf = zipfile.ZipFile(os.path.join(eval_dir, f))\n"
        "    except: continue\n"
        "    if 'summaries.json' not in zf.namelist(): continue\n"
        "    for s in json.loads(zf.read('summaries.json')):\n"
        "        gm = s.get('scores',{}).get('gold_match')\n"
        "        if not gm: continue\n"
        "        gold = s.get('metadata',{}).get('gold') or gm.get('metadata',{}).get('gold')\n"
        "        if not gold: continue\n"
        "        m = gm.get('metadata',{})\n"
        "        a = agg.setdefault(gold, {'best_token_f1':0,'best_focus_recovery':0,'n_runs':0,'ever_conllu_complete':False})\n"
        "        a['n_runs'] += 1\n"
        "        f1 = m.get('token_f1',0)\n"
        "        if f1 > a['best_token_f1']: a['best_token_f1'] = f1\n"
        "        fr = m.get('focus_recovery',0)\n"
        "        if fr > a['best_focus_recovery']: a['best_focus_recovery'] = fr\n"
        "        if m.get('conllu_complete'): a['ever_conllu_complete'] = True\n"
        "json.dump(agg, sys.stdout)\n"
    )
    import subprocess
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script, str(eval_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"    WARNING: system python3 failed: {result.stderr[:500]}")
        return {}
    return json.loads(result.stdout)


def _aggregate_eval_native(eval_dir: Path) -> dict[str, dict]:
    """Aggregate eval signals using native zipfile (needs Python >= 3.13 for ZSTD)."""
    agg: dict[str, dict] = defaultdict(lambda: {
        "best_token_f1": 0.0,
        "best_focus_recovery": 0.0,
        "n_runs": 0,
        "ever_conllu_complete": False,
    })

    for fname in sorted(eval_dir.glob("*.eval")):
        try:
            zf = zipfile.ZipFile(fname)
            if "summaries.json" not in zf.namelist():
                continue
            summaries = json.loads(zf.read("summaries.json"))
            for sample in summaries:
                scores = sample.get("scores", {})
                gm = scores.get("gold_match")
                if not gm:
                    continue
                # Prefer sample.metadata.gold (original casing, matches CoNLL-U # text)
                gold = sample.get("metadata", {}).get("gold") or gm.get("metadata", {}).get("gold")
                if not gold:
                    continue
                meta = gm.get("metadata", {})
                a = agg[gold]
                a["n_runs"] += 1
                f1 = meta.get("token_f1", 0.0)
                if f1 > a["best_token_f1"]:
                    a["best_token_f1"] = f1
                fr = meta.get("focus_recovery", 0.0)
                if fr > a["best_focus_recovery"]:
                    a["best_focus_recovery"] = fr
                if meta.get("conllu_complete"):
                    a["ever_conllu_complete"] = True
        except NotImplementedError:
            raise  # let caller fall back to subprocess
        except Exception:
            continue

    return dict(agg)

# ---------------------------------------------------------------------------
# 4. Formenreichtum (richness) score
# ---------------------------------------------------------------------------

# Weights: normalised to sum ≈ 1.0
_W_UPSOS  = 0.10   # distinct UPOS count
_W_FEATS  = 0.20   # distinct (UPOS, feats) bundles
_W_CASES  = 0.15   # distinct case values
_W_TEMPS  = 0.10   # distinct tense values
_W_LEMMA  = 0.15   # lemma diversity  (distinct lemmas / n_tokens)
_W_LENGTH = 0.10   # normalised token count (capped at 30)
_W_LEMMA_RAW = 0.20 # raw distinct lemma count (capped at 15)


def _extract_feats(feats_str: str) -> dict[str, str]:
    """Parse FEATS column into {Feature: Value}."""
    if not feats_str or feats_str == "_":
        return {}
    out = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _richness(tokens: list[dict]) -> float:
    """Compute normalised richness ∈ [0, 1]."""
    n = len(tokens)
    if n == 0:
        return 0.0

    upos_set = set()
    bundles = set()
    cases = set()
    temps = set()
    lemmas = set()

    for t in tokens:
        upos = t["upos"]
        upos_set.add(upos)
        feats = _extract_feats(t.get("feats", ""))
        bundles.add((upos, tuple(sorted(feats.items()))))
        if "Case" in feats:
            cases.add(feats["Case"])
        if "Tense" in feats:
            temps.add(feats["Tense"])
        lem = t.get("lemma", "_")
        if lem != "_":
            lemmas.add(lem)

    score = 0.0
    # UPOS diversity (max ~10)
    score += _W_UPSOS * min(len(upos_set) / 10, 1.0)
    # Bundle diversity (max ~20)
    score += _W_FEATS * min(len(bundles) / 20, 1.0)
    # Case diversity (max ~7)
    score += _W_CASES * min(len(cases) / 7, 1.0)
    # Tense diversity (max ~4)
    score += _W_TEMPS * min(len(temps) / 4, 1.0)
    # Lemma diversity ratio
    score += _W_LEMMA * min(len(lemmas) / max(n, 1), 1.0)
    # Length (cap at 30)
    score += _W_LENGTH * min(n / 30, 1.0)
    # Raw lemma count (cap at 15)
    score += _W_LEMMA_RAW * min(len(lemmas) / 15, 1.0)

    return round(score, 4)

# ---------------------------------------------------------------------------
# 5. Build & export
# ---------------------------------------------------------------------------

def build() -> None:
    print("Parsing CoNLL-U ...")
    sents = parse_silver(_SILVER)
    total = len(sents)
    n_full_lookup = sum(1 for s in sents.values() if s["full_lookup"])
    n_disambig = sum(1 for s in sents.values() if s["fully_disambiguated"])
    n_both = sum(1 for s in sents.values() if s["full_lookup"] and s["fully_disambiguated"])
    print(f"  {total} sentences, {n_full_lookup} full_lookup, {n_disambig} disambiguated, {n_both} both")

    print("Loading EN translations ...")
    trans = load_translations(_CORPUS)
    print(f"  {len(trans)} sentences with EN translation")
    n_trans_ok = sum(1 for v in trans.values() if v["translation_ok"])
    print(f"  {n_trans_ok} translation_ok (no heuristic flags)")

    print("Aggregating eval signals ...")
    eval_sig = _aggregate_eval_signals(_EVAL_DIR)
    print(f"  {len(eval_sig)} gold texts with eval data")

    print("Scoring & exporting ...")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_path = _OUT_DIR / "sentences_scored.jsonl"
    gold_path = _OUT_DIR / "quasi_gold.jsonl"

    records = []
    for text, sinfo in sents.items():
        tr = trans.get(text)
        ev = eval_sig.get(text, {})

        tokens = sinfo["tokens"]
        rich = _richness(tokens)
        f1 = ev.get("best_token_f1", 0.0)
        fr = ev.get("best_focus_recovery", 0.0)
        gold_score = round(rich + f1 + 0.5 * fr, 4)

        fl = sinfo["full_lookup"]
        fd = sinfo["fully_disambiguated"]
        tok = tr["translation_ok"] if tr else False

        rec = {
            "sent_id": sinfo["sent_id"],
            "text_clean": text,
            "text_norm": tr["text_norm"] if tr else "",
            "translation_en": tr["translation_en"] if tr else "",
            "frequency": tr["frequency"] if tr else 0,
            "n_words": len(tokens),
            "tokens": tokens,
            "full_lookup": fl,
            "fully_disambiguated": fd,
            "translation_ok": tok,
            "translation_flags": tr["translation_flags"] if tr else [],
            "eval": {
                "n_runs": ev.get("n_runs", 0),
                "best_token_f1": ev.get("best_token_f1", 0.0),
                "best_focus_recovery": ev.get("best_focus_recovery", 0.0),
                "ever_conllu_complete": ev.get("ever_conllu_complete", False),
            },
            "richness": rich,
            "gold_score": gold_score,
            "is_gold": fl and fd and tok,
        }
        records.append(rec)

    # Write full scored file
    with scored_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: (-r["gold_score"], r["sent_id"])):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write gold subset
    gold_recs = [r for r in records if r["is_gold"]]
    gold_recs.sort(key=lambda r: (-r["gold_score"], r["sent_id"]))
    with gold_path.open("w", encoding="utf-8") as fh:
        for rec in gold_recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print()
    print("=" * 60)
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Total sentences (CoNLL-U):     {total}")
    print(f"  full_lookup:                  {n_full_lookup}")
    print(f"  fully_disambiguated:          {n_disambig}")
    print(f"  both (lookup ∧ disambig):     {n_both}")
    print(f"EN translations available:      {len(trans)}")
    print(f"  translation_ok (no flags):    {n_trans_ok}")
    print(f"Eval data:                      {len(eval_sig)} sentences")
    print(f"  unique gold texts with eval:  {sum(1 for v in eval_sig.values() if v['n_runs'] > 0)}")
    print(f"  best_token_f1 ≥ 0.5:          {sum(1 for v in eval_sig.values() if v['best_token_f1'] >= 0.5)}")
    print(f"  best_token_f1 ≥ 0.8:          {sum(1 for v in eval_sig.values() if v['best_token_f1'] >= 0.8)}")
    print()
    print(f"Gold subset size:               {len(gold_recs)}")
    print(f"  (full_lookup ∧ disambig ∧ translation_ok)")
    print()
    print(f"Output:")
    print(f"  {scored_path}  ({total} records)")
    print(f"  {gold_path}   ({len(gold_recs)} records)")

    # Top-5 gold
    print()
    print("Top-5 by gold_score:")
    for r in gold_recs[:5]:
        print(f"  [{r['gold_score']:.3f}] {r['sent_id']}: {r['text_clean']!r}")
        print(f"         EN: {r['translation_en']!r}  richness={r['richness']:.3f}  "
              f"f1={r['eval']['best_token_f1']:.2f}  fr={r['eval']['best_focus_recovery']:.2f}")

    # Bottom-5 of gold (just above threshold)
    print()
    print("Bottom-5 of gold (just qualified):")
    for r in gold_recs[-5:]:
        print(f"  [{r['gold_score']:.3f}] {r['sent_id']}: {r['text_clean']!r}")
        print(f"         EN: {r['translation_en']!r}  richness={r['richness']:.3f}")

    # 5 filtered-out sentences (full_lookup ∧ disambig but NOT translation_ok)
    print()
    print("5 filtered-out (lookup ∧ disambig, no EN or bad translation):")
    rejected = [r for r in records if r["full_lookup"] and r["fully_disambiguated"] and not r["is_gold"]]
    rejected.sort(key=lambda r: (-r["gold_score"], r["sent_id"]))
    for r in rejected[:5]:
        flags = r["translation_flags"]
        reason = "no_en" if not r["translation_en"] else f"flags={flags}"
        print(f"  [{r['gold_score']:.3f}] {r['sent_id']}: {r['text_clean']!r}  ({reason})")

    # Eval cross-check: f1 ≥ 0.8 sentences
    high_f1 = [r for r in gold_recs if r["eval"]["best_token_f1"] >= 0.8]
    print()
    print(f"Gold sentences with best_token_f1 ≥ 0.8: {len(high_f1)}")
    for r in high_f1:
        print(f"  [{r['gold_score']:.3f}] {r['sent_id']}: {r['text_clean']!r}  "
              f"f1={r['eval']['best_token_f1']:.2f}  fr={r['eval']['best_focus_recovery']:.2f}")


if __name__ == "__main__":
    build()
