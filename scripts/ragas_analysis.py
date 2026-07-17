#!/usr/bin/env python3
"""Post-process inspect-ai eval logs with RAGAS-style metrics.

Two tiers of metrics:

  Deterministic (no LLM, no env vars needed beyond loading the engine):
    context_recall_lemma  — fraction of gold lemmata that appear in the
                            search_dictionary results for this sample.
                            Gold lemmata are obtained by calling
                            lookup_prussian_word(gold_sentence) — no manual
                            annotation needed.
    faithfulness_lexical  — fraction of candidate words (after normalization)
                            that appear somewhere in any tool output for this
                            sample.  A word absent from all tool outputs is a
                            likely hallucination.

  LLM-based (needs OPENAI_API_KEY, pass --llm):
    faithfulness_llm      — RAGAS Faithfulness: LLM decomposes answer into
                            atomic claims, checks each against contexts.
    context_recall_llm    — RAGAS LLMContextRecall: checks whether gold
                            statements are attributable to retrieved contexts.

Usage:
    source env.hf-voyage.sh          # sets OPENAI_API_KEY + embedding vars
    uv run python scripts/ragas_analysis.py logs/
    uv run python scripts/ragas_analysis.py logs/ --llm          # + LLM metrics
    uv run python scripts/ragas_analysis.py logs/specific.eval   # one file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ── text helpers (mirrored from evals/reconstruction.py) ──────────────────────

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(_PUNCT.sub(" ", s).lower().split())


def _norm_tokens(s: str) -> set[str]:
    return set(_normalize(s).split())


# ── inspect-ai log extraction ──────────────────────────────────────────────────


def _msg_role(msg: Any) -> str:
    return getattr(msg, "role", "") or ""


def _msg_function(msg: Any) -> str:
    return getattr(msg, "function", "") or ""


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(item.text or "")
        return "\n".join(parts)
    return str(content)


def extract_tool_outputs(messages: list[Any]) -> dict[str, list[str]]:
    """Return {tool_name: [output_str, ...]} for every tool response."""
    result: dict[str, list[str]] = {}
    for msg in messages:
        if _msg_role(msg) != "tool":
            continue
        fn = _msg_function(msg)
        text = _msg_text(msg)
        if fn and text.strip():
            result.setdefault(fn, []).append(text)
    return result


def extract_submitted(messages: list[Any]) -> str | None:
    """Return the answer passed to submit(), or None."""
    for msg in reversed(messages):
        if _msg_role(msg) != "assistant":
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            if getattr(tc, "function", None) == "submit":
                args = getattr(tc, "arguments", None) or {}
                answer = args.get("answer", "")
                if isinstance(answer, str) and answer.strip():
                    return answer.strip().strip('"').strip("'")
    return None


def candidate_from_score(sample: Any) -> str | None:
    """Read the pre-extracted candidate from the gold_match scorer metadata.

    The existing scorer already does submit()/transcript fallback extraction
    and stores the result in metadata['candidate'].  Re-using that is cheaper
    and more accurate than redoing the extraction here.
    """
    scores = getattr(sample, "scores", None) or {}
    gm = scores.get("gold_match")
    if gm is None:
        return None
    meta = getattr(gm, "metadata", None) or {}
    candidate = meta.get("candidate")
    return candidate if isinstance(candidate, str) else None


# ── gold-lemma extraction via lookup_prussian_word ─────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from prussian.engine.search import SearchEngine
        _engine = SearchEngine()
    return _engine


def gold_lemmata(gold_sentence: str) -> list[str]:
    """Return FST lemmata for the gold Prussian sentence via lookup_tool."""
    from prussian.tools import lookup_tool

    engine = _get_engine()
    raw = lookup_tool(engine, gold_sentence, fuzzy=False)
    lemmata = []
    for token in raw:
        for analysis in token.get("analyses") or []:
            lemma = analysis.get("lemma")
            if lemma:
                lemmata.append(_normalize(lemma))
    return list(dict.fromkeys(lemmata))  # deduplicated, order preserved


# ── deterministic metrics ──────────────────────────────────────────────────────


def faithfulness_lexical(candidate: str, all_tool_outputs: dict[str, list[str]]) -> float:
    """Fraction of candidate tokens grounded in retrieval outputs.

    Only search_dictionary and get_word_forms are counted as "retrieval".
    validate_prussian and lookup_prussian_word are excluded: they echo the
    candidate back, which would make the metric circular.

    Returns None when no retrieval calls were made at all.
    """
    cand_tokens = _norm_tokens(candidate)
    if not cand_tokens:
        return 0.0
    retrieval_texts = (
        all_tool_outputs.get("search_dictionary", [])
        + all_tool_outputs.get("get_word_forms", [])
    )
    if not retrieval_texts:
        return 0.0  # no retrieval → nothing is grounded
    combined = " ".join(retrieval_texts)
    context_tokens = _norm_tokens(combined)
    grounded = cand_tokens & context_tokens
    return len(grounded) / len(cand_tokens)


def context_recall_lemma(
    lemmata: list[str],
    search_outputs: list[str],
) -> float | None:
    """Fraction of gold lemmata that appear in the search_dictionary outputs.

    Returns None when the gold sentence has no FST lemmata (OOV-only sentence).
    """
    if not lemmata:
        return None
    combined = _normalize(" ".join(search_outputs))
    context_tokens = set(combined.split())
    found = sum(1 for lem in lemmata if lem in context_tokens)
    return found / len(lemmata)


# ── LLM-based RAGAS metrics ────────────────────────────────────────────────────


async def _ragas_scores(sample_data: dict, llm: Any) -> dict[str, float]:
    """Compute RAGAS Faithfulness + LLMContextRecall for one sample."""
    from ragas.metrics import Faithfulness, LLMContextRecall
    from ragas.dataset_schema import SingleTurnSample

    faithfulness_metric = Faithfulness(llm=llm)
    recall_metric = LLMContextRecall(llm=llm)

    sample = SingleTurnSample(
        user_input=sample_data["question"],
        response=sample_data["answer"] or "",
        retrieved_contexts=sample_data["search_contexts"],
        reference=sample_data["reference"],
    )
    f = await faithfulness_metric.single_turn_ascore(sample)
    cr = await recall_metric.single_turn_ascore(sample)
    return {"faithfulness_llm": float(f), "context_recall_llm": float(cr)}


async def compute_llm_metrics(samples: list[dict]) -> list[dict]:
    """Add LLM-based RAGAS metrics to each sample dict (mutates in place)."""
    import os
    from ragas.llms import LangchainLLMWrapper
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("langchain-openai not installed; skipping LLM metrics", file=sys.stderr)
        return samples

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    llm = LangchainLLMWrapper(ChatOpenAI(**kwargs))

    for s in samples:
        try:
            scores = await _ragas_scores(s, llm)
            s.update(scores)
        except Exception as exc:
            print(f"  LLM metric failed for sample: {exc}", file=sys.stderr)
            s["faithfulness_llm"] = None
            s["context_recall_llm"] = None
    return samples


# ── per-file analysis ──────────────────────────────────────────────────────────


def analyse_log(log_path: Path, use_engine: bool = True) -> list[dict]:
    """Load one .eval file and compute all deterministic metrics."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    samples_out = []

    for sample in (log.samples or []):
        messages = list(sample.messages or [])
        tool_outputs = extract_tool_outputs(messages)
        search_outputs = tool_outputs.get("search_dictionary", [])
        all_outputs_flat = {k: v for k, v in tool_outputs.items()}

        # Prefer the candidate already extracted by gold_match scorer (handles
        # both submit() calls and transcript fallback).  Fall back to our own
        # submit() extractor for logs that have no scorer results yet.
        candidate = candidate_from_score(sample)
        if candidate is None:
            candidate = extract_submitted(messages)
        reference = sample.target if isinstance(sample.target, str) else (
            (sample.target.text if hasattr(sample.target, "text") else str(sample.target))
        )
        gold_raw = (sample.metadata or {}).get("gold", reference)

        lemmata: list[str] = []
        cr_lemma: float | None = None
        if use_engine:
            try:
                lemmata = gold_lemmata(gold_raw)
                cr_lemma = context_recall_lemma(lemmata, search_outputs)
            except Exception as exc:
                print(f"  lookup failed: {exc}", file=sys.stderr)

        faith_lex = faithfulness_lexical(candidate or "", all_outputs_flat)

        samples_out.append({
            "input": sample.input if isinstance(sample.input, str) else str(sample.input),
            "question": sample.input if isinstance(sample.input, str) else str(sample.input),
            "answer": candidate or "",
            "reference": reference,
            "gold_lemmata": lemmata,
            "search_contexts": search_outputs,
            "n_search_calls": len(search_outputs),
            "n_tool_calls_total": sum(len(v) for v in tool_outputs.values()),
            "faithfulness_lexical": faith_lex,
            "context_recall_lemma": cr_lemma,
        })

    return samples_out


# ── reporting ──────────────────────────────────────────────────────────────────


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  n/a"
    if pct:
        return f"{v:.1%}"
    return f"{v:.2f}"


def print_report(log_path: Path, samples: list[dict]) -> None:
    print(f"\n{'─'*70}")
    print(f"  {log_path.name}  ({len(samples)} samples)")
    print(f"{'─'*70}")

    faith_lex = _mean([s["faithfulness_lexical"] for s in samples])
    cr_lemma = _mean([s["context_recall_lemma"] for s in samples])
    faith_llm = _mean([s.get("faithfulness_llm") for s in samples])
    cr_llm = _mean([s.get("context_recall_llm") for s in samples])
    avg_search = _mean([float(s["n_search_calls"]) for s in samples])
    avg_tools = _mean([float(s["n_tool_calls_total"]) for s in samples])

    print(f"  faithfulness_lexical   {_fmt(faith_lex)}   (candidate tokens in tool outputs)")
    print(f"  context_recall_lemma   {_fmt(cr_lemma)}   (gold lemmata in search results)")
    if faith_llm is not None:
        print(f"  faithfulness_llm       {_fmt(faith_llm)}   (RAGAS LLM judge)")
    if cr_llm is not None:
        print(f"  context_recall_llm     {_fmt(cr_llm)}   (RAGAS LLM judge)")
    print(f"  avg search_dict calls  {_fmt(avg_search, pct=False)}")
    print(f"  avg total tool calls   {_fmt(avg_tools, pct=False)}")

    print()
    for i, s in enumerate(samples):
        faith = _fmt(s["faithfulness_lexical"])
        cr = _fmt(s["context_recall_lemma"])
        q = s["input"][:50].replace("\n", " ")
        cand = (s["answer"] or "")[:40]
        faith_llm_str = f"  llm_faith={_fmt(s.get('faithfulness_llm'))}" if "faithfulness_llm" in s else ""
        print(f"  [{i+1}] q={q!r}")
        print(f"       cand={cand!r}")
        print(f"       faith_lex={faith}  cr_lemma={cr}{faith_llm_str}")
        if s["gold_lemmata"]:
            found = [
                lem for lem in s["gold_lemmata"]
                if any(lem in _normalize(ctx) for ctx in s["search_contexts"])
            ]
            missing = [l for l in s["gold_lemmata"] if l not in found]
            if missing:
                print(f"       missing_lemmata={missing}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="logs/", help="Directory of .eval files or single .eval file (default: logs/)")
    parser.add_argument("--llm", action="store_true", help="Also compute LLM-based RAGAS metrics (Faithfulness, LLMContextRecall)")
    parser.add_argument("--no-engine", action="store_true", help="Skip loading the SearchEngine (skips context_recall_lemma)")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="Write per-sample results as JSON Lines to FILE")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        log_files = sorted(target.glob("*.eval"))
    elif target.suffix == ".eval":
        log_files = [target]
    else:
        print(f"No .eval files found at {target}", file=sys.stderr)
        sys.exit(1)

    if not log_files:
        print(f"No .eval files found in {target}", file=sys.stderr)
        sys.exit(1)

    use_engine = not args.no_engine
    all_samples: list[dict] = []

    for log_path in log_files:
        print(f"Reading {log_path.name} …", end=" ", flush=True)
        try:
            samples = analyse_log(log_path, use_engine=use_engine)
            print(f"{len(samples)} samples")
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            continue

        if args.llm and samples:
            import asyncio
            samples = asyncio.run(compute_llm_metrics(samples))

        print_report(log_path, samples)
        all_samples.extend({"log": log_path.name, **s} for s in samples)

    if args.json_out:
        out = Path(args.json_out)
        with out.open("w", encoding="utf-8") as fh:
            for s in all_samples:
                # strip large context lists for readability
                row = {k: v for k, v in s.items() if k not in ("search_contexts",)}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(all_samples)} rows → {out}")


if __name__ == "__main__":
    main()
