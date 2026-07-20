"""Corpus-reconstruction eval — the single configurable filter task.

Feeds the model an English gloss and scores its Old-Prussian output against the
gold subtitle.  The input surface is a knob (``instruct``):

* ``minimal``  — no system prompt, user message exactly ``"Translate: <en>"``,
  the only context is the four tool descriptions (smolagents draft).  This is
  the pure baseline.
* ``contract`` — adds a short output contract (use the tools; end with a single
  ``PRUSSIAN: <sentence>`` line).  Fixes empty/commentary finals.
* ``syntax``   — the Prussian syntax rules (``prompts/syntax_rules.txt``) as
  reference, nothing else (no contract).
* ``basevocab`` — contract + ``prompts/base_vocab.md`` + syntax rules
  (function words in context instead of burning search steps on them).
* ``toolguide`` — basevocab + tool-workflow guide (``_TOOL_GUIDE``:
  procedure, ``features`` parameter, FST tag legend — for models that
  don't infer the procedure from the tool descriptions alone).
* ``vocab``    — contract + base vocab + adverb rules + syntax rules.

The final answer is collected via a ``submit()`` tool (``basic_agent``): the
model must call ``submit(answer=<prussian sentence>)`` to finish, so the
candidate is a structured tool argument rather than regex-scraped prose.  The
scorer falls back to transcript extraction for models that never submit and
records which path was used (``candidate_source`` / ``submit_rate``).

Dataset: ``data/quasi_gold.jsonl`` (606 curated sentences).

Run::

    inspect eval evals/reconstruction.py --limit 3
    inspect eval evals/reconstruction.py -T instruct=syntax -T pos=VERB
    inspect view      # browse pass/fail transcripts

The ``pos="ADV"`` default is the adverb example (embedding-recovery probe);
``pos`` / ``instruct`` are knobs on the same pipeline.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser

import evals.providers.cohere  # noqa: F401 — registers the "cohere" model API
from inspect_ai.scorer import (
    SampleScore,
    Score,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import Generate, TaskState, basic_agent, solver

from prussian.adapters.agent.runner import extract_candidate
from prussian.tools import validate_tool

from evals.corpus_dataset import make_dataset
from prussian.adapters.inspect_tools import (
    get_word_forms,
    lookup_prussian_word,
    search_dictionary,
    validate_prussian,
)

# prompts/ lives in the sibling mcp/ checkout (see pyproject [tool.uv.sources]).
_PROMPTS = Path(__file__).resolve().parent.parent.parent / "mcp" / "prompts"

# Task statement — neutral register framing (reconstructed Prussian, no
# "Old Prussian (Palmaitis/Klussis system)" labelling, which misleads models
# into academic mode).  Lives in the user message so the surface stays under
# the eval's control (no system prompt).
_CONTRACT = (
    "Translate the sentence below into reconstructed Prussian (Prūsiskan).\n"
    "Do not answer from memory — every Prussian word must come from the "
    "tools: find each word with search_dictionary, fetch the exact "
    "inflected form with get_word_forms, and check your draft with "
    "validate_prussian.\n"
    "Then call submit() with ONLY the Prussian sentence."
)


# Tool workflow + FST tag legend — added after transcript analysis of
# command-r-08-2024 runs: without explicit workflow text the model never
# used filter_tags, rarely passed features, barely validated and never
# submitted.  Tool descriptions alone don't anchor the procedure for
# every model.  submit()/validate sequencing stays in _CONTRACT (it is
# part of the output contract, not of the tool API).
_TOOL_GUIDE = (
    "TOOL WORKFLOW (follow this for EVERY content word):\n"
    "\n"
    "a) Find the lemma: search_dictionary (source-language concept →\n"
    "   Prussian entries).  To analyze Prussian words you already have,\n"
    "   pass them to lookup_prussian_word instead (it can FST-analyze a\n"
    "   whole draft sentence at once).\n"
    "b) Determine the word's syntactic role (subject / object / attribute /\n"
    "   prepositional object …).\n"
    "c) Derive its case from role and government (preposition/verb); take\n"
    "   gender and number from the head noun.\n"
    "d) Fetch the EXACT inflected form with get_word_forms and its\n"
    "   `features` parameter, e.g. features=\"Akk+Sg+Masc\",\n"
    "   features=\"Nom+Pl+Fem\", features=\"Gen+Pl\", features=\"Ind+Pres+P3\".\n"
    "   Never use a bare lemma in the sentence and never pick a form\n"
    "   freehand — always request it via `features`.\n"
    "\n"
    "Never repeat a tool call with identical arguments — the result will\n"
    "be identical.  If a result lacks what you need, change the arguments\n"
    "or move on.\n"
    "If get_word_forms returns no matching form, that combination does\n"
    "not exist: choose the closest tags from `available_features`.\n"
    "Lemmas with empty `forms` are indeclinable — use them unchanged.\n"
    "\n"
    "FST TAG LEGEND (the values for `features` and `filter_tags`):\n"
    "\n"
    "- POS: N (noun), Adj (adjective), V (verb), Part (participle),\n"
    "  Pron (pronoun), Adv (adverb), Prp (preposition), Num (numeral)\n"
    "- Mood: Ind (indicative), Opt (optative), Imp (imperative),\n"
    "  Subj (subjunctive)\n"
    "- Tense: Pres (present), Pret (preterite), Inf (infinitive)\n"
    "- Case: Nom, Gen, Dat, Akk\n"
    "- Number: Sg, Pl — Gender: Masc, Fem, Neut — Person: P1, P2, P3\n"
    "- Other: Pass (passive), Refl (reflexive), Cmp (comparative),\n"
    "  Sup (superlative)\n"
    "\n"
    "Combine tags with \"+\": features=\"Akk+Sg+Fem\", filter_tags=\"Part+Pass\".\n"
    "3rd-person verb forms carry no Sg/Pl distinction — request them with\n"
    "P3 alone (\"Ind+Pres+P3\", never \"…+P3+Sg\")."
)


# Adverb formation — the corpus is adverb-focused (pos="ADV") and failure
# analysis showed models paraphrasing adverbs away (e.g. "en prūsisku"
# instead of prūsiskai).  Mirrors the ADVERBS block of the agent prompt.
_ADVERBS = (
    "ADVERBS (do not paraphrase them away):\n"
    "\n"
    '- "in language X" / "the X way" is ONE adverb in -iskai:\n'
    "  prūsiskai (in Prussian), leītawiskai (in Lithuanian),\n"
    "  miksiskai (in German) — never a prepositional phrase.\n"
    "- Manner adverbs come from the adjective: a-stem -s → -ai\n"
    "  (labs → labbai), -is → -ei, u-stem → -jai (grazzus → grazzjai).\n"
    "- Predicative state with būtwei takes the invariable neuter form\n"
    "  and a DATIVE experiencer: mennei ast labban (I feel good).\n"
    "- Check adverbs with lookup_prussian_word before submitting."
)


def _instruction(instruct: str) -> str:
    """Header text prepended to the ``Translate: <en>`` line (or '' for minimal)."""
    if instruct == "minimal":
        return ""
    if instruct == "contract":
        return _CONTRACT
    if instruct == "syntax":
        # The task statement plus the syntax rules as reference.  Rules
        # alone don't work: without a task line the model neither uses
        # the tools nor knows the target language (verified 2026-07-14:
        # submit-only transcripts, English echoed back).
        rules = (_PROMPTS / "syntax_rules.txt").read_text(encoding="utf-8").strip()
        return _CONTRACT + "\n\nSyntax rules to respect:\n\n" + rules
    if instruct == "basevocab":
        # Contract + base vocabulary + syntax rules ("vocab" without the
        # adverb block).  The base vocabulary is in context because trace
        # analysis showed function words cause search retry loops
        # (5 queries for "I") that blow up the step count and, with it,
        # the quadratic token bill.
        rules = (_PROMPTS / "syntax_rules.txt").read_text(encoding="utf-8").strip()
        vocab = (_PROMPTS / "base_vocab.md").read_text(encoding="utf-8").strip()
        return (
            _CONTRACT
            + "\n\n" + vocab
            + "\n\nSyntax rules to respect:\n\n" + rules
        )
    if instruct == "toolguide":
        # basevocab plus the tool-usage guide (see _TOOL_GUIDE).
        rules = (_PROMPTS / "syntax_rules.txt").read_text(encoding="utf-8").strip()
        vocab = (_PROMPTS / "base_vocab.md").read_text(encoding="utf-8").strip()
        return (
            _CONTRACT
            + "\n\n" + _TOOL_GUIDE
            + "\n\n" + vocab
            + "\n\nSyntax rules to respect:\n\n" + rules
        )
    if instruct == "vocab":
        # "syntax" plus the engine-verified base vocabulary (pronouns,
        # copula, particles, prepositions with government).  Failure
        # analysis of the syntax runs showed content words mostly hit
        # while discourse particles and function words missed.
        rules = (_PROMPTS / "syntax_rules.txt").read_text(encoding="utf-8").strip()
        vocab = (_PROMPTS / "base_vocab.md").read_text(encoding="utf-8").strip()
        return (
            _CONTRACT
            + "\n\n" + vocab
            + "\n\n" + _ADVERBS
            + "\n\nSyntax rules to respect:\n\n" + rules
        )
    raise ValueError(f"unknown instruct mode: {instruct!r}")


@solver
def build_prompt(instruct: str = "minimal", prefix: str = "Translate: "):
    """Set the user message from the sample input (no str.format — the syntax
    rules may contain braces)."""
    header = _instruction(instruct)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        content = (header + "\n\n" if header else "") + prefix + state.input_text
        state.messages = [ChatMessageUser(content=content)]
        return state

    return solve

# ── text helpers ──────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


_PRUSSIAN_LABEL = re.compile(r"PRUSSIAN:\s*(.+)", re.IGNORECASE)


def normalize(s: str | None) -> str:
    """Lowercase, drop punctuation, collapse whitespace; keep macron letters."""
    if not s:
        return ""
    return " ".join(_PUNCT.sub(" ", s).lower().split())


def submitted_answer(state: TaskState) -> str | None:
    """The answer the model passed to the ``submit()`` tool, or None if it
    never submitted.  Strips a ``PRUSSIAN:`` label should the model add one
    despite the tool description."""
    for m in reversed(state.messages):
        if m.role == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.function == "submit":
                    answer = (tc.arguments or {}).get("answer")
                    if isinstance(answer, str) and answer.strip():
                        answer = answer.strip()
                        label = _PRUSSIAN_LABEL.search(answer)
                        if label:
                            answer = label.group(1).strip()
                        return answer.strip().strip('"').strip("'")
    return None


def extract_from_transcript(state: TaskState) -> str:
    """Pull the Prussian candidate, robust to markdown and trailing empty/tool
    messages.  Scans all assistant messages for the last ``PRUSSIAN:`` line;
    falls back to the last non-empty assistant message via ``extract_candidate``.
    """
    assistant_texts = [
        (m.text or "") for m in state.messages if m.role == "assistant" and (m.text or "").strip()
    ]
    # last PRUSSIAN: line anywhere in the conversation (markdown stripped)
    for text in reversed(assistant_texts):
        clean = text.replace("`", " ").replace("*", " ")
        matches = list(_PRUSSIAN_LABEL.finditer(clean))
        if matches:
            return matches[-1].group(1).strip().strip('"').strip("'")
    # fallback: last non-empty assistant message
    if assistant_texts:
        return extract_candidate(assistant_texts[-1]) or ""
    return extract_candidate(state.output.completion) or ""


def token_f1(candidate: str, gold: str) -> float:
    c = Counter(normalize(candidate).split())
    g = Counter(normalize(gold).split())
    if not c or not g:
        return 0.0
    overlap = sum((c & g).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(c.values())
    rec = overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _char_ngrams(s: str, n: int) -> Counter:
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf_score(candidate: str, gold: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF), β=2 (recall-weighted)."""
    c_norm = normalize(candidate)
    g_norm = normalize(gold)
    if not c_norm or not g_norm:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        c_ng = _char_ngrams(c_norm, n)
        g_ng = _char_ngrams(g_norm, n)
        if not c_ng or not g_ng:
            continue
        overlap = sum((c_ng & g_ng).values())
        precisions.append(overlap / sum(c_ng.values()))
        recalls.append(overlap / sum(g_ng.values()))
    if not precisions:
        return 0.0
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    if avg_p + avg_r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


def _conllu_sentence_complete(sent: dict) -> bool:
    """A gold-like, whole sentence: exactly one root, no OOV, no unparsed head."""
    conllu = sent.get("conllu")
    if not conllu:
        return False
    if sent.get("coverage", {}).get("oov"):
        return False
    roots = 0
    for line in conllu.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 7 or not cols[0].isdigit():  # skip multiword ranges
            continue
        head = cols[6]
        if head == "0":
            roots += 1
        elif not head.lstrip("-").isdigit():
            return False  # unparsed / dangling head
    return roots == 1


def _candidate_is_whole(candidate: str) -> tuple[str | None, bool | None]:
    """Validate a non-matching candidate: (worst status, all-sentences-complete)."""
    if not candidate:
        return None, False
    try:
        parsed = json.loads(validate_tool(candidate, include_conllu=True))
    except Exception:
        return None, None
    sentences = parsed.get("sentences", [])
    status = (parsed.get("overall") or {}).get("status")
    complete = bool(sentences) and all(_conllu_sentence_complete(s) for s in sentences)
    return status, complete


# ── custom metrics (read Score.metadata; ignore samples where the value is None) ─


def _mean_of(key):
    def _metric(scores: list[SampleScore]) -> float:
        vals = []
        for s in scores:
            md = s.score.metadata
            if md and md.get(key) is not None:
                vals.append(float(md[key]))
        return sum(vals) / len(vals) if vals else 0.0

    return _metric


@metric
def focus_recovery_rate():
    """Mean fraction of gold focus-POS tokens (e.g. adverbs) that reappear."""
    return _mean_of("focus_recovery")


@metric
def conllu_complete_rate():
    """Among non-matching candidates, share that still parse as whole sentences."""
    return _mean_of("conllu_complete")


@metric
def mean_token_f1():
    return _mean_of("token_f1")


@metric
def submit_rate():
    """Share of samples whose candidate came from the submit() tool."""
    return _mean_of("submitted")


@metric
def mean_chrf():
    """Mean chrF score across samples."""
    return _mean_of("chrf")


# ── scorer ────────────────────────────────────────────────────────────────────


@scorer(metrics=[mean_chrf(), focus_recovery_rate(), conllu_complete_rate(), mean_token_f1(), submit_rate()])
def gold_match():
    async def score(state: TaskState, target: Target) -> Score:
        candidate = submitted_answer(state)
        candidate_source = "submit"
        if candidate is None:
            candidate = extract_from_transcript(state)
            candidate_source = "transcript"
        cand_norm = normalize(candidate)
        gold = target.text
        correct = bool(cand_norm) and cand_norm == normalize(gold)

        focus = (state.metadata or {}).get("focus", []) or []
        cand_tokens = set(cand_norm.split())
        recovered = [
            (normalize(f.get("form")) in cand_tokens)
            or (normalize(f.get("lemma")) in cand_tokens)
            for f in focus
        ]
        focus_recovery = (sum(recovered) / len(recovered)) if focus else None

        chrf = chrf_score(candidate, gold)
        correct = bool(cand_norm) and cand_norm == normalize(gold)

        meta: dict = {
            "candidate": candidate,
            "candidate_source": candidate_source,
            "submitted": candidate_source == "submit",
            "gold": gold,
            "chrf": chrf,
            "token_f1": token_f1(candidate, gold),
            "focus_recovery": focus_recovery,
            "conllu_status": None,
            "conllu_complete": None,
        }
        if not correct:
            import anyio

            status, complete = await anyio.to_thread.run_sync(_candidate_is_whole, candidate)
            meta["conllu_status"] = status
            meta["conllu_complete"] = complete

        missing = [f["form"] for f, ok in zip(focus, recovered) if not ok]
        explanation = (
            f"cand={candidate!r} [{candidate_source}] gold={gold!r} chrF={chrf:.2f} f1={meta['token_f1']:.2f}"
            + (f" missing_focus={missing}" if missing else "")
            + (f" conllu={meta['conllu_status']}/whole={meta['conllu_complete']}" if not correct else "")
        )
        return Score(
            value=chrf,
            answer=candidate,
            explanation=explanation,
            metadata=meta,
        )

    return score


# ── task ──────────────────────────────────────────────────────────────────────


@task
def reconstruction(
    pos: str | None = "ADV",
    instruct: str = "minimal",
    prefix: str = "Translate: ",
    message_limit: int = 80,
) -> Task:
    return Task(
        dataset=make_dataset(pos=pos),
        solver=basic_agent(
            # init replaces basic_agent's default system message — the
            # surface stays exactly build_prompt's user message.
            init=build_prompt(instruct=instruct, prefix=prefix),
            tools=[
                search_dictionary(),
                lookup_prussian_word(),
                get_word_forms(),
                validate_prussian(),
            ],
            max_attempts=1,
            # Each tool round costs 2 messages; thorough models need ~20
            # rounds on a 4-7-word sentence (gpt-oss-120b hit 40).  Models
            # that never submit() burn the full budget — cap via
            # -T message_limit=40 for such runs (input cost is quadratic
            # in rounds, no prompt caching on OpenAI-compatible proxies).
            message_limit=message_limit,
            submit_description=(
                "Submit the final Prussian translation. Pass ONLY the "
                "Prussian sentence as the answer — no commentary, no "
                "quotes, no label."
            ),
            continue_message=(
                "You have not submitted an answer yet. When you have the "
                "final Prussian translation, call submit() with the "
                "sentence as the answer."
            ),
        ),
        scorer=gold_match(),
        metadata={"instruct": instruct, "pos": pos},
    )
