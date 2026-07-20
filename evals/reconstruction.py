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
from inspect_ai.solver import Generate, TaskState, solver

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
    "First call consult_linguist() to analyse the sentence structure and "
    "identify which words you need to look up. Do NOT call search_dictionary "
    "for any word listed in the BASE VOCABULARY or in the WORDS FOUND SO FAR "
    "section — those forms are already exact. Then find each remaining word "
    "with search_dictionary, fetch the exact inflected form with "
    "get_word_forms, and check your draft with validate_prussian.\n"
    "When you are done, briefly explain your translation choices (which "
    "case / ending you picked and why), then call submit() with ONLY the "
    "Prussian sentence."
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


# ── compacting agent ──────────────────────────────────────────────────────────

from inspect_ai.model import execute_tools, get_model
from inspect_ai.tool._tool import tool
from inspect_ai.tool._tool_with import tool_with


_SUBMIT_NAME = "submit"
_SUBMIT_DESCRIPTION = (
    "Submit the final Prussian translation. Pass ONLY the "
    "Prussian sentence as the answer — no commentary, no "
    "quotes, no label."
)


def _tool_name(t):
    ri = getattr(t, "__registry_info__", None)
    if ri is not None:
        return ri.name
    return getattr(t, "__name__", repr(t))


@tool
def consult_linguist():
    async def execute(question: str, analysis: str) -> str:
        """Consult a Prussian linguistics expert. Call this to analyse the
        sentence structure, identify which words need lookup vs. which are
        in the base vocabulary, and plan your translation approach. After
        your initial analysis, proceed directly with search_dictionary and
        get_word_forms — do not call consult_linguist again unless you are
        stuck.

        Args:
          question: The linguistic question or sentence you are working on.
          analysis: Your analysis of the sentence: structure, which words
            to look up, which are in the base vocab, and your translation
            plan.
        """
        return f"Analysis recorded. Proceed with your plan: {analysis[:120]}"

    return execute


def _extract_findings(messages, last_assistant_msg):
    """Parse tool-call arguments + tool-result JSON from the last round.

    Returns a dict with keys: analyses, words, forms, lookup, validation.
    """
    findings: dict = {
        "analyses": [],
        "words": [],
        "forms": [],
        "lookup": [],
        "validation": None,
    }

    # collect tool-call arguments from the last assistant message
    tcs = getattr(last_assistant_msg, "tool_calls", None) or []
    call_args: dict[str, list[dict]] = {}
    for tc in tcs:
        call_args.setdefault(tc.function, []).append(tc.arguments or {})

    # consult_linguist → analysis text
    for args in call_args.get("consult_linguist", []):
        a = args.get("analysis") or ""
        q = args.get("question") or ""
        if a:
            findings["analyses"].append(a if not q else f"Q: {q}\n{a}")

    # tool results (role=tool messages after the last assistant message)
    tool_msgs = [m for m in messages if m.role == "tool"]
    # map tool_call_id → function name from the assistant message
    id_to_fn = {tc.id: tc.function for tc in tcs}

    for tm in tool_msgs:
        fn = tm.function or id_to_fn.get(getattr(tm, "tool_call_id", None), "")
        raw = tm.content
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            content = " ".join(
                c.text if hasattr(c, "text") else str(c)
                for c in raw
            )
        else:
            content = str(raw or "")
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        if fn == "search_dictionary":
            if isinstance(data, list):
                # only take the top result per query (search returns 10)
                for entry in data[:1]:
                    word = entry.get("word", "")
                    trs = entry.get("translations", {})
                    engl = ", ".join(trs.get("engl", [])) or ", ".join(
                        v for vs in trs.values() for v in (vs if isinstance(vs, list) else [vs])
                    )
                    if word:
                        findings["words"].append(f"{word} ({engl})" if engl else word)

        elif fn == "get_word_forms":
            if isinstance(data, list):
                for entry in data:
                    lemma = entry.get("lemma", "")
                    forms = entry.get("forms", [])
                    form_strs = [
                        f"{f.get('form','')} [{', '.join(f.get('tags',[]))}]"
                        for f in forms
                        if f.get("form")
                    ]
                    if lemma:
                        desc = entry.get("desc", "")
                        findings["forms"].append(
                            f"{lemma}" + (f" ({desc})" if desc else "")
                            + (f": {'; '.join(form_strs)}" if form_strs else " — no forms matched")
                        )

        elif fn == "lookup_prussian_word":
            if isinstance(data, list):
                for tok in data:
                    surface = tok.get("token", tok.get("word", ""))
                    lemma = tok.get("lemma", "")
                    tags = ", ".join(tok.get("tags", []))
                    if surface:
                        findings["lookup"].append(
                            f"{surface}" + (f" → {lemma}" if lemma else "")
                            + (f" [{tags}]" if tags else "")
                        )

        elif fn == "validate_prussian":
            overall = data.get("overall", {}) if isinstance(data, dict) else {}
            status = overall.get("status", "")
            n_viol = overall.get("n_violations", "?")
            sentences = data.get("sentences", []) if isinstance(data, dict) else []
            violations = []
            for s in sentences:
                for v in s.get("violations", []):
                    violations.append(
                        f"{v.get('check','')}: {v.get('message','')}"
                    )
            val = f"status={status}, violations={n_viol}"
            if violations:
                val += "\n  " + "\n  ".join(violations)
            findings["validation"] = val

    return findings


def _build_compacted_message(header: str, findings: dict, prefix: str, input_text: str) -> str:
    """Build a fresh user message: original header + findings block + translate line."""
    parts = [header]

    # accumulate findings sections
    sections: list[str] = []
    if findings["analyses"]:
        sections.append(
            "PREVIOUS ANALYSIS (from consult_linguist):\n"
            + "\n---\n".join(findings["analyses"])
        )
    if findings["words"]:
        sections.append(
            "WORDS FOUND SO FAR (do NOT search for these again):\n"
            + "\n".join(f"- {w}" for w in findings["words"])
        )
    if findings["forms"]:
        sections.append(
            "FORMS RETRIEVED:\n"
            + "\n".join(f"- {f}" for f in findings["forms"])
        )
    if findings["lookup"]:
        sections.append(
            "WORDS CHECKED (lookup_prussian_word):\n"
            + "\n".join(f"- {l}" for l in findings["lookup"])
        )
    if findings["validation"]:
        sections.append(
            "LAST VALIDATION:\n"
            + findings["validation"]
        )

    if sections:
        parts.append("\n\n".join(sections))

    parts.append(prefix + input_text)
    return "\n\n".join(parts)


@solver
def compacting_agent(tools, init=None, message_limit=80, max_rounds=12, prefix="Translate: "):
    """Agent that compacts the chat history after every tool round.

    After each round of tool calls, the entire message history is replaced
    with a single user message that contains the original instruction
    header plus an accumulated 'findings' block (words found, forms
    retrieved, validation status, prior analyses).  This keeps token cost
    linear (not quadratic) and prevents Cohere command-r from re-firing
    the same searches on a growing history.
    """

    @tool
    def submit():
        async def execute(answer: str) -> str:
            """Submit an answer for evaluation.

            Args:
              answer (str): Submitted answer
            """
            return answer

        return execute

    all_tools = [*tools, tool_with(submit(), _SUBMIT_NAME, _SUBMIT_DESCRIPTION)]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.message_limit = message_limit or state.message_limit or 50

        # apply init (build initial user prompt)
        if init:
            state = await init(state, generate)

        # capture original header (everything before the "Translate: " line)
        original_content = state.messages[0].content if state.messages else ""
        header = original_content
        # strip the trailing "Translate: <text>" to get the header
        # (init puts it at the end)
        prefix_marker = original_content.rfind(prefix)
        if prefix_marker >= 0:
            header = original_content[:prefix_marker].rstrip()

        input_text = state.input_text
        model = get_model()
        findings: dict = {
            "analyses": [],
            "words": [],
            "forms": [],
            "lookup": [],
            "validation": None,
        }

        for _round in range(max_rounds):
            # drop consult_linguist after round 2 to force progression
            round_tools = all_tools if _round < 2 else [
                t for t in all_tools
                if _tool_name(t) != "consult_linguist"
            ]
            # generate with current (single) user message
            state.output = await model.generate(
                input=state.messages,
                tools=round_tools,
            )

            if state.output.stop_reason == "model_length":
                break

            msg = state.output.message
            tcs = getattr(msg, "tool_calls", None) or []

            if not tcs:
                # model produced text, no tool calls → treat as final answer
                state.messages.append(msg)
                break

            # check for submit() before executing tools
            submitted = any(tc.function == _SUBMIT_NAME for tc in tcs)
            # execute tools
            result = await execute_tools([msg], all_tools)
            if result.output:
                state.output = result.output

            if submitted:
                # keep the submit call + result in the transcript
                state.messages.append(msg)
                state.messages.extend(result.messages)
                break

            # extract findings from this round
            round_findings = _extract_findings(result.messages, msg)
            for k in ("analyses", "words", "forms", "lookup"):
                # dedup while preserving order
                seen = set(findings[k])
                for item in round_findings[k]:
                    if item not in seen:
                        findings[k].append(item)
                        seen.add(item)
            if round_findings["validation"]:
                findings["validation"] = round_findings["validation"]

            # compact: replace entire history with a fresh user message
            compacted = _build_compacted_message(header, findings, prefix, input_text)
            state.messages = [ChatMessageUser(content=compacted)]

        return state

    return solve


@task
def reconstruction(
    pos: str | None = "ADV",
    instruct: str = "minimal",
    prefix: str = "Translate: ",
    message_limit: int = 80,
) -> Task:
    return Task(
        dataset=make_dataset(pos=pos),
        solver=compacting_agent(
            tools=[
                consult_linguist(),
                search_dictionary(),
                lookup_prussian_word(),
                get_word_forms(),
                validate_prussian(),
            ],
            init=build_prompt(instruct=instruct, prefix=prefix),
            message_limit=message_limit,
            prefix=prefix,
        ),
        scorer=gold_match(),
        metadata={"instruct": instruct, "pos": pos},
    )
