"""Cohere model adapter for inspect-ai.

Cohere's OpenAI-compatible API sends ``null`` (JSON) for optional tool
parameters that have no value, but the JSON-Schema validator in inspect-ai
rejects ``null`` against ``{"type": "string"}`` schemas.  This adapter
sanitises tool-call arguments *after* parsing (so the schema stays plain
``"type": "string"`` — Cohere rejects ``anyOf`` too) and *before* the
jsonschema validation step.

Usage::

    --model cohere/openai/$OPENAI_MODEL
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._openai import chat_choices_from_openai
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolInfo

from inspect_ai.model._model_output import ChatCompletionChoice
from openai.types.chat import ChatCompletion


def _sanitize_tool_args(choices: list[ChatCompletionChoice]) -> list[ChatCompletionChoice]:
    """Replace ``null`` values in tool-call arguments with ``""``.

    Cohere sends ``filter_tags: null`` / ``features: null`` for optional
    string parameters.  jsonschema rejects ``null`` against ``{"type":
    "string"}``.  We convert ``None → ""`` so validation passes and the
    downstream ``or None`` in the tool body maps it back correctly.
    """
    for choice in choices:
        msg = choice.message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                for key, val in list(tc.arguments.items()):
                    if val is None:
                        tc.arguments[key] = ""
    return choices


@modelapi(name="cohere")
class CohereAPI(OpenAICompatibleAPI):
    """OpenAI-compatible adapter with Cohere tool-argument sanitisation."""

    def chat_choices_from_completion(
        self, completion: ChatCompletion, tools: list[ToolInfo]
    ) -> list[ChatCompletionChoice]:
        choices = chat_choices_from_openai(completion, tools)
        return _sanitize_tool_args(choices)
