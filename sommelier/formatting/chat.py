from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from sommelier.artifacts import ArtifactRef, make_artifact_ref
from sommelier.config import SommelierConfig
from sommelier.data.types import ToolCall
from sommelier.errors import ArtifactNotFoundError, SchemaValidationError
from sommelier.run_context import (
    RunContext,
    read_jsonl_records,
    record_stage_success,
    write_jsonl_records,
)

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def canonical_json(payload: object) -> str:
    """Serializes a payload as canonical JSON.

    Canonical form uses sorted keys, compact separators, and ASCII escapes,
    so byte-identical inputs always produce byte-identical prompt text and
    digests (RFC-0003).
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def validate_gold_calls(gold_calls: object, *, context: str) -> list[ToolCall]:
    """Validates the gold tool calls used as the assistant target.

    Requires a non-empty list (INV-DATA-004) of objects shaped like
    ``{"name": str, "arguments": dict}``. Raises SchemaValidationError
    instead of repairing malformed calls.
    """
    if not isinstance(gold_calls, list) or not gold_calls:
        raise SchemaValidationError(
            f"{context}: gold_calls must be a non-empty list",
            hint="Prepared examples must contain at least one gold tool call.",
        )
    calls: list[ToolCall] = []
    for index, call in enumerate(gold_calls):
        if not isinstance(call, dict):
            raise SchemaValidationError(
                f"{context}: gold_calls[{index}] must be an object",
                hint="Each gold call needs a name and an arguments object.",
            )
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise SchemaValidationError(
                f"{context}: gold_calls[{index}].name must be a non-empty string",
                hint="Each gold call needs a name and an arguments object.",
            )
        if not isinstance(arguments, dict):
            raise SchemaValidationError(
                f"{context}: gold_calls[{index}].arguments must be an object",
                hint="Each gold call needs a name and an arguments object.",
            )
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def validate_assistant_target(content: str, *, context: str = "assistant target") -> list[ToolCall]:
    """Rejects assistant targets that are not exactly the canonical tool call JSON.

    Explanatory prefixes or suffixes, markdown fences, or non-canonical key
    ordering all fail: the content must parse as JSON and re-serialize to the
    identical canonical string (RFC-0003: the target is only the JSON call).
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise SchemaValidationError(
            f"{context}: assistant target is not valid JSON",
            hint="The assistant target must contain only the JSON tool call.",
        ) from error
    calls = validate_gold_calls(parsed, context=context)
    if canonical_json(parsed) != content:
        raise SchemaValidationError(
            f"{context}: assistant target is not canonical JSON",
            hint="Serialize the target with sorted keys and compact separators.",
        )
    return calls


def build_messages(
    *,
    query: str,
    tools: list[object],
    gold_calls: object,
    system_prompt: str,
    context: str = "example",
) -> list[ChatMessage]:
    """Builds the three-message chat sequence for one prepared example.

    Per RFC-0003: the system message carries the instruction plus the
    canonical JSON tool schemas, the user message carries the raw query, and
    the assistant message carries only the canonical JSON gold calls.
    """
    calls = validate_gold_calls(gold_calls, context=context)
    tools_json = canonical_json(tools)
    system_content = f"{system_prompt.strip()}\n\nAvailable tools:\n{tools_json}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": canonical_json(calls)},
    ]


def _format_prepared_example(
    example: dict[str, object],
    config: SommelierConfig,
) -> dict[str, object]:
    example_id = str(example.get("example_id", "<unknown>"))
    tools = example.get("tools")
    if not isinstance(tools, list):
        raise SchemaValidationError(
            f"example {example_id}: tools must be a list",
            hint="Regenerate the prepared split with the current pipeline version.",
        )
    messages = build_messages(
        query=str(example["query"]),
        tools=tools,
        gold_calls=example["gold_calls"],
        system_prompt=config.formatting.system_prompt,
        context=f"example {example_id}",
    )
    target_text = messages[2]["content"]
    prompt_text = canonical_json(messages[:2])
    full_text = canonical_json(messages)
    return {
        "schema_version": "sommelier.formatted_example.v1",
        "example_id": example["example_id"],
        "split": example["split"],
        "messages": messages,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "full_text": full_text,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "tokenizer_id": config.model.base_model_id,
        "tokenizer_revision": config.model.tokenizer_revision,
        "template_policy": config.formatting.template_policy,
    }


def build_formatted_splits_fixture(
    config: SommelierConfig,
    *,
    data_dir: Path,
    out_dir: Path,
    context: RunContext,
    command: list[str],
) -> list[ArtifactRef]:
    if not data_dir.exists():
        raise ArtifactNotFoundError(
            f"prepared data directory not found: {data_dir}",
            hint="Run sommelier data prepare before format build.",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    input_refs: list[ArtifactRef] = []
    output_refs: list[ArtifactRef] = []

    for split in SPLITS:
        split_path = data_dir / f"{split}.jsonl"
        records = read_jsonl_records(split_path)
        input_refs.append(
            make_artifact_ref(
                split_path,
                artifact_root=context.artifact_root,
                kind="dataset_split",
                schema_version="sommelier.prepared_example.v1",
            )
        )
        formatted_records = [_format_prepared_example(record, config) for record in records]
        formatted_path = out_dir / f"{split}.jsonl"
        write_jsonl_records(formatted_path, formatted_records)
        output_refs.append(
            make_artifact_ref(
                formatted_path,
                artifact_root=context.artifact_root,
                kind="formatted_split",
                schema_version="sommelier.formatted_example.v1",
            )
        )

    record_stage_success(
        context,
        stage="format",
        command=command,
        seed=config.project.seed,
        inputs=input_refs,
        outputs=output_refs,
    )
    return output_refs
