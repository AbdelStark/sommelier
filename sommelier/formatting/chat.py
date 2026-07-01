from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from sommelier.artifacts import ArtifactRef, make_artifact_ref
from sommelier.config import SommelierConfig
from sommelier.errors import ArtifactNotFoundError
from sommelier.run_context import (
    RunContext,
    read_jsonl_records,
    record_stage_success,
    write_jsonl_records,
)

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


def _format_prepared_example(
    example: dict[str, object],
    config: SommelierConfig,
) -> dict[str, object]:
    query = str(example["query"])
    gold_calls = example["gold_calls"]
    target_text = json.dumps(gold_calls, separators=(",", ":"), sort_keys=True)
    tools_json = json.dumps(example["tools"], separators=(",", ":"), sort_keys=True)
    system_content = f"{config.formatting.system_prompt.strip()}\n\nAvailable tools:\n{tools_json}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": target_text},
    ]
    prompt_text = json.dumps(messages[:2], separators=(",", ":"), sort_keys=True)
    full_text = json.dumps(messages, separators=(",", ":"), sort_keys=True)
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
