"""Modal entrypoint that builds the French paired dataset on a GPU.

Usage:

    SOMMELIER_GPU=L40S uv run modal run --detach remote_translate.py \
        --config examples/config.full.yaml --run-id fr-translate-1 \
        [--mode smoke --max-rows 2500] [--limit 50] \
        [--model-id mistralai/Mistral-Nemo-Instruct-2407]

Selection must match the pipeline run that will consume the rows: use the
same config, mode, and --max-rows as the intended remote_pipeline run so
the seeded split picks the same root examples. The full pipeline run
copies rows.fr.jsonl next to its exported root rows under the
<input stem>.fr.jsonl convention that data prepare expects.

The remote function exports the root dataset's raw rows, runs the same
validation, dedupe, and seeded split selection the pipeline uses (CPU,
seconds), translates exactly the selected rows with a pinned vLLM served
model, audits every output against its protected spans, and writes
rows.fr.jsonl plus translation_summary.json to the artifacts volume under
artifacts/translation/<run_id>/. Progress is checkpointed per row, so
re-running with the same run id resumes instead of restarting.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from sommelier.remote.images import translation_image

APP_NAME = "sommelier-translate"

GPU = os.environ.get("SOMMELIER_GPU", "L40S")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))

DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"
DEFAULT_MODEL_REVISION = "main"

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)

# Env vars are set inside the function at runtime: the image mounts the
# package source as its final layer, so no build step may follow it.
image = translation_image()


@app.function(
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/artifacts": artifacts_volume, "/hf-cache": hf_cache_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def run_remote_translation(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    limit: int,
) -> str:
    os.environ.setdefault("HF_HOME", "/hf-cache")
    os.environ.setdefault("VLLM_CACHE_ROOT", "/vllm-cache")

    from sommelier.config import load_config
    from sommelier.data.export import export_raw_rows
    from sommelier.data.load import load_raw_rows
    from sommelier.data.split import all_examples, prepare_split_result
    from sommelier.data.translate import (
        TranslatorInfo,
        load_vllm_translator,
        translate_rows,
        write_translation_outputs,
    )
    from sommelier.pipeline import apply_smoke_overrides

    work = Path("/artifacts/translation") / run_id
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "config.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")
    config = load_config(config_path)
    if mode == "smoke":
        config = apply_smoke_overrides(config)
    source = config.root_dataset

    rows_path = work / "rows.en.jsonl"
    if not rows_path.exists():
        exported = export_raw_rows(
            source, rows_path, seed=config.project.seed, max_rows=max_rows
        )
        print(f"[translate] exported {exported} raw rows", flush=True)
        artifacts_volume.commit()
    rows = load_raw_rows(rows_path)

    # The pipeline's own selection: same validation, dedupe, seed, and
    # counts, so the translated set is exactly the reference run's rows.
    root_result = prepare_split_result(
        rows,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
        language=source.language,
    )
    selected_ids = {example["example_id"] for example in all_examples(root_result)}
    selected = [row for row in rows if row["source_id"] in selected_ids]
    if limit:
        selected = selected[:limit]
    print(f"[translate] translating {len(selected)} selected rows", flush=True)

    translator = TranslatorInfo(
        model_id=model_id,
        model_revision=model_revision,
        max_new_tokens=max_new_tokens,
    )
    model = load_vllm_translator(translator)
    translated, stats = translate_rows(
        selected,
        model,
        progress_path=work / "translation_progress.jsonl",
        max_query_chars=config.data.max_query_chars,
    )
    rows_out, summary_path = write_translation_outputs(
        work,
        translated,
        stats,
        translator=translator,
        input_description=(
            f"{source.dataset_id}@{source.dataset_revision} via {rows_path.name}, "
            f"selected {len(selected)} rows with seed {config.project.seed}"
        ),
    )
    artifacts_volume.commit()
    print(f"[translate] stats: {stats}", flush=True)
    return f"rows={rows_out} summary={summary_path}"


@app.local_entrypoint()
def main(
    config: str = "examples/config.full.yaml",
    run_id: str = "fr-translate-1",
    mode: str = "full",
    max_rows: int = 0,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    max_new_tokens: int = 1024,
    limit: int = 0,
) -> None:
    config_yaml = Path(config).read_text(encoding="utf-8")
    result = run_remote_translation.remote(
        config_yaml,
        run_id,
        mode,
        max_rows,
        model_id,
        model_revision,
        max_new_tokens,
        limit,
    )
    print(result)
