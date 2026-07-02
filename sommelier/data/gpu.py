from __future__ import annotations

from pathlib import Path

from sommelier.config import SommelierConfig
from sommelier.data.load import load_raw_rows
from sommelier.data.types import RawToolCallRow
from sommelier.errors import ExternalDependencyError


def coarse_filter_raw_rows(
    raw_rows: list[RawToolCallRow],
    config: SommelierConfig,
) -> list[RawToolCallRow]:
    try:
        import cudf
    except ImportError as error:
        raise ExternalDependencyError(
            "GPU dataframe support requires the data-gpu optional extra",
            hint="Install with: uv sync --extra data-gpu",
        ) from error

    if not raw_rows:
        return []

    frame = cudf.DataFrame([dict(row) for row in raw_rows])
    frame = frame.dropna(subset=["query", "tools", "answers"])
    frame = frame[frame["query"].str.len() >= config.data.min_query_chars]
    frame = frame[frame["query"].str.len() <= config.data.max_query_chars]

    filtered: list[RawToolCallRow] = []
    for record in frame.to_pandas().to_dict(orient="records"):
        filtered.append(
            RawToolCallRow(
                schema_version="sommelier.raw_tool_call_row.v1",
                source_id=str(record["source_id"]),
                query=str(record["query"]),
                tools=str(record["tools"]),
                answers=str(record["answers"]),
                source_revision=str(record["source_revision"]),
            )
        )
    return filtered


def load_and_coarse_filter(config: SommelierConfig, input_path: Path) -> list[RawToolCallRow]:
    rows = load_raw_rows(input_path)
    return coarse_filter_raw_rows(rows, config)
