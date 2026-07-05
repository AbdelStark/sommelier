from __future__ import annotations

from pathlib import Path

import pytest

from sommelier.config import (
    DataConfig,
    DatasetSourceConfig,
    EvalConfig,
    FormattingConfig,
    ModelConfig,
    ProjectConfig,
    RemoteConfig,
    ReportConfig,
    SommelierConfig,
    TrainConfig,
)
from sommelier.data.gpu import coarse_filter_raw_rows
from sommelier.data.split import prepare_split_result
from sommelier.data.types import RawToolCallRow
from sommelier.errors import ExternalDependencyError


def _config(min_chars: int = 10, max_chars: int = 2000) -> SommelierConfig:
    return SommelierConfig(
        schema_version="sommelier.config.v2",
        project=ProjectConfig(name="test", artifact_root=Path("artifacts"), seed=1),
        model=ModelConfig(
            base_model_id="test/model",
            base_model_revision="main",
            tokenizer_revision="main",
        ),
        datasets=[
            DatasetSourceConfig(
                language="en", dataset_id="test/dataset", dataset_revision="main"
            )
        ],
        data=DataConfig(
            min_query_chars=min_chars,
            max_query_chars=max_chars,
            n_train=1,
            n_validation=1,
            n_test=1,
        ),
        formatting=FormattingConfig(system_prompt="prompt"),
        train=TrainConfig(target_modules=["q_proj"]),
        eval=EvalConfig(),
        remote=RemoteConfig(gpu="A10G"),
        report=ReportConfig(),
    )


def _row(source_id: str, query: str) -> RawToolCallRow:
    return RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=source_id,
        query=query,
        tools='[{"name":"lookup_weather","description":"d","parameters":{}}]',
        answers='[{"name":"lookup_weather","arguments":{"city":"Paris"}}]',
        source_revision="fixture",
    )


def test_gpu_path_requires_optional_dependency() -> None:
    config = _config()
    rows = [_row("a", "long enough query")]
    with pytest.raises(ExternalDependencyError):
        coarse_filter_raw_rows(rows, config)


def test_cpu_and_mocked_gpu_filter_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(min_chars=10, max_chars=30)
    rows = [
        _row("keep", "valid query here"),
        _row("drop-short", "tiny"),
        _row("drop-long", "this query is far too long to pass"),
    ]

    class FakeStringAccessor:
        def __init__(self, values: list[str]) -> None:
            self._values = values

        def len(self) -> FakeIntSeries:
            return FakeIntSeries([len(value) for value in self._values])

    class FakeIntSeries:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def __ge__(self, other: int) -> FakeBoolSeries:
            return FakeBoolSeries([value >= other for value in self._values])

        def __le__(self, other: int) -> FakeBoolSeries:
            return FakeBoolSeries([value <= other for value in self._values])

    class FakeQuerySeries:
        def __init__(self, values: list[str]) -> None:
            self._values = values

        @property
        def str(self) -> FakeStringAccessor:
            return FakeStringAccessor(self._values)

    class FakeBoolSeries:
        def __init__(self, values: list[bool]) -> None:
            self._values = values

    class FakeFrame:
        def __init__(self, records: list[dict[str, str]]) -> None:
            self._records = records

        def dropna(self, subset: list[str]) -> FakeFrame:
            kept = [
                record
                for record in self._records
                if all(record[key].strip() for key in subset)
            ]
            return FakeFrame(kept)

        def __getitem__(self, key: str | FakeBoolSeries) -> FakeQuerySeries | FakeFrame:
            if isinstance(key, str):
                return FakeQuerySeries([record[key] for record in self._records])
            return FakeFrame(
                [record for record, keep in zip(self._records, key._values, strict=True) if keep]
            )

        def to_pandas(self) -> FakeFrame:
            return self

        def to_dict(self, orient: str) -> list[dict[str, str]]:
            assert orient == "records"
            return self._records

    class FakeCudf:
        @staticmethod
        def DataFrame(records: list[dict[str, str]]) -> FakeFrame:
            return FakeFrame(records)

    monkeypatch.setitem(__import__("sys").modules, "cudf", FakeCudf())
    filtered = coarse_filter_raw_rows(rows, config)
    assert [row["source_id"] for row in filtered] == ["keep"]

    cpu_result = prepare_split_result(
        filtered,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=1,
        n_validation=0,
        n_test=0,
        seed=1,
    )
    assert len(cpu_result.train) == 1
