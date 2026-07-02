from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from sommelier.config import SommelierConfig
from sommelier.errors import UserInputError

if TYPE_CHECKING:
    import modal

PYTHON_VERSION = "3.13"

# Version pins are added after the first green remote smoke run per
# RFC-0007; until then the stacks below track latest releases.
BASE_PACKAGES = ("pydantic>=2.0", "pyyaml>=6.0")
DATA_PACKAGES = ("cudf-cu12",)
TRAIN_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
)
EVAL_PACKAGES = ("torch", "transformers", "datasets")
SERVING_PACKAGES = ("torch", "transformers", "peft", "fastapi", "uvicorn")

NVIDIA_INDEX_URL = "https://pypi.nvidia.com"

PipelineStage = Literal["data", "train", "eval"]


class RemoteStageOptions(TypedDict):
    gpu: str
    timeout: int


def _base_image() -> modal.Image:
    import modal

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*BASE_PACKAGES)
        .add_local_python_source("sommelier")
    )


def data_image() -> modal.Image:
    """GPU dataframe stack for coarse filtering plus the package source.

    cudf wheels come from the NVIDIA index; the image is only used by the
    remote data stage, never imported locally.
    """
    import modal

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*BASE_PACKAGES)
        .pip_install(*DATA_PACKAGES, extra_index_url=NVIDIA_INDEX_URL)
        .add_local_python_source("sommelier")
    )


def train_image() -> modal.Image:
    """Model loading, quantization, and adapter training stack."""
    return _base_image().pip_install(*TRAIN_PACKAGES)


def eval_image() -> modal.Image:
    """Deterministic generation, parser, and metrics stack.

    The v1 evaluation runner generates through transformers; vllm is not
    included until the runner actually uses it, keeping the image truthful
    about its capabilities.
    """
    return _base_image().pip_install(*EVAL_PACKAGES)


def serving_image() -> modal.Image:
    """Optional OpenAI-compatible adapter serving stack."""
    return _base_image().pip_install(*SERVING_PACKAGES)


def stage_options(config: SommelierConfig, stage: PipelineStage) -> RemoteStageOptions:
    """GPU and timeout hooks for one remote pipeline stage.

    Values come from the validated remote config; unknown stages fail
    explicitly instead of inheriting another stage's budget.
    """
    timeouts = {
        "data": config.remote.data_timeout_seconds,
        "train": config.remote.train_timeout_seconds,
        "eval": config.remote.eval_timeout_seconds,
    }
    if stage not in timeouts:
        raise UserInputError(
            f"unknown remote stage: {stage}",
            hint="Remote stage options exist for data, train, and eval.",
        )
    return RemoteStageOptions(gpu=config.remote.gpu, timeout=timeouts[stage])
