from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypedDict

from sommelier.config import SommelierConfig
from sommelier.errors import UserInputError

if TYPE_CHECKING:
    import modal

# ``from_registry(..., add_python=...)`` accepts a major.minor selector, while
# ``debian_slim`` accepts an exact patch. Keep the registry selector separate
# so rebuilding the evidence pipeline cannot silently advance its Python patch
# and then fail the runtime-identity gate after a paid allocation starts.
PYTHON_VERSION = "3.13"
PIPELINE_PYTHON_VERSION: Final = "3.13.3"

# Base and vLLM versions are the exact environment observed in the first
# successful Hebrew decoder-path smoke. The pipeline versions are the exact
# compatible environment observed by its first end-to-end remote probe.
BASE_PACKAGES = ("pydantic==2.13.4", "PyYAML==6.0.3")
DATA_PACKAGES = ("cudf-cu12",)
TRAIN_PACKAGES = (
    "torch==2.13.0",
    "transformers==5.13.1",
    "tokenizers==0.22.2",
    "accelerate==1.14.0",
    "peft==0.19.1",
    "bitsandbytes==0.49.2",
    "datasets==5.0.0",
    "huggingface_hub==1.23.0",
)

# Immutable expected identity for full Hebrew pipeline evidence. Keeping this
# adjacent to the image definition makes it difficult for the install and
# runtime gates to drift apart. The Debian pipeline image pins this exact
# Python patch; registry-based auxiliary images retain the major.minor selector.
PIPELINE_RUNTIME_VERSIONS: Final = (
    ("python", PIPELINE_PYTHON_VERSION),
    ("torch", "2.13.0"),
    ("transformers", "5.13.1"),
    ("tokenizers", "0.22.2"),
    ("accelerate", "1.14.0"),
    ("peft", "0.19.1"),
    ("bitsandbytes", "0.49.2"),
    ("datasets", "5.0.0"),
    ("huggingface_hub", "1.23.0"),
)

# Hugging Face downloads use the ordinary HTTP path with an explicit timeout.
# The same values are forced inside the function and recorded in runtime
# metadata, rather than relying only on image-layer defaults.
PIPELINE_HF_ENV: Final = (
    ("HF_HUB_DISABLE_XET", "1"),
    ("HF_HUB_DOWNLOAD_TIMEOUT", "600"),
)
EVAL_PACKAGES = ("torch", "transformers", "datasets")
SERVING_PACKAGES = ("torch", "transformers", "peft", "fastapi", "uvicorn")
VLLM_PACKAGES = (
    "vllm==0.24.0",
    "huggingface_hub==1.22.0",
    "torch==2.11.0",
    "transformers==5.13.1",
    "tokenizers==0.22.2",
)
TRANSLATION_PACKAGES = (
    "datasets==5.0.0",
    "accelerate==1.14.0",
    "sentencepiece==0.2.2",
)

# MADLAD's weights load correctly through the probe-established Transformers
# v4 stack.  Keep this dependency graph completely separate from vLLM's
# Transformers v5 graph: installing both into one image lets pip choose a
# nominally satisfiable but scientifically unproven environment.
SEQ2SEQ_TRANSLATION_PACKAGES = (
    "torch==2.11.0",
    "transformers==4.57.6",
    "tokenizers==0.22.2",
    "accelerate==1.14.0",
    "huggingface_hub==0.36.2",
    "sentencepiece==0.2.2",
    "datasets==5.0.0",
    "safetensors==0.8.0",
)

# Exact identity observed in the pinned translation image. Modal only accepts a
# major.minor Python selector, so the patch version is enforced inside the
# remote function before a full Hebrew producer may export data or load a
# model. Smoke and other-language runs retain their diagnostic role and record
# the environment without enforcing this identity.
VLLM_TRANSLATION_RUNTIME_VERSIONS: Final = (
    ("python", "3.13.0"),
    ("vllm", "0.24.0"),
    ("huggingface_hub", "1.22.0"),
    ("torch", "2.11.0"),
    ("transformers", "5.13.1"),
    ("tokenizers", "0.22.2"),
    ("datasets", "5.0.0"),
    ("accelerate", "1.14.0"),
    ("sentencepiece", "0.2.2"),
)

SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS: Final = (
    ("python", "3.13.0"),
    ("torch", "2.11.0"),
    ("transformers", "4.57.6"),
    ("tokenizers", "0.22.2"),
    ("accelerate", "1.14.0"),
    ("huggingface_hub", "0.36.2"),
    ("sentencepiece", "0.2.2"),
    ("datasets", "5.0.0"),
    ("safetensors", "0.8.0"),
)

# Provider-backed translation is deliberately isolated from both GPU model
# stacks.  Modal's Debian image accepts the full patch version, and the OpenAI
# SDK is pinned to the exact version enforced again by the adapter factory at
# runtime. ``datasets`` remains because this function performs the same source
# export and deterministic cohort selection before provider requests begin.
OPENAI_TRANSLATION_PYTHON_VERSION: Final = "3.13.3"
OPENAI_TRANSLATION_PACKAGES: Final = (
    "openai==2.45.0",
    "datasets==5.0.0",
)
OPENAI_TRANSLATION_RUNTIME_VERSIONS: Final = (
    ("python", OPENAI_TRANSLATION_PYTHON_VERSION),
    ("openai", "2.45.0"),
    ("datasets", "5.0.0"),
)

# Compatibility name for existing vLLM consumers. New code should select an
# explicit backend identity instead of treating both incompatible stacks as
# one translation runtime.
TRANSLATION_RUNTIME_VERSIONS: Final = VLLM_TRANSLATION_RUNTIME_VERSIONS
# The live Marian probe resolved Modal's floating ``3.13`` Debian image to
# 3.13.3. Unlike ``from_registry(..., add_python=...)``, ``debian_slim`` accepts
# a full micro version, so this producer can and does pin the observed patch.
SEMANTIC_REVIEW_PYTHON_VERSION: Final = "3.13.3"
SEMANTIC_REVIEW_PACKAGES = (
    "torch==2.11.0",
    "transformers==5.13.1",
    "tokenizers==0.22.2",
    "accelerate==1.14.0",
    "huggingface_hub==1.22.0",
    "sentencepiece==0.2.2",
    "sacremoses==0.1.1",
)
SEMANTIC_REVIEW_HF_ENV: Final = (
    ("HF_HUB_DISABLE_XET", "1"),
    ("HF_HUB_DOWNLOAD_TIMEOUT", "600"),
)

NVIDIA_INDEX_URL = "https://pypi.nvidia.com"

# vLLM JIT-compiles kernels at startup and needs the full CUDA toolkit.
CUDA_DEVEL_BASE = (
    "nvidia/cuda:12.8.1-devel-ubuntu24.04"
    "@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
)

PipelineStage = Literal["data", "train", "eval"]


class RemoteStageOptions(TypedDict):
    gpu: str
    timeout: int


def _python_base(python_version: str = PIPELINE_PYTHON_VERSION) -> modal.Image:
    import modal

    return modal.Image.debian_slim(python_version=python_version).pip_install(*BASE_PACKAGES)


def _with_source(image: modal.Image) -> modal.Image:
    # add_local_python_source must be the final layer: Modal rejects build
    # steps after a non-copy local-file layer at build time.
    return image.add_local_python_source("sommelier")


def data_image() -> modal.Image:
    """GPU dataframe stack for coarse filtering plus the package source.

    cudf wheels come from the NVIDIA index; the image is only used by the
    remote data stage, never imported locally.
    """
    return _with_source(
        _python_base().pip_install(*DATA_PACKAGES, extra_index_url=NVIDIA_INDEX_URL)
    )


def train_image() -> modal.Image:
    """Model loading, quantization, and adapter training stack."""
    return _with_source(
        _python_base()
        .apt_install("openssh-client")
        .pip_install(*TRAIN_PACKAGES)
        .env(dict(PIPELINE_HF_ENV))
    )


def eval_image() -> modal.Image:
    """Deterministic generation, parser, and metrics stack.

    The v1 evaluation runner generates through transformers; vllm is not
    included until the runner actually uses it, keeping the image truthful
    about its capabilities.
    """
    return _with_source(_python_base().pip_install(*EVAL_PACKAGES))


def serving_image() -> modal.Image:
    """Optional OpenAI-compatible adapter serving stack."""
    return _with_source(_python_base().pip_install(*SERVING_PACKAGES))


def vllm_serving_image() -> modal.Image:
    """vLLM inference server stack for high-throughput adapter serving.

    Built from the CUDA devel base image because vLLM's startup warm-up
    JIT-compiles kernels with nvcc, which slim images lack. The container
    runs vLLM's own OpenAI-compatible entrypoint and never imports
    sommelier, so the package source is deliberately not mounted.
    """
    import modal

    return modal.Image.from_registry(
        CUDA_DEVEL_BASE,
        add_python=PYTHON_VERSION,
    ).pip_install(*VLLM_PACKAGES)


def vllm_translation_image() -> modal.Image:
    """Pinned vLLM chat-translation stack with package source mounted.

    The datasets package rides along because the remote producer also exports
    and selects the root cohort before translating it.
    """
    import modal

    return _with_source(
        modal.Image.from_registry(
            CUDA_DEVEL_BASE,
            add_python=PYTHON_VERSION,
        ).pip_install(*VLLM_PACKAGES, *BASE_PACKAGES, *TRANSLATION_PACKAGES)
    )


def seq2seq_translation_image() -> modal.Image:
    """Probe-established Transformers-v4 MADLAD translation stack.

    It intentionally shares the pinned CUDA base with the live probe while
    excluding vLLM and its incompatible Transformers-v5 dependency graph.
    """
    import modal

    return _with_source(
        modal.Image.from_registry(
            CUDA_DEVEL_BASE,
            add_python=PYTHON_VERSION,
        ).pip_install(*BASE_PACKAGES, *SEQ2SEQ_TRANSLATION_PACKAGES)
    )


def openai_translation_image() -> modal.Image:
    """CPU-only Responses API producer with no local model dependencies."""
    return _with_source(
        _python_base(OPENAI_TRANSLATION_PYTHON_VERSION).pip_install(*OPENAI_TRANSLATION_PACKAGES)
    )


def translation_image() -> modal.Image:
    """Compatibility alias for the historical vLLM translation image."""
    return vllm_translation_image()


def semantic_review_image() -> modal.Image:
    """Pinned Marian runtime for the independent semantic-review gate."""
    return _with_source(
        _python_base(SEMANTIC_REVIEW_PYTHON_VERSION)
        .pip_install(*SEMANTIC_REVIEW_PACKAGES)
        .env(dict(SEMANTIC_REVIEW_HF_ENV))
    )


def stage_options(config: SommelierConfig, stage: PipelineStage) -> RemoteStageOptions:
    """Return legacy-shaped planning options for one nominal pipeline stage.

    The timeout-named value is a planning estimate retained for config/API
    compatibility. The current pipeline runs as one remote function and does
    not attach these values to per-stage watchdogs. Unknown stages still fail
    explicitly instead of inheriting another stage's estimate.
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
