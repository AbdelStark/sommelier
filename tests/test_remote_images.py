from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sommelier.config import load_config
from sommelier.errors import UserInputError
from sommelier.remote.images import (
    OPENAI_TRANSLATION_PACKAGES,
    OPENAI_TRANSLATION_PYTHON_VERSION,
    OPENAI_TRANSLATION_RUNTIME_VERSIONS,
    PIPELINE_HF_ENV,
    PIPELINE_RUNTIME_VERSIONS,
    SEQ2SEQ_TRANSLATION_PACKAGES,
    SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS,
    TRAIN_PACKAGES,
    TRANSLATION_PACKAGES,
    TRANSLATION_RUNTIME_VERSIONS,
    VLLM_PACKAGES,
    VLLM_TRANSLATION_RUNTIME_VERSIONS,
    stage_options,
)

SMOKE_CONFIG = Path("examples/config.smoke.yaml")


def test_images_module_import_stays_free_of_modal() -> None:
    code = (
        "import json, sys\n"
        "import sommelier.remote.images\n"
        "print(json.dumps('modal' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) is False


def test_vllm_stack_pins_the_observed_dicta_decoder_runtime() -> None:
    assert VLLM_PACKAGES == (
        "vllm==0.24.0",
        "huggingface_hub==1.22.0",
        "torch==2.11.0",
        "transformers==5.13.1",
        "tokenizers==0.22.2",
    )


def test_translation_stacks_are_pinned_and_mutually_exclusive() -> None:
    assert TRANSLATION_PACKAGES == (
        "datasets==5.0.0",
        "accelerate==1.14.0",
        "sentencepiece==0.2.2",
    )
    assert dict(VLLM_TRANSLATION_RUNTIME_VERSIONS) == {
        "python": "3.13.0",
        "vllm": "0.24.0",
        "huggingface_hub": "1.22.0",
        "torch": "2.11.0",
        "transformers": "5.13.1",
        "tokenizers": "0.22.2",
        "datasets": "5.0.0",
        "accelerate": "1.14.0",
        "sentencepiece": "0.2.2",
    }
    assert TRANSLATION_RUNTIME_VERSIONS == VLLM_TRANSLATION_RUNTIME_VERSIONS
    assert SEQ2SEQ_TRANSLATION_PACKAGES == (
        "torch==2.11.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
        "accelerate==1.14.0",
        "huggingface_hub==0.36.2",
        "sentencepiece==0.2.2",
        "datasets==5.0.0",
        "safetensors==0.8.0",
    )
    assert dict(SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS) == {
        "python": "3.13.0",
        "torch": "2.11.0",
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "accelerate": "1.14.0",
        "huggingface_hub": "0.36.2",
        "sentencepiece": "0.2.2",
        "datasets": "5.0.0",
        "safetensors": "0.8.0",
    }
    assert "vllm" not in dict(SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS)

    assert OPENAI_TRANSLATION_PYTHON_VERSION == "3.13.3"
    assert OPENAI_TRANSLATION_PACKAGES == (
        "openai==2.45.0",
        "datasets==5.0.0",
    )
    assert dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS) == {
        "python": "3.13.3",
        "openai": "2.45.0",
        "datasets": "5.0.0",
    }
    assert "vllm" not in dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)
    assert "torch" not in dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)


def test_pipeline_stack_pins_the_observed_probe_runtime() -> None:
    assert TRAIN_PACKAGES == (
        "torch==2.13.0",
        "transformers==5.13.1",
        "tokenizers==0.22.2",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "bitsandbytes==0.49.2",
        "datasets==5.0.0",
        "huggingface_hub==1.23.0",
    )
    assert dict(PIPELINE_RUNTIME_VERSIONS) == {
        "python": "3.13.3",
        "torch": "2.13.0",
        "transformers": "5.13.1",
        "tokenizers": "0.22.2",
        "accelerate": "1.14.0",
        "peft": "0.19.1",
        "bitsandbytes": "0.49.2",
        "datasets": "5.0.0",
        "huggingface_hub": "1.23.0",
    }
    assert dict(PIPELINE_HF_ENV) == {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
    }


def test_image_construction_smoke() -> None:
    import modal

    from sommelier.remote.images import (
        data_image,
        eval_image,
        openai_translation_image,
        seq2seq_translation_image,
        serving_image,
        train_image,
        vllm_serving_image,
        vllm_translation_image,
    )

    images = {
        "data": data_image(),
        "train": train_image(),
        "eval": eval_image(),
        "serving": serving_image(),
        "vllm": vllm_serving_image(),
        "vllm-translation": vllm_translation_image(),
        "seq2seq-translation": seq2seq_translation_image(),
        "openai-translation": openai_translation_image(),
    }
    for name, image in images.items():
        assert isinstance(image, modal.Image), name
    assert len({id(image) for image in images.values()}) == 8


def test_stage_options_map_config_planning_estimates_and_gpu() -> None:
    config = load_config(SMOKE_CONFIG)

    assert stage_options(config, "data") == {
        "gpu": config.remote.gpu,
        "timeout": config.remote.data_timeout_seconds,
    }
    assert stage_options(config, "train")["timeout"] == config.remote.train_timeout_seconds
    assert stage_options(config, "eval")["timeout"] == config.remote.eval_timeout_seconds


def test_stage_options_reject_unknown_stage() -> None:
    config = load_config(SMOKE_CONFIG)
    with pytest.raises(UserInputError):
        stage_options(config, "serve")  # type: ignore[arg-type]
