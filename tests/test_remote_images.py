from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sommelier.config import load_config
from sommelier.errors import UserInputError
from sommelier.remote.images import stage_options

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


def test_image_construction_smoke() -> None:
    import modal

    from sommelier.remote.images import (
        data_image,
        eval_image,
        serving_image,
        train_image,
    )

    images = {
        "data": data_image(),
        "train": train_image(),
        "eval": eval_image(),
        "serving": serving_image(),
    }
    for name, image in images.items():
        assert isinstance(image, modal.Image), name
    assert len({id(image) for image in images.values()}) == 4


def test_stage_options_map_config_timeouts_and_gpu() -> None:
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
