from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN_MODULES = (
    "modal",
    "cudf",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "vllm",
    "wandb",
)

IMPORT_ALL_SNIPPET = """
import importlib
import json
import pkgutil
import sys

import sommelier

modules = ["sommelier"]
for info in pkgutil.walk_packages(sommelier.__path__, prefix="sommelier."):
    modules.append(info.name)
for name in modules:
    importlib.import_module(name)

print(json.dumps({"modules": modules, "loaded": sorted(sys.modules)}))
"""


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_sommelier_succeeds_in_clean_interpreter() -> None:
    result = _run_python("import sommelier")
    assert result.returncode == 0, result.stderr


def test_package_modules_never_import_heavy_dependencies() -> None:
    result = _run_python(IMPORT_ALL_SNIPPET)
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    imported = payload["modules"]
    assert "sommelier.cli" in imported
    assert "sommelier.data.gpu" in imported

    loaded = set(payload["loaded"])
    for forbidden in FORBIDDEN_MODULES:
        assert forbidden not in loaded, f"{forbidden} imported at package import time"
        assert not any(name.startswith(f"{forbidden}.") for name in loaded), (
            f"{forbidden} submodule imported at package import time"
        )
