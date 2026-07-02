from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sommelier.remote.app import smoke_square

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_square_is_shared_package_logic() -> None:
    assert smoke_square(7) == 49


def test_remote_module_import_stays_free_of_modal() -> None:
    code = (
        "import json, sys\n"
        "import sommelier.remote.app\n"
        "print(json.dumps('modal' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) is False


def test_build_app_registers_smoke_wrapper() -> None:
    from sommelier.remote.app import build_app, registered_function_names

    app = build_app()
    assert "square" in registered_function_names(app)


def test_compatibility_entrypoint_still_runs() -> None:
    result = subprocess.run(
        [sys.executable, "sommelier_entrypoint.py"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
