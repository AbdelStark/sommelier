import subprocess
import sys

from sommelier.cli import main
from sommelier.errors import (
    ConfigError,
    ExternalDependencyError,
    InvariantViolation,
    SecurityPolicyError,
    UserInputError,
    format_cli_error,
)


def test_exit_codes() -> None:
    assert UserInputError("bad input").exit_code == 2
    assert ExternalDependencyError("missing dependency").exit_code == 3
    assert InvariantViolation("broken invariant").exit_code == 5
    assert SecurityPolicyError("secret leak").exit_code == 5


def test_format_cli_error_includes_hint() -> None:
    message = format_cli_error(
        ConfigError("invalid config", hint="Fix the YAML and retry.")
    )
    assert "SOM201" in message
    assert "hint: Fix the YAML and retry." in message


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sommelier.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "sommelier" in result.stdout


def test_cli_config_validate_smoke_example() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sommelier.cli",
            "config",
            "validate",
            "--config",
            "examples/config.smoke.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "config ok" in result.stdout


def test_cli_missing_config_returns_exit_code_2() -> None:
    assert main(["config", "validate", "--config", "missing.yaml"]) == 2


def test_cli_invalid_config_returns_exit_code_2() -> None:
    assert main(["config", "validate", "--config", "examples/config.invalid.yaml"]) == 2