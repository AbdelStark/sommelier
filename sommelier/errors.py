from __future__ import annotations


class SommelierError(Exception):
    code: str = "SOM000"
    exit_code: int = 5
    hint: str | None = None

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint


class UserInputError(SommelierError):
    code = "SOM002"
    exit_code = 2


class ConfigError(UserInputError):
    code = "SOM201"


class SchemaValidationError(UserInputError):
    code = "SOM202"


class ArtifactNotFoundError(UserInputError):
    code = "SOM203"


class ExternalDependencyError(SommelierError):
    code = "SOM003"
    exit_code = 3


class RemoteExecutionError(SommelierError):
    code = "SOM004"
    exit_code = 4


class ResourceError(RemoteExecutionError):
    code = "SOM401"


class EvaluationError(SommelierError):
    code = "SOM005"
    exit_code = 4


class InvariantViolation(SommelierError):
    code = "SOM005"
    exit_code = 5


class SecurityPolicyError(SommelierError):
    code = "SOM006"
    exit_code = 5


def format_cli_error(error: SommelierError) -> str:
    lines = [f"sommelier: {error.code}: {error}"]
    if error.hint:
        lines.append(f"hint: {error.hint}")
    return "\n".join(lines)
