from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal

APP_NAME = "sommelier"


def smoke_square(x: int) -> int:
    """Shared smoke payload executed inside the remote container.

    Remote wrappers must call importable package functions instead of
    defining their own logic; this is the connectivity smoke
    payload the Modal wrapper delegates to.
    """
    return x * x


def build_app() -> modal.App:
    """Builds the Modal app wrapping shared package functions.

    modal is imported lazily so ``import sommelier`` (and every submodule)
    stays free of remote-execution dependencies. The smoke image mounts the
    sommelier package source; stage-specific dependency images are defined
    separately (one image per dependency stack).
    """
    import modal

    smoke_image = modal.Image.debian_slim(python_version="3.13").add_local_python_source(
        "sommelier"
    )
    app = modal.App(name=APP_NAME)

    # serialized=True lets the wrapper live inside this factory, keeping the
    # module importable without modal; the image still carries the package
    # source so the wrapper resolves shared functions remotely.
    @app.function(image=smoke_image, serialized=True, name="square")
    def square(x: int) -> int:
        from sommelier.remote.app import smoke_square

        return smoke_square(x)

    return app


def registered_function_names(app: Any) -> list[str]:
    """Names of remote functions registered on a built app (test surface)."""
    return list(app.registered_functions.keys())
