"""Thin compatibility wrapper for the Modal smoke app.

The app definition lives in sommelier.remote.app; this file materializes
it and hosts the local entrypoint (Modal requires entrypoints in global
scope) so existing workflows keep working:

    uv run python sommelier_entrypoint.py
    uv run modal run sommelier_entrypoint.py
"""

from sommelier.remote.app import build_app

app = build_app()


@app.local_entrypoint()
def main() -> None:
    square = app.registered_functions["square"]
    print(f"Squaring 42 is: {square.remote(42)}")
