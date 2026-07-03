"""Serve the trained tool-calling adapter with vLLM on Modal.

Deploy the OpenAI-compatible server (scales to zero when idle):

    uv run modal deploy remote_serving.py

Smoke-test the deployed endpoint through sommelier's own prompt policy
and conservative parser:

    uv run modal run remote_serving.py

The server loads the base model with vLLM and registers the LoRA adapter
as a selectable model, so one deployment serves both variants:

    "model": "sommelier-tool-caller"   → base + trained adapter
    "model": "<base model id>"         → base model (A/B comparison)

Adapter source (first match wins):

- SOMMELIER_ADAPTER_VOLUME_PATH — a path on the sommelier-artifacts
  volume, e.g. artifacts/runs/nemotron-8b-full-3/train/adapter, to serve
  an adapter straight from a pipeline run;
- SOMMELIER_ADAPTER_REPO / SOMMELIER_ADAPTER_REVISION — a Hugging Face
  repo (default: the published adapter).

Launch-time knobs (read when the app is deployed): SOMMELIER_GPU
(default L40S), SOMMELIER_MAX_MODEL_LEN (default 4096, the training
budget). If SOMMELIER_SERVE_API_KEY is present in .env it is required
as a Bearer token; otherwise the endpoint is open — it stays an
optional, illustrative service either way.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

import modal

# The image builder lives in the sommelier package, which is intentionally
# not mounted into the lean vLLM image; this module is also imported inside
# the container, so the import must stay local-only.
if modal.is_local():
    from sommelier.remote.images import vllm_serving_image

    _image = vllm_serving_image()
else:
    _image = modal.Image.debian_slim()

APP_NAME = "sommelier-vllm"

BASE_MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
BASE_MODEL_REVISION = "main"
DEFAULT_ADAPTER_REPO = "abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora"
ADAPTER_MODEL_NAME = "sommelier-tool-caller"

VLLM_PORT = 8000
MINUTES = 60

GPU = os.environ.get("SOMMELIER_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("SOMMELIER_MAX_MODEL_LEN", "4096"))

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("sommelier-vllm-cache", create_if_missing=True)

serving_image = _image.env({"HF_HOME": "/hf-cache", "VLLM_CACHE_ROOT": "/vllm-cache"})


def _resolve_adapter_path() -> str:
    """Resolves the adapter directory inside the container."""
    volume_path = os.environ.get("SOMMELIER_ADAPTER_VOLUME_PATH", "")
    if volume_path:
        adapter_dir = Path("/artifacts") / volume_path
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"no adapter_config.json under {adapter_dir}; check "
                "SOMMELIER_ADAPTER_VOLUME_PATH against the artifacts volume"
            )
        return str(adapter_dir)

    from huggingface_hub import snapshot_download

    repo = os.environ.get("SOMMELIER_ADAPTER_REPO", DEFAULT_ADAPTER_REPO)
    revision = os.environ.get("SOMMELIER_ADAPTER_REVISION", "main")
    return str(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer*",
                "chat_template*",
            ],
        )
    )


@app.function(
    image=serving_image,
    gpu=GPU,
    timeout=60 * MINUTES,
    scaledown_window=5 * MINUTES,
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    volumes={
        "/artifacts": artifacts_volume,
        "/hf-cache": hf_cache_volume,
        "/vllm-cache": vllm_cache_volume,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    adapter_path = _resolve_adapter_path()
    print(f"[serve] adapter resolved to {adapter_path}", flush=True)

    command = _vllm_command(adapter_path)
    api_key = os.environ.get("SOMMELIER_SERVE_API_KEY", "")
    if api_key:
        command.extend(["--api-key", api_key])
        print("[serve] endpoint protected by SOMMELIER_SERVE_API_KEY", flush=True)
    else:
        print(
            "[serve] WARNING: no SOMMELIER_SERVE_API_KEY set; the endpoint "
            "is open to anyone with the URL (illustrative service only)",
            flush=True,
        )

    subprocess.Popen(command)


def _vllm_command(adapter_path: str) -> list[str]:
    command = [
        "vllm",
        "serve",
        BASE_MODEL_ID,
        "--revision",
        BASE_MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"{ADAPTER_MODEL_NAME}={adapter_path}",
    ]
    print(f"[serve] command: {' '.join(command)}", flush=True)
    return command


@app.function(
    image=serving_image,
    gpu=GPU,
    timeout=25 * MINUTES,
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    volumes={
        "/artifacts": artifacts_volume,
        "/hf-cache": hf_cache_volume,
        "/vllm-cache": vllm_cache_volume,
    },
)
def diagnose(wait_seconds: int = 900) -> None:
    """Runs the exact serve command in the foreground with full logs.

    Use when the web server fails to come up: streams every engine line to
    the client instead of relying on Modal's retained container logs.

        uv run modal run remote_serving.py::diagnose
    """
    import socket
    import time

    adapter_path = _resolve_adapter_path()
    print(f"[diagnose] adapter resolved to {adapter_path}", flush=True)
    process = subprocess.Popen(_vllm_command(adapter_path))

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"[diagnose] vllm exited with code {process.returncode}", flush=True)
            return
        try:
            with socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=2):
                print("[diagnose] port open — server came up healthy", flush=True)
                process.terminate()
                return
        except OSError:
            time.sleep(5)
    print("[diagnose] timed out waiting for port or exit", flush=True)
    process.terminate()


def _request_json(
    url: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    api_key = os.environ.get("SOMMELIER_SERVE_API_KEY", "")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=300) as response:
        body: dict[str, object] = json.loads(response.read().decode("utf-8"))
    return body


def _wait_until_ready(url: str, *, attempts: int = 60, delay_seconds: float = 15.0) -> None:
    """Polls /v1/models until vLLM answers; cold starts take minutes.

    While the container boots, Modal's proxy answers with redirect loops or
    connection resets — both count as "not ready yet", not failures.
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            models = _request_json(f"{url}/v1/models")
            names = [entry.get("id") for entry in models.get("data", [])]  # type: ignore[union-attr]
            print(f"[smoke] server ready; models: {names}")
            return
        except Exception as error:  # noqa: BLE001 - readiness probe
            print(f"[smoke] not ready (attempt {attempt}/{attempts}): {error}", flush=True)
            time.sleep(delay_seconds)
    raise RuntimeError(f"server at {url} did not become ready")


@app.local_entrypoint()
def smoke(url: str = "", model: str = ADAPTER_MODEL_NAME) -> None:
    """Sends one canonical tool-calling request and parses the reply.

    Builds the exact training/eval prompt policy through the sommelier
    package and classifies the completion with the conservative parser,
    so a passing smoke means the deployed endpoint reproduces the
    evaluated behavior.
    """
    from sommelier.evaluation.parse import parse_tool_call
    from sommelier.formatting.chat import build_prompt_messages

    if not url:
        url = serve.get_web_url() or ""
    if not url:
        raise RuntimeError("no server URL; deploy first or pass --url")
    url = url.rstrip("/")
    _wait_until_ready(url)

    tools = [
        {
            "name": "lookup_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {"city": {"description": "Name of the city.", "type": "str"}},
        }
    ]
    messages = build_prompt_messages(
        query="What is the weather in Paris today?",
        tools=list(tools),
        system_prompt=(
            "You are a tool-calling model. Select the correct tool and return "
            "only the JSON tool call. Do not include explanations."
        ),
    )

    completion = _request_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": 0.0,
            "max_tokens": 128,
        },
    )
    choices = completion.get("choices")
    assert isinstance(choices, list) and choices, completion
    raw_text = choices[0]["message"]["content"]
    parsed_call, parse_status = parse_tool_call(raw_text)

    print(json.dumps({
        "url": url,
        "model": model,
        "raw_text": raw_text,
        "parse_status": parse_status,
        "parsed_call": parsed_call,
    }, indent=2))
    assert parse_status == "ok", f"expected ok, got {parse_status}: {raw_text!r}"
    assert parsed_call is not None and parsed_call["name"] == "lookup_weather"
    print("smoke ok")
