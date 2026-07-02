from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from sommelier.config import load_config
from sommelier.errors import (
    ExternalDependencyError,
    SchemaValidationError,
    UserInputError,
)
from sommelier.evaluation.generate import DecodingConfig
from sommelier.formatting.templates import render_formatted_example
from sommelier.logs import read_log_events
from sommelier.serving.openai_compat import AdapterService, build_adapter_service
from sommelier.serving.schemas import build_serve_response

SMOKE_CONFIG = Path("examples/config.smoke.yaml")

TOOLS = [
    {
        "name": "lookup_weather",
        "description": "Look up weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]

QUERY = "What is the weather in Paris today?"


class StubRenderer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        rendered = "".join(
            f"<|{message['role']}|>{message['content']}<|end|>" for message in conversation
        )
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


class StubGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []
        self.decodings: list[DecodingConfig] = []

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        self.prompts.append(prompt_text)
        self.decodings.append(decoding)
        return self.text


def make_service(tmp_path: Path, text: str) -> tuple[AdapterService, StubGenerator]:
    config = load_config(SMOKE_CONFIG)
    adapter_dir = tmp_path / "train" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    generator = StubGenerator(text)
    service = build_adapter_service(
        config,
        adapter_dir,
        generator=generator,
        renderer=StubRenderer(),
        log_dir=tmp_path / "logs",
    )
    return service, generator


def payload(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "client system prompt to be ignored"},
            {"role": "user", "content": QUERY},
        ],
        "tools": [dict(tool) for tool in TOOLS],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    base.update(changes)
    return base


def test_smoke_request_returns_parse_status(tmp_path: Path) -> None:
    service, generator = make_service(
        tmp_path, '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'
    )
    response = service.handle(payload())

    assert response["parse_status"] == "ok"
    assert response["parsed_call"] is not None
    assert response["parsed_call"]["name"] == "lookup_weather"
    assert response["model_kind"] == "adapter"
    assert generator.decodings[0] == {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 128,
    }


def test_prompt_parity_with_evaluation_policy(tmp_path: Path) -> None:
    service, generator = make_service(tmp_path, "irrelevant")
    service.handle(payload())

    config = load_config(SMOKE_CONFIG)
    formatted = render_formatted_example(
        {
            "example_id": "parity-1",
            "split": "test",
            "query": QUERY,
            "tools": [dict(tool) for tool in TOOLS],
            "gold_calls": [{"name": "lookup_weather", "arguments": {"city": "Paris"}}],
        },
        tokenizer=StubRenderer(),
        tokenizer_id="stub",
        tokenizer_revision="stub",
        system_prompt=config.formatting.system_prompt,
        template_policy=config.formatting.template_policy,
    )
    assert generator.prompts[0] == formatted["prompt_text"]


def test_client_system_message_is_ignored(tmp_path: Path) -> None:
    service, generator = make_service(tmp_path, "x")
    service.handle(payload())
    prompt = generator.prompts[0]
    assert "client system prompt to be ignored" not in prompt
    config = load_config(SMOKE_CONFIG)
    assert config.formatting.system_prompt.strip().splitlines()[0] in prompt


def test_parse_status_is_logged(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, "no json here")
    service.handle(payload())

    events = read_log_events(tmp_path / "logs" / "serve.jsonl")
    assert len(events) == 1
    assert events[0]["stage"] == "serve"
    assert events[0]["fields"]["parse_status"] == "no_json"


def test_request_without_user_message_fails(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, "x")
    with pytest.raises(UserInputError):
        service.handle(
            payload(messages=[{"role": "system", "content": "only system"}])
        )


def test_invalid_payload_fails_closed(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, "x")
    with pytest.raises(SchemaValidationError):
        service.handle(payload(temperature=0.9))


def test_missing_adapter_dir_fails(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG)
    with pytest.raises(UserInputError):
        build_adapter_service(
            config,
            tmp_path / "missing",
            generator=StubGenerator("x"),
            renderer=StubRenderer(),
        )


def test_response_builder_reuses_parser() -> None:
    response = build_serve_response("nothing structured")
    assert response["parse_status"] == "no_json"


@pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is not None,
    reason="fastapi installed; missing-dependency path not reachable",
)
def test_http_app_requires_fastapi(tmp_path: Path) -> None:
    from sommelier.serving.openai_compat import build_http_app

    service, _ = make_service(tmp_path, "x")
    with pytest.raises(ExternalDependencyError):
        build_http_app(service)


@pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="fastapi not installed; startup smoke runs where the stack exists",
)
def test_http_app_startup_smoke(tmp_path: Path) -> None:
    from sommelier.serving.openai_compat import build_http_app

    service, _ = make_service(
        tmp_path, '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'
    )
    app = build_http_app(service)
    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/v1/chat/completions" in routes
    assert "/health" in routes


def test_serving_module_import_stays_light() -> None:
    import subprocess
    import sys

    code = (
        "import json, sys\n"
        "import sommelier.serving.openai_compat\n"
        "print(json.dumps([m for m in ('fastapi', 'uvicorn', 'torch') if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
