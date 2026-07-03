from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sommelier.config import SommelierConfig
from sommelier.errors import ExternalDependencyError, UserInputError
from sommelier.evaluation.generate import DecodingConfig, TextGenerator
from sommelier.formatting.chat import build_prompt_messages
from sommelier.formatting.templates import ChatTemplateRenderer
from sommelier.logs import StageLogger
from sommelier.serving.schemas import (
    ServeRequest,
    ServeResponse,
    build_serve_response,
    validate_serve_request,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


class AdapterService:
    """Framework-free core of the optional adapter service.

    Serves one adapter with the exact evaluation prompt policy: the system
    message is rebuilt from the configured instruction and the request's
    canonical tool schemas; client-supplied system messages are ignored so
    private prompt variants cannot drift serving away from evaluation.
    """

    def __init__(
        self,
        config: SommelierConfig,
        *,
        generator: TextGenerator,
        renderer: ChatTemplateRenderer,
        logger: StageLogger,
    ) -> None:
        self.config = config
        self.generator = generator
        self.renderer = renderer
        self.logger = logger

    def _last_user_query(self, request: ServeRequest) -> str:
        for message in reversed(request["messages"]):
            if message["role"] == "user":
                return message["content"]
        raise UserInputError(
            "serve request contains no user message",
            hint="Send at least one message with role user.",
        )

    def prompt_text(self, request: ServeRequest) -> str:
        messages = build_prompt_messages(
            query=self._last_user_query(request),
            tools=list(request["tools"]),
            system_prompt=self.config.formatting.system_prompt,
        )
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        return self.renderer.apply_chat_template(
            plain,
            tokenize=False,
            add_generation_prompt=True,
        )

    def handle(self, payload: object) -> ServeResponse:
        request = validate_serve_request(payload)
        decoding = DecodingConfig(
            temperature=0.0,
            do_sample=False,
            max_new_tokens=request["max_tokens"],
        )
        raw_text = self.generator.generate(self.prompt_text(request), decoding=decoding)
        response = build_serve_response(raw_text)
        self.logger.info(
            "request_served",
            "served one adapter completion",
            parse_status=response["parse_status"],
            max_tokens=request["max_tokens"],
        )
        return response


def build_adapter_service(
    config: SommelierConfig,
    adapter_dir: Path,
    *,
    generator: TextGenerator | None = None,
    renderer: ChatTemplateRenderer | None = None,
    log_dir: Path | None = None,
) -> AdapterService:
    """Builds the service core, loading the adapter model when not injected."""
    if not adapter_dir.exists():
        raise UserInputError(
            f"adapter directory not found: {adapter_dir}",
            hint="Train an adapter first: sommelier train run ...",
        )

    if generator is None:
        from sommelier.evaluation.generate import load_model_generator

        generator = load_model_generator(config, "adapter", adapter_dir)
    if renderer is None:
        from sommelier.formatting.templates import load_tokenizer

        renderer = load_tokenizer(config)

    logger = StageLogger(
        run_id=adapter_dir.parent.parent.name or "serve",
        stage="serve",
        log_dir=(log_dir if log_dir is not None else adapter_dir.parent / "logs"),
    )
    return AdapterService(config, generator=generator, renderer=renderer, logger=logger)


def build_http_app(service: AdapterService) -> FastAPI:
    """Wraps the service core in an OpenAI-compatible FastAPI app.

    fastapi is part of the optional serving stack and imported lazily.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as error:
        raise ExternalDependencyError(
            "the adapter service requires the fastapi package",
            hint="Run serving remotely or install the serving extra stack.",
        ) from error

    from sommelier.errors import SommelierError

    app = FastAPI(
        title="sommelier adapter service",
        description="Optional, illustrative single-adapter endpoint. "
        "Not a production serving system.",
    )

    # The payload annotation must resolve from module globals (postponed
    # annotations + get_type_hints); locally imported types like Request
    # would silently degrade to a query parameter.
    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> Any:
        try:
            return service.handle(payload)
        except SommelierError as error:
            return JSONResponse(
                status_code=422 if error.exit_code == 2 else 500,
                content={"error": {"code": error.code, "message": str(error)}},
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_kind": "adapter"}

    return app


def serve_adapter(
    config: SommelierConfig,
    adapter_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Starts the optional adapter service with uvicorn (blocking)."""
    service = build_adapter_service(config, adapter_dir)
    app = build_http_app(service)
    try:
        import uvicorn
    except ImportError as error:
        raise ExternalDependencyError(
            "the adapter service requires the uvicorn package",
            hint="Run serving remotely or install the serving extra stack.",
        ) from error
    uvicorn.run(app, host=host, port=port)
