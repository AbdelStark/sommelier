# Serving the adapter

Sommelier ships two ways to put the trained adapter behind an HTTP endpoint: a local single-adapter service built into the CLI, and a vLLM deployment on Modal. Both exist so you can inspect the adapter's behavior by hand, and neither is a production system: no autoscaling guarantees and no multi-tenant isolation, and the local service does not stream. The [evaluation claim](../results/reference-run.md) never depends on serving; if both endpoints disappeared, the comparison report would be untouched.

## Local: `sommelier serve adapter`

```bash
uv run sommelier serve adapter --config examples/config.full.yaml \
  --adapter <path-to-adapter-dir>
```

`--config` and `--adapter` are required; `--host` defaults to `127.0.0.1` and `--port` to `8000`. The adapter directory is a pipeline run's `train/adapter/` or a local download of the [published adapter](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora). The base install cannot serve: the service needs the model stack plus `fastapi` and `uvicorn`, and without them it fails with an `ExternalDependencyError` (exit 3) instead of degrading. See [errors](../reference/errors.md).

The service is deliberately rigid, because its job is to reproduce evaluated behavior, not to be a flexible API:

- **The prompt is rebuilt server-side.** The system message is constructed from the configured `formatting.system_prompt` plus the canonical JSON of the request's tool schemas, through the same `build_prompt_messages` function that formatting and evaluation use. Client-supplied system messages are ignored, so a private prompt variant cannot drift serving away from what was evaluated. The last `user` message becomes the query; a request with no user message is rejected.
- **Requests must have exactly four fields**: `messages`, `tools`, `temperature`, `max_tokens`. Unknown fields are rejected, so silent client drift cannot change behavior.
- **`temperature` must be exactly `0.0`** and `max_tokens` a positive integer. Serving decodes deterministically, like [evaluation](../concepts/evaluation.md).

The request shape matches the README example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Paris today?"}],
    "tools": [{"name": "lookup_weather",
               "description": "Look up the current weather for a city.",
               "parameters": {"city": {"description": "Name of the city.", "type": "str"}}}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

```json
{"raw_text": "[{\"arguments\":{\"city\":\"Paris\"},\"name\":\"lookup_weather\"}]",
 "parsed_call": {"name": "lookup_weather", "arguments": {"city": "Paris"}},
 "parse_status": "ok", "model_kind": "adapter"}
```

Tool schemas should use the xlam-style flat parameter map shown above (`"parameters": {"<param>": {"description": ..., "type": ...}}`). That is the shape the adapter was trained on; JSON-Schema-style `{"type": "object", "properties": ...}` tools are out of distribution and typically yield `invalid_json`.

The response reuses the [conservative evaluation parser](../concepts/evaluation.md) and never repairs output. `raw_text` is always returned for inspection, and `parse_status` tells you what the parser concluded:

| `parse_status` | Meaning |
|----------------|---------|
| `ok` | One schema-valid call parsed; `parsed_call` is populated |
| `no_json` | The output contains no opening `{` or `[` |
| `invalid_json` | A bracket exists but no balanced span parses as JSON |
| `invalid_shape` | Valid JSON, but not a single `{"name": str, "arguments": object}` call |

Input errors return HTTP 422 and other failures 500, both as `{"error": {"code": ..., "message": ...}}`; `GET /health` reports liveness. Every served request appends an event with its parse status to `<adapter-dir>/../logs/serve.jsonl`, so a session of manual probing leaves a record.

## Remote: vLLM on Modal

[`remote_serving.py`](https://github.com/AbdelStark/sommelier/blob/main/remote_serving.py) deploys the adapter behind vLLM's OpenAI-compatible server on a Modal GPU. It scales to zero when idle (five-minute scaledown window) and costs nothing while scaled down, but it is still an illustrative endpoint, not a product.

```bash
uv run modal deploy remote_serving.py   # deploy; the URL is printed
uv run modal run remote_serving.py      # smoke-test the deployed endpoint
```

One deployment registers two selectable models, so base-versus-adapter A/B requests hit the same endpoint with only the `model` field changed:

| `"model"` value | Serves |
|-----------------|--------|
| `sommelier-tool-caller` | Base model plus the trained LoRA adapter |
| `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | Base model alone |

Under the hood it runs `vllm serve` on the base model with `--dtype bfloat16`, `--enable-lora`, `--max-lora-rank 16`, and the adapter registered via `--lora-modules`. Launch-time knobs. `SOMMELIER_GPU` and `SOMMELIER_MAX_MODEL_LEN` are read from your shell environment at deploy time; the adapter-source variables and `SOMMELIER_SERVE_API_KEY` reach the container through the dotenv-backed Modal secret, so they belong in `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SOMMELIER_GPU` | `L40S` | GPU type for the serving container |
| `SOMMELIER_MAX_MODEL_LEN` | `4096` | vLLM context length; matches the training budget |
| `SOMMELIER_ADAPTER_VOLUME_PATH` | unset | Serve an adapter straight from the artifacts volume |
| `SOMMELIER_ADAPTER_REPO` / `SOMMELIER_ADAPTER_REVISION` | published adapter, `main` | Hugging Face adapter source |
| `SOMMELIER_SERVE_API_KEY` | unset | Require this value as a Bearer token |

Adapter resolution takes the first match: if `SOMMELIER_ADAPTER_VOLUME_PATH` is set (a path on the `sommelier-artifacts` volume, for example `artifacts/runs/nemotron-8b-full-3/train/adapter`), the server loads it directly and fails with a clear error if no `adapter_config.json` exists there. Otherwise it downloads the configured repo, pulling only the adapter weights, config, and tokenizer files. This makes "serve the adapter my [pipeline run](remote-execution.md) just produced" a one-variable change from "serve the published adapter".

!!! warning "Without an API key the endpoint is open"

    If `SOMMELIER_SERVE_API_KEY` is absent from `.env`, anyone with the URL can send requests that run on your billed GPU. The server logs a warning at startup but deploys anyway, because it is an optional, illustrative service either way. Set the key.

**Cold starts take minutes.** The first request after a scale-to-zero boots the container, loads the 8B model, and lets vLLM warm up; the deployment allows up to 20 minutes for startup. The smoke entrypoint accounts for this by polling `GET /v1/models` for up to 15 minutes (60 attempts, 15 s apart) and treating Modal's proxy redirect loops and connection resets as "not ready yet" rather than failures.

The smoke test is not a ping. It builds the canonical prompt through sommelier's own prompt policy, sends one tool-calling request at `temperature: 0.0`, and classifies the completion with the same conservative parser evaluation uses, asserting `parse_status == "ok"` and the right function name. A passing smoke therefore means the deployed endpoint reproduces evaluated behavior, not merely that a port is open. Pass `--model "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"` to smoke the base side instead, or `--url` to target a specific deployment.

Two operational details worth knowing:

- **Debugging a server that will not boot**: `uv run modal run remote_serving.py::diagnose` runs the exact serve command in the foreground and streams every engine log line to your terminal, polling the port until it opens, the process exits, or a 15-minute default wait runs out. Use it when the web server times out and Modal's retained container logs are not enough.
- **The image is built from a CUDA devel base** (`nvidia/cuda:12.8.1-devel-ubuntu24.04`) because vLLM's startup warm-up JIT-compiles kernels with `nvcc`, which slim Python images lack. The container runs vLLM's own entrypoint and never imports sommelier, so the package source is deliberately not mounted into it.

The deployment mounts three volumes: `sommelier-artifacts` (adapters from runs), `sommelier-hf-cache` (weights, shared with the [remote pipeline](remote-execution.md)), and `sommelier-vllm-cache` (compiled kernels, so warm-up work survives restarts).
