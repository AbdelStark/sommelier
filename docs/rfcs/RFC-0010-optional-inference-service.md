# RFC-0010: Optional Inference Service

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v1.0

## Summary

Sommelier includes an optional single-adapter inference service that exposes an OpenAI-compatible chat completions endpoint for demonstration and manual inspection. It is not a production serving system and is not required for the core evaluation claim.

## Motivation

The PRD lists serving as optional and illustrative. A bounded service lets users try the trained adapter through a familiar request shape without expanding v1.0 into deployment, scaling, or multi-tenant operations.

## Goals

- Serve one configured adapter with the same prompt policy used in evaluation.
- Accept chat-completion-like requests with tool schemas.
- Return model text and parser status.
- Document that the service is illustrative.

## Non-Goals

- Autoscaling.
- Authentication beyond the remote provider boundary.
- Multi-tenant isolation.
- Streaming.
- Production latency guarantees.

## Proposed Design

### Command

```text
sommelier serve adapter --config config.yaml --adapter artifacts/runs/<run_id>/train/adapter
```

### Request Shape

```python
class ServeRequest(TypedDict):
    messages: list[ChatMessage]
    tools: list[ToolSchema]
    temperature: Literal[0.0]
    max_tokens: int
```

### Response Shape

```python
class ServeResponse(TypedDict):
    raw_text: str
    parsed_call: ToolCall | None
    parse_status: ParseStatus
    model_kind: Literal["adapter"]
```

The service reuses `parse_tool_call` from RFC-0005 and logs parse status.

### Prompt Policy

The service uses the same system instruction and canonical tool schema serialization as formatting. It does not use private prompt variants.

## Alternatives Considered

- Omit serving entirely. Rejected because the PRD includes optional serving and it is useful for manual inspection.
- Build production OpenAI-compatible infrastructure. Rejected because it exceeds v1.0 scope.
- Merge the adapter permanently into a new model artifact. Rejected for v1.0 because adapter-only artifacts are easier to license and inspect.

## Drawbacks

- Users may mistake the service for production-ready deployment.
- Inference dependencies add another optional environment.
- Serving can produce outputs that differ from batch evaluation if request formatting drifts.

## Migration / Rollout

1. Implement request/response schema tests.
2. Reuse formatter and parser functions.
3. Add remote serving entrypoint.
4. Document limitations in README and report.

## Testing Strategy

- Unit-test request validation.
- Unit-test prompt construction parity with evaluation.
- Unit-test response parser status.
- Smoke-test service startup with a fixture or tiny model stub.
- Docs test that serving is labeled optional and illustrative.

## Open Questions

None for v1.0.

## References

- [02-public-api](../spec/02-public-api.md)
- [06-security](../spec/06-security.md)
