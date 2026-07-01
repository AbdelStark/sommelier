# RFC-0003: Tool-Call Chat Formatting

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.2

## Summary

Sommelier formats each prepared example as a three-message chat sequence: system instructions and available tools, user request, and assistant target containing only the gold JSON tool call. The formatter uses the selected tokenizer's chat template and records prompt digests so training and evaluation can prove prompt identity.

## Motivation

The PRD requires tools in the system message, the request in the user message, and the gold call as the target. The training objective depends on masking prompt tokens and training only on assistant tool-call tokens. Evaluation fairness depends on using the same prompt text for base and adapter runs.

## Goals

- Define one prompt policy for v1.0.
- Record rendered messages, prompt text, target text, full text, tokenizer ID, tokenizer revision, and prompt digest.
- Keep reasoning or explanatory text out of the assistant target.
- Support golden fixture tests for prompt stability.

## Non-Goals

- Prompt optimization.
- Multi-turn conversation formatting.
- Multiple model families with unrelated template policies.
- Chain-of-thought collection or training.

## Proposed Design

### Messages

```python
SYSTEM_INSTRUCTION = (
    "You are a tool-calling model. Select the correct tool and return only "
    "the JSON tool call. Do not include explanations."
)

def build_messages(example: PreparedExample) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": f"{SYSTEM_INSTRUCTION}\n\nAvailable tools:\n{json_tools}",
        },
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": json_gold_calls},
    ]
```

`json_tools` and `json_gold_calls` are canonical JSON strings with sorted keys and compact separators.

### Rendering

```python
def render_training_example(
    example: PreparedExample,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> FormattedExample: ...
```

The function renders:

- `prompt_text`: system and user messages with the generation prompt.
- `target_text`: assistant gold call.
- `full_text`: prompt plus target, rendered through the chat template.

The formatter stores `prompt_sha256 = sha256(prompt_text.encode("utf-8"))`.

### Evaluation Prompt

Evaluation uses `prompt_text` from `FormattedExample`. It does not reconstruct prompts from raw prepared rows unless the formatted artifact is being built.

### Training Labels

Training uses `full_text`, but the collator masks all prompt tokens and computes loss only on target tokens. If token boundary detection fails, training exits rather than falling back to full-sequence loss.

## Alternatives Considered

- Put tool schemas in a separate field outside chat messages. Rejected because the base model consumes chat text and the PRD specifies tools in the system message.
- Train on prompt and target tokens together. Rejected because it wastes capacity and can teach the model to reproduce prompts.
- Store only rendered text. Rejected because message-level fixtures are useful for auditing and future template migrations.
- Allow free-form assistant explanations. Rejected because the task is schema-valid tool calls.

## Drawbacks

- The prompt is tuned to single-call JSON outputs and may not generalize to multi-call plans.
- Tokenizer template changes can invalidate fixture digests.
- Keeping both messages and rendered text increases artifact size.

## Migration / Rollout

1. Add formatting fixtures with one valid example.
2. Implement canonical JSON serialization.
3. Implement tokenizer rendering and digest recording.
4. Implement collator boundary tests.
5. Use formatted artifacts in training and evaluation only.

## Testing Strategy

- Golden-test formatted messages and rendered prompt digests.
- Unit-test canonical JSON ordering.
- Unit-test that the assistant target contains no explanatory prefix.
- Unit-test collator masking for prompt and target tokens.
- Integration-test that base and adapter evaluation read identical `prompt_sha256` values.

## Open Questions

None for v1.0.

## References

- [03-data-model](../spec/03-data-model.md)
- [07-testing-strategy](../spec/07-testing-strategy.md)
