# 10 Glossary

- Status: Draft
- Target milestone: v1.0

## Adapter

The parameter-efficient weights trained on top of the base model. In v1.0 this means a LoRA adapter trained through the configured QLoRA path.

## Argument Exact Match

A metric that equals 1 for an example only when the parsed predicted argument object equals the gold argument object after canonical JSON normalization.

## Argument F1

A partial-credit metric over flattened argument key-value pairs. Precision counts predicted pairs that match gold; recall counts gold pairs recovered by the prediction.

## Base Model

The configured pretrained language model before Sommelier training.

## Comparison Report

The Markdown and JSON artifact that compares base and adapter metrics on the same held-out test split.

## Formatted Example

A prepared example rendered into the selected chat template, including prompt text and assistant target text.

## Full-Call Exact Match

A metric that equals 1 only when function-name accuracy and argument exact match are both 1.

## Gold Call

The expected tool call from the dataset, represented as a function name and JSON argument object.

## Held-Out Test Split

The split used only for final base and adapter evaluation. It is never used for training, hyperparameter selection, or prompt iteration after the reference baseline has been recorded.

## Manifest

A schema-versioned JSON document that records a stage's inputs, outputs, command, config digest, git commit, checksums, and status.

## Prepared Example

A validated row with parsed tool schemas, parsed gold calls, split assignment, and stable identifiers.

## Prompt Digest

The SHA-256 digest of the exact prompt text sent to the model for an example.

## Run ID

A stable identifier for one pipeline execution. All logs, manifests, and reports from the execution share it.

## Tool Schema

The JSON schema describing a callable tool name, description, and parameters.

## Valid-JSON Rate

The fraction of model outputs from which the parser extracts a syntactically valid JSON object or array with the expected tool-call shape.
