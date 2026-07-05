# Why everything is a file

Every output Sommelier produces is a plain file under one directory, and every stage records what it read, what it wrote, and what those bytes hash to. There is no database and no experiment-tracking service in the provenance path (optional [W&B tracking](../reference/configuration.md) is additive, never authoritative). The design goal: a run directory, zipped and handed to a stranger, contains everything needed to audit the run three weeks later, including which commit, which config, which inputs, and whether each stage actually finished.

## The artifact root and the run directory

`project.artifact_root` in the config names the root. It must be a relative path; the config loader rejects an absolute one, because manifests store artifact paths relative to this root and a machine-specific prefix would make them non-portable. Every pipeline invocation writes under `<artifact_root>/runs/<run_id>/`, where the run ID is a UTC timestamp plus a random suffix, for example `20260702T091500Z-3f9a1c2b`.

That format is doing quiet work. A rerun gets a fresh run ID by default, so it lands in a new directory instead of overwriting a previous successful stage directory. Smoke runs additionally get a `smoke-` prefix, so a later full run can never land on top of smoke artifacts. Inside the run directory, each stage owns one subdirectory (`data/`, `formatted/`, `train/`, `eval/base/`, `eval/adapter/`, `report/`). The full file-by-file layout lives in [Artifacts and schemas](../reference/artifacts.md).

## The manifest chain

Each stage writes a `<stage>_manifest.json` (schema `sommelier.manifest.v1`) when it finishes. Its fields are the whole reproducibility story in one object:

| Field | What it records |
|-------|-----------------|
| `stage` | One of `data`, `format`, `train`, `eval`, `report`, `serve` |
| `run_id` | The run this stage belongs to |
| `created_at` | UTC timestamp at write time |
| `git_commit` | `git rev-parse HEAD`, or `"unknown"` when git is unavailable |
| `config_sha256` | Digest of the run's `config.resolved.yaml` |
| `dependency_lock_sha256` | Digest of `uv.lock`, null when the lock file is absent |
| `command` | The argv that produced this stage |
| `seed` | `project.seed` from the config |
| `inputs`, `outputs` | Lists of `ArtifactRef` (below) |
| `status` | `succeeded` or `failed` |

A root `manifest.json` indexes the run: it maps each stage name to its stage manifest path, carries an `ArtifactRef` for the resolved config, and tracks overall run status (`running`, `succeeded`, or `failed`).

Every input and output is an `ArtifactRef`:

| Field | Meaning |
|-------|---------|
| `path` | POSIX path relative to the artifact root; a path that escapes the root fails schema validation |
| `kind` | What the file is (`dataset_split`, `generations`, `manifest`, ...) |
| `schema_version` | The schema its records claim |
| `sha256` | Digest of the file's bytes |
| `bytes` | File size |

Because manifests are written after their outputs, a manifest entry is a checkable claim: this file existed, at this size, hashing to this digest, when the stage declared success. The [comparison gate](determinism.md) is built on top of exactly these digests.

## Atomic writes

Artifacts are never written in place. `write_artifact_atomic` writes to a sibling temp file (`<name>.tmp.<pid>`), then moves it over the final name; if the writer fails, the temp file is deleted and the error propagates. So the final path only ever holds a complete file, and a process killed mid-write leaves at worst a temp file that no reader looks at. The `ArtifactRef` digest is computed from the published file, not the writer's buffer.

## Failure leaves a record

The manifest schema reserves a failed shape: `status: "failed"` plus two extra fields, `error_code` (the [SOM error code](../reference/errors.md)) and `error_message`. The message is redacted at build time: if it contains anything that looks like credential material (`hf_`, `sk-`, `ghp_`, `token`, `secret`, `password`, case-insensitive), the entire message is replaced with `stage failed; details redacted`; otherwise it is truncated to 500 characters. Every manifest additionally passes a secret scan before being written, and the scan fails the write rather than publishing a finding.

In the current pipeline, a stage that cannot meet its contract raises: the error propagates to the CLI with its exit code, and no manifest is written for that stage. The trust rule stays simple either way. Only files listed in a `succeeded` manifest are claimed by the run; a stage without one did not finish, and its outputs carry no claim. What you never get is a directory of plausible-looking files that passes for a finished run.

## Schema versioning: fail closed

Every JSON and JSONL record carries a `schema_version` field, and the package keeps a closed set of the fifteen versions it understands (`sommelier.config.v2`, `sommelier.manifest.v1`, `sommelier.formatted_example.v1`, and so on; the full list is in [Artifacts and schemas](../reference/artifacts.md)). Readers fail closed: a record with a missing or unrecognized `schema_version` raises a schema validation error (`SOM202`, exit 2); for an unrecognized version, the hint lists the supported ones.

The alternative, tolerating unknown versions and reading whatever fields happen to be present, is how silent semantic drift happens: a reader scores records whose meaning it does not actually know. Failing closed makes a format change a visible, deliberate event. When a schema needs to change, it gets a new version identifier, and old readers reject it by construction instead of misreading it.

## Why this design

The staged pipeline in [The pipeline](pipeline.md) only works because the files between stages are trustworthy. Three properties fall out of the manifest chain:

1. **Any claim in a report is traceable.** The report stage manifest names the evaluation reports it compared; those name the generations files they scored; those carry digests of the prompts they answered. The chain bottoms out at the raw dataset revision recorded in the config.
2. **Corruption is detectable.** If a file was edited, truncated, or swapped after a stage ran, its digest no longer matches the manifest.
3. **The evidence outlives the environment.** No service has to be up, and no account has to exist, for someone to verify a run. This is why the [reference run's evidence](../results/reference-run.md) can be published as a directory of files on the Hugging Face Hub.

The cost is verbosity: a full run writes manifests, digests, and logs alongside every payload. For a project whose entire purpose is a checkable claim, that trade is the right one.
