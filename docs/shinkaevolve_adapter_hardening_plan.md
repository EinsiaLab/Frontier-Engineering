# ShinkaEvolve Adapter Hardening Plan

## Scope

This document describes how to harden the Frontier Eval -> ShinkaEvolve adapter
without switching to native Shinka task directories.

The goal is not to mimic OpenEvolve exactly. The goal is to make the current
adapter robust, context-rich, and predictable for Frontier Eval tasks.

Primary target:

- `frontier_eval/algorithms/shinkaevolve/shinkaevolve_entrypoint.py`

Secondary target:

- unit tests around feedback synthesis and failure paths

## Why The Adapter Is Not A One-Liner

ShinkaEvolve can be very simple when a task already exists in Shinka's native
task-dir format:

- `initial.<ext>`
- `evaluate.py`
- `metrics.json`
- `correct.json`

Frontier Eval is different:

- tasks already implement `Task.evaluate_program(...)`
- most tasks return `EvaluationResult(metrics, artifacts)`
- many tasks produce rich artifacts that should not be discarded
- task configuration, repo root, and timeouts need to be forwarded into the
  evaluation subprocess
- we want Shinka's standard optimization and fix loops to receive useful
  context, not just a pass/fail bit

Because of that, the adapter must do more than invoke a command. It must
translate Frontier Eval task outputs into the context channels Shinka actually
uses.

## Current Problems

### 1. `text_feedback` is underpowered

The current implementation uses a narrow, partially task-specific whitelist when
synthesizing `metrics["text_feedback"]`.

This causes real failures to be missed:

- `program_stderr`
- `program_stdout`
- `traceback`
- task-specific raw outputs
- important collected artifacts from unified tasks

Result:

- optimization prompts often see only a generic error string
- the root cause is lost even when the task already produced it

### 2. Shinka has two context channels, but we only feed one well

Shinka uses:

- `text_feedback` in regular optimization prompts
- `stdout_log` / `stderr_log` in fix prompts

The current adapter writes `metrics.json`, `correct.json`, and `artifacts.json`,
but it does not deliberately bridge Frontier task artifacts into the job logs
that Shinka later reads as `stdout_log` and `stderr_log`.

Result:

- fix prompts may have little or no actionable raw error output

### 3. Exception paths lose detail

When the entrypoint itself fails, the current logic usually keeps only
`str(exception)`.

Result:

- traceback information is lost
- failures in the adapter layer are hard to diagnose
- later generations get weak repair context

### 4. Context budgeting is implicit and fragile

There is no clear total budget for:

- synthesized feedback
- raw stderr bridge
- raw stdout bridge

Result:

- either too much noisy context or too little signal
- behavior depends on ad hoc truncation points

### 5. Tests do not cover the real failure modes

Current tests cover a small part of `text_feedback` synthesis, but they do not
protect against:

- traceback omission
- `*_stderr_full` vs `*_stderr` priority
- unified task collected artifacts
- fix-prompt bridge quality
- exception-path regressions

## Design Principles

1. Preserve Frontier Eval as the source of truth.
2. Keep the Shinka adapter generic across tasks.
3. Separate summarized context from raw evidence.
4. Prefer deterministic ordering over ad hoc whitelists.
5. Never silently discard the primary diagnosis.
6. Make failures debuggable from run artifacts on disk.

## Proposed Design

## 1. Introduce A `ContextBundle`

Add an internal context-building layer in the entrypoint.

Suggested shape:

```python
@dataclass
class ContextBundle:
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    correct: bool
    primary_error: str
    text_feedback: str
    stdout_bridge: str
    stderr_bridge: str
    selected_keys: list[str]
    omitted_keys: list[str]
```

This bundle is not a new public API. It is an internal representation used to
drive all outputs consistently.

## 2. Normalize Evaluation Results First

The entrypoint should first normalize any task result into:

- `metrics`
- `artifacts`

Supported cases:

- plain `dict` metrics
- nested `{"metrics": ..., "artifacts": ...}`
- `EvaluationResult`

If evaluation raises:

- set `metrics = {"combined_score": 0.0, "valid": 0.0, "error": str(e)}`
- set `artifacts["error_message"] = str(e)`
- set `artifacts["traceback"] = traceback.format_exc()`

This guarantees the later context builder always runs on a consistent input.

## 3. Classify Artifacts By Role

Do not use a narrow static whitelist. Use role-based classification.

### Diagnostic keys

Highest priority:

- `error_message`
- `failure_summary`
- `traceback`
- `readonly_violations`
- `metrics_json_error`
- `artifacts_json_error`

### Raw evidence keys

High priority:

- `*_stderr_full`
- `*_stderr`
- `*_stdout_full`
- `*_stdout`
- `*log*`
- `check`
- `score_line`

### Task context keys

Medium priority:

- `constraints`
- `interface_contract`
- `task_spec`
- `task_spec_zh_cn`
- `agent_files`
- selected `agent_file::*`
- selected `collected_artifact::*`

### Everything else

Keep as fallback context if budget allows.

## 4. Build `text_feedback` As A Summary, Not A Dump

`text_feedback` should be optimized for regular Shinka optimization prompts.
It should be compact, structured, and repeatable across generations.

Suggested structure:

### Outcome

- `combined_score`
- `valid`
- `timeout`
- key return codes
- `runtime_s`

### Primary Diagnosis

- `error_message`
- `failure_summary`
- traceback summary

### Key Evidence

1-3 short excerpts selected from the most relevant raw outputs.

Priority:

1. `*_stderr_full`
2. `*_stderr`
3. `traceback`
4. `*_stdout_full`
5. `*_stdout`
6. `*log*`

### Constraints And Interface

- `constraints`
- `interface_contract`

### Task Context

- one task spec block if available
- a few selected file excerpts if they are likely to explain the failure

### Omitted Context

If the budget is exceeded, add a final line listing omitted keys instead of
dropping them silently.

## 5. Build Raw Log Bridges For Fix Prompts

This is the key Shinka-specific improvement.

Fix prompts use `stdout_log` and `stderr_log`, which Shinka loads from
`job_log.out` and `job_log.err`.

The adapter should therefore create:

- `stdout_bridge`
- `stderr_bridge`

### `stderr_bridge`

Should contain the highest-value debugging evidence:

- `error_message`
- `failure_summary`
- `traceback`
- best `*_stderr_full` or `*_stderr`
- fallback diagnostic summary if no stderr exists

### `stdout_bridge`

Should contain useful execution context:

- best `*_stdout_full` or `*_stdout`
- selected benchmark summaries
- selected logs if they are informative

### Delivery mechanism

Before the entrypoint exits, print:

- `stdout_bridge` to real stdout
- `stderr_bridge` to real stderr

This allows Shinka's `job_log.out` / `job_log.err` capture to receive the
bridged evidence without modifying Shinka internals.

## 6. Persist More Debug Artifacts On Disk

In addition to the existing outputs, also write:

- `text_feedback.txt`
- `stdout_bridge.txt`
- `stderr_bridge.txt`
- `context_manifest.json`

Suggested manifest contents:

- selected keys
- omitted keys
- per-section character counts
- primary error
- whether evaluation raised before returning a result

This makes adapter behavior inspectable without reading code.

## 7. Budget Strategy

Use explicit budgets, with separate limits for summary and raw evidence.

Suggested defaults:

- `text_feedback`: 8_000 chars
- `stderr_bridge`: 12_000 chars
- `stdout_bridge`: 6_000 chars
- per artifact excerpt: 2_000-4_000 chars depending on class

Heuristic:

- incorrect programs bias budget toward diagnosis and stderr
- correct but low-performing programs bias budget toward constraints, benchmark
  checks, and selected task context

## 8. Key Selection Heuristics

Use deterministic ordering.

Rules:

1. Prefer `foo_full` over `foo` when both exist.
2. Deduplicate equal text blocks.
3. Keep at most a few file excerpts in the summary path.
4. Never omit the primary error if present.
5. Never omit traceback if evaluation raised in the adapter.

For file excerpts:

- prefer `runtime/problem.*`
- prefer `baseline/solution.*`
- prefer task spec / README / interface files
- only include them when they add information beyond logs

## 9. Suggested Internal Functions

Keep the implementation in the adapter layer. Suggested helpers:

- `_extract_metrics_and_artifacts(result)`
- `_capture_exception_context(exc)`
- `_classify_artifact_keys(artifacts)`
- `_choose_primary_error(artifacts)`
- `_select_summary_evidence(artifacts)`
- `_select_stdout_bridge(artifacts)`
- `_select_stderr_bridge(artifacts)`
- `_render_text_feedback(bundle_like_inputs)`
- `_write_context_outputs(results_dir, bundle)`

If the entrypoint file becomes too large, move the context logic into a local
module such as:

- `frontier_eval/algorithms/shinkaevolve/context_feedback.py`

## 10. Output Contract From The Entrypoint

After evaluation and context construction:

- write `metrics.json`
- write `correct.json`
- write `artifacts.json` when non-empty
- write the new debug files
- print bridged stdout/stderr
- always return exit code `0`

Keeping exit code `0` is still correct because Shinka expects to read
`metrics.json` / `correct.json` instead of treating a non-zero exit code as a
recoverable task failure.

## Testing Plan

## Unit tests

Add tests for:

1. nested result extraction
2. traceback capture on exception
3. `*_stderr_full` preferred over `*_stderr`
4. `program_stderr` included in text feedback
5. `traceback` included in text feedback
6. `user_artifact::error_message` promoted correctly
7. unified task `agent_file::*` and `collected_artifact::*` selection
8. omitted-key reporting when over budget
9. stdout/stderr bridge generation
10. deduplication of repeated diagnostics

## Regression tests

Create synthetic fixtures resembling:

- `smoke`
- `malloclab`
- `mla`
- `car_aerodynamics_sensing`
- unified tasks with `artifacts.json`

These should verify that the adapter keeps the most important failure evidence
for each style of task.

## Rollout Plan

Phase 1:

- refactor entrypoint into `ContextBundle`
- preserve current behavior where possible
- add debug outputs and tests

Phase 2:

- improve evidence selection and budgets
- tune key ranking based on real run traces

Phase 3:

- optionally share some context-building utilities with AB-MCTS if useful

## Acceptance Criteria

The adapter is considered hardened when all of the following are true:

1. any adapter exception produces both `error_message` and traceback context
2. incorrect programs always provide useful fix-prompt stderr context
3. common task stderr fields are visible in synthesized feedback
4. the primary diagnosis is never lost due to ordering or truncation
5. context generation is deterministic and test-covered
6. a run directory clearly shows what context the adapter synthesized

## Recommended Next Implementation Order

1. implement `ContextBundle` and normalization
2. implement stdout/stderr bridges
3. replace the old feedback whitelist with role-based selection
4. add disk debug artifacts
5. expand tests

