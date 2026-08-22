---
paths:
  - "console/**"
  - "src/interfaces/web/**"
  - "tests/interfaces/web/**"
---

# The console

The console is the surface a human reads; `poker-solver` is the scriptable one.
The console reimplements nothing.

- **It may COMPOSE command payloads. It may not COMPUTE one.** A join may
  filter, group and cross-reference; it may not derive a quantity no command can
  answer. `views.py` may import the command registry and **nothing else**, which
  `test_no_second_read_path.py` checks. The line in practice: the run list ships
  `task_id -> run_id`, and the CLIENT decides which Batch states mean "running".
- **Coverage is complete and enforced, at two grains.** `test_command_coverage.py`
  fails until a command has an endpoint or a declared reason — `NO_PAYLOAD` (`serve`,
  `blueprint-serve` never return one) or `NODE_ONLY` (node compute, each listed
  with the dispatching command that IS the console's door). `status` is covered
  by the three panels it composes — do not add `/api/status`.
- **Never hand-write a schema in the console.**
  `src/interfaces/web/contract.py` → `console/src/api/openapi.json` (via
  `response_model`; a test fails if it is stale) → `types.gen.ts` (regenerated
  by `npm run build`) → `api/types.ts`, which only names them. The hand-written
  Zod this replaced disagreed with `cancel` about `job_id`/`task_id`, so the
  console terminated a task and then reported that it had failed.
- **A payload's shape is declared by the command that PRODUCES it, and
  `contract.py` imports it.** `run()` returns a model; `render()` takes the same
  model. Measured 08-13: while `contract.py` restated the shape, renaming a
  REQUIRED field in `jobs.py` passed 1061 tests — both the model and the
  `PAYLOADS` fixture were hand-written and drifted together. `ty` (pre-commit)
  catches it now. `_base.Payload` is `BaseModel`: the union with `dict` is gone,
  so a handler returning a dict fails pre-commit rather than a shape nobody checks.
- **Never re-pass a value equal to a command's parser default.**
  `Command.invoke` fills every one from `add_arguments`, so `rate=""` or
  `tasks_dir=None` at a call site is a second declaration that can drift.
  `/api/box` had drifted this way — it restated `serve-box`'s resource group, VM
  and subscription, putting the blueprint host's name in two files — so
  `test_endpoint_arguments.py` now checks it. Spelling `action="status"` there is
  the one allowed exception: those three endpoints differ in exactly one argument
  and read as a set.
- **A trimmed view part gets its own TYPE.** `views._summarised` returns a
  `TasksSummary`, which has no `rows` field, so TypeScript cannot offer one. One
  model describing both the full log and the stub meant `parts.tasks.payload.rows`
  was `[]` on a trimmed part and correct about nothing.
- **`response_model` is for OpenAPI only.** FastAPI skips validation for a
  handler returning a `Response`, and every endpoint returns `PayloadResponse`
  to keep `jsonio.dumps`. Enforcement is `test_contract.py`.
- **An optional body field means *omitted*.** `web.app.given()` drops it so the
  command's own parser supplies the default — never re-declare a default in a
  request model. That is also what makes `compact-legs` default to the dry run.
  **A dispatching command's flags must ALL have a field**, which is the second
  grain `test_endpoint_arguments.py` guards: `SubmitBody` was missing `--kernel`
  and the six flags it gates, so the console could queue neither a board-free nor
  a warm-started run while `submit` had had an endpoint all along. A gap at this
  grain reads as an unbuilt feature, not a missing field.
- **Guard flags (`--force`, `--delete`) are opt-in, never pre-checked.**
- **Writes use `TtlCache(0.0)`.** A dispatch must not be memoised.
- **Two shapes of endpoint, and both go through a command.** One per command is
  the grain for an ad-hoc question; `/api/view/{now,runs,run/{id},experiment/{id}}`
  fan several out concurrently via `src/interfaces/commands/_compose.py` and
  join in `src/interfaces/web/views.py`.
