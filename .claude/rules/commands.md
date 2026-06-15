---
paths:
  - "src/interfaces/commands/**"
  - "src/interfaces/cli/**"
  - "src/interfaces/errors.py"
  - "src/interfaces/telemetry.py"
  - "tests/interfaces/commands/**"
---

# The command layer

A subcommand is one module under `src/interfaces/commands/`, listed once under
one of the four `GROUPS`. The package sits beside `cli/` and `web/`, not inside
either.

- **`Command` carries parser, handler AND renderer together.** When those lived
  apart, a command borrowed another's renderer and died on a missing key.
- **`render()` is deliberately NOT abstracted.** It is the terminal's renderer;
  for any other surface the payload is the interface.
- **The registry is `CommandRef` — a name and a help line — and importing it
  imports NO handler.** Eagerly importing every module cost 1.2s on every
  invocation, `--help` included. Do not add an import that loads handlers.
- **Grouping is STRUCTURAL: a ref lives inside one `CommandGroup`, and
  `COMMANDS` is derived by flattening them.** So a command is in exactly one
  group by construction — no label to typo, no default to fall into. `--help`
  renders the groups from `headless._listing()` as the parser's epilog, and
  `add_parser` is given no `help=`, which is what suppresses argparse's own
  flat block. Do not re-add it: that block, plus the choice list `metavar`
  replaces, is how `--help` came to print every name twice in one long token.
- **`Command.invoke(**kwargs)` builds arguments from the command's own parser**,
  so a second surface cannot drift from it, and returns the payload unrendered.
- **Refusals are values.** Anything the caller could have got right raises
  `CommandError`; a bug still tracebacks. Only `headless.py` turns a
  `CommandError` into stderr + exit 1. **Never `raise SystemExit` in a command
  module** — a guard test fails if one reappears.
- **`errors.attempt` decides once which failures a surface may survive**:
  `refusal` (understood, the answer is no → 422) or `unavailable` (Azure did not
  answer → 503). The Azure SDK's `ClientAuthenticationError`/`HttpResponseError`
  are caught THERE and nowhere else; a guard test fails if anything under
  `interfaces/` names either again. The SDK imports lazily inside `attempt`,
  because `azure.core.exceptions` costs 76ms against a 0.18s `--help`.
- **`Command.execute` is the observed seam, not `invoke`** — the command line
  parses argv and calls the handler, so anything wrapped around `invoke` sees
  the console and misses the CLI.
- **Telemetry writes are best-effort and must stay that way.** Laptop-local
  under `$POKER_SOLVER_CACHE`, never the share: the share has no atomic append,
  a document per invocation would outgrow `legs/` in hours, and every write
  there would add a round trip to the thing being measured.
  `POKER_SOLVER_TELEMETRY=0` turns it off, which is what the test suite does.
- **Both doors serialise through `shared.jsonio.dumps`** — `--json` and the
  console's `PayloadResponse` — so a numpy scalar or `Path` cannot print fine on
  one surface and 500 the other. The console keeps `allow_nan=False`, because
  `JSON.parse` rejects `NaN`.
- **Readers take no `--source` and no `--runs-dir`**; they answer against the
  published record, materialised into a temp tree and discarded. The eval index
  is rebuilt from per-run documents on every read, so `ledger --rebuild` and
  `--migrate` do not exist.
- **`train-static` covers starting AND continuing.** `--iterations` is an
  ABSOLUTE target and `--run <id>` continues an existing directory, so
  re-running past the target is a no-op — that is what makes a scheduler retry
  converge instead of training twice. There is no separate `resume`.
- **Nothing on the laptop reads a card abstraction.** Submitting must not call
  `build_card_abstraction` — it loads ~773 MB to answer a question about another
  machine. Config overrides go through `--set k=v` and nothing else.
