# Agent Guidelines for McDermott Health AI Lab Projects

This file is read by AI coding agents (Claude Code, Cursor, Copilot, Codex CLI, Aider, Gemini CLI,
and others). `CLAUDE.md` is a symlink to this file.

## Read `CONTRIBUTORS.md` first

**The project's conventions live in [`CONTRIBUTORS.md`](CONTRIBUTORS.md): build system (uv),
testing, code style, issue/PR workflow, repository conventions.** Follow them. The notes below are
agent-specific additions and reminders that build on those conventions — they are not a substitute.

## Agent-specific tooling

- **Use `gh` CLI for all GitHub operations** (PRs, issues, code search, actions logs). Do **not**
  use the GitHub MCP server — `gh` is faster, more reliable, and uses far fewer tokens for the
  same operations.
- **Doctest namespace is pre-populated.** The project root `conftest.py` registers `Path`,
  `datetime`, `tempfile`, plus `yaml_to_disk` and `print_directory` / `PrintConfig` (if installed)
  into the doctest namespace. You rarely need explicit imports inside doctests.

## Working style

- **TDD.** When fixing bugs or adding features, write a failing test first, confirm the failure
  captures the intended behavior, then implement the fix.
- **Pre-commit hooks auto-fix many issues** (formatting, import sorting, EOF newlines, markdown
  format). Run `uv run pre-commit run --all-files` before committing — it will modify files. Stage
  the result.
- **Doctests are first-class.** Prefer adding API-validating tests as doctests in docstrings or
  markdown files; standalone `tests/**/test_*.py` files are for cases where a doctest would be
  excessively long or unclear.

## What not to do

- Do not use `pip install` directly — use `uv sync` or `uv add` (see CONTRIBUTORS.md "Build
  System: uv").
- Do not skip or disable pre-commit hooks (no `--no-verify`).
- Do not add broad `# noqa` or `# type: ignore` comments without a specific rule code and a
  one-line justification.
- Do not modify CI workflows (`.github/workflows/*.yaml`) without discussing with maintainers
  first. Workflow security is enforced by `zizmor` (see `.github/zizmor.yml`); changes there often
  surface real security findings.
- Do not commit data files or secrets. `gitleaks` runs as a pre-commit hook but treat it as a
  backstop, not a substitute for care.

## Project-specific notes for `simple-scaling-laws`

- **The package's whole value is its statistical defaults.** Before changing anything in
    `fitting.py`, `uncertainty.py` or `data.py`, read the module docstring: each one states *why*
    the current choice was made (run-level aggregation, wild cluster bootstrap over configurations,
    Webb weights, leverage correction, replicate-vs-residual run variance). If you change a default,
    change the docstring's justification with it, and add a test that pins the new behavior.
- **Run-level aggregation is load-bearing.** Every fit and every diagnostic works on one observation
    per `training_run_id`. Anything that starts treating evaluation rows as independent observations
    is a bug, and `tests/test_variance_components.py` exists to catch it.
- **Never fit on a degenerate design.** Constant targets and predictors held fixed are short-circuited
    on purpose: handing them to `scipy.optimize.least_squares` costs thousands of vanishing
    trust-region steps per fit. If you touch that path, re-check the suite's wall-clock.
- **Everything is seeded.** `fit(..., seed=...)` and `predict(..., seed=...)` must stay
    reproducible, because `tests/test_serialization.py` asserts a reloaded artifact predicts
    bit-identically.
- **Warnings are data, not logging.** New failure modes get a `Note` with a stable `code`, and are
    persisted into `diagnostics.json`. Callers branch on those codes.
- **Doctests carry the documentation.** `README.md` and every module docstring are executed by
    `pytest`. Keep doctest output width-independent (no wide Polars table renderings) so it does not
    depend on terminal size in CI.
