## Project overview

Marimo + Plotly dashboard for monitoring the **statistical stability of model input features** over time.

- `simulation.py` — synthetic data layer: `FeatureSpec`, `DashboardConfig`, `simulate_feature_stats()`, `detect_instability()`
- `dashboard.py` — Marimo reactive notebook; run with `marimo run dashboard.py` (production) or `marimo edit dashboard.py` (dev)

## Running

```bash
marimo edit dashboard.py    # development mode
marimo run  dashboard.py    # production mode (code hidden)
```

Smoke-test the data layer alone:

```bash
python -c "import simulation; df = simulation.simulate_feature_stats(simulation.CONFIG); print(df.shape, list(df.columns))"
# expected: (60, 25)
```

## Coding Standards

### Type annotations
- All public functions and methods must have full type annotations (parameters + return type), including `-> None`.
- Use native union syntax (`X | Y`, `list[X]`) — Python 3.12 floor, no `from __future__ import annotations` needed.
- Prefer concrete types over `Any`; use `Any` only when truly unavoidable and leave a comment explaining why.

### Docstrings
- Every public module, class, and function must have a one-line summary in imperative mood ("Return …", "Apply …", "Raise …").
- Use [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for multi-line docstrings: `Args:`, `Returns:`, `Raises:`, and `Attributes:` sections.
- **Class docstrings** describe the class's responsibility. Include an `Attributes:` section for non-obvious fields.
- **Functions/methods**: add `Args:` for non-obvious parameters and `Returns:` when the shape or conditions of the return value are non-trivial. Omit sections that would only restate the type annotation.
- Private helpers (`_name`) do not need docstrings unless the logic is subtle.

### Dataclasses & immutability
- Use `@dataclass(frozen=True)` for config/value objects that must not change after creation (see `DashboardConfig`, `FeatureSpec`).
- Use plain `@dataclass` for mutable state objects.

### Constants & magic numbers
- Domain constants belong in `DashboardConfig`; avoid bare numeric literals in logic files.
- Module-level constants (non-config) go at the top of the module in `SCREAMING_SNAKE_CASE`.

### No bare `print`
- Use `rich.console.Console` for any terminal output; never use `print()` in library code.

### Error handling
- Raise `ValueError` / `TypeError` at public API boundaries with a descriptive message.
- Do not swallow exceptions silently; re-raise or log.
- Avoid returning `None` to signal failure — raise instead.

## Marimo conventions

- Each cell function's parameters must exactly match the names of values returned by upstream cells — Marimo resolves the reactive graph by name.
- Use `mo.output.replace(...)` inside cells that produce UI output; do not rely on implicit last-expression rendering for complex layouts.
- `mo.ui.dropdown` is used for single-feature selection; its `.value` is the currently selected string option.
- Marimo auto-sorts cell parameter lists alphabetically; this is expected and does not affect functionality.
