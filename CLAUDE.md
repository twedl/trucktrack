# CLAUDE.md

## Build

```
uv run maturin develop
```

## Test

```
uv run pytest tests/ -v
```

## Lint

```
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
uv run ruff check python/ tests/
uv run ruff format --check python/ tests/
uv run maturin develop --release && uv run mypy python/trucktrack
```

## Version check

pyproject.toml and Cargo.toml versions must match:

```
PY_VER=$(grep -E '^version[[:space:]]*=' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
RS_VER=$(grep -E '^version[[:space:]]*=' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
[ "$PY_VER" = "$RS_VER" ] || echo "MISMATCH: pyproject.toml=$PY_VER Cargo.toml=$RS_VER"
```
