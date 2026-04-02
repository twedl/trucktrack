# trucktrack

High-performance Python package with a Rust extension backend (PyO3 + maturin).

## Requirements

- Python 3.11+
- Rust stable toolchain
- maturin

## Quick Start

```bash
# 1. Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install maturin and dev dependencies
pip install "maturin>=1.7,<2.0" pytest pytest-benchmark ruff mypy

# 4. Build and install the Rust extension
maturin develop

# 5. Run tests
pytest tests/ -v
```

## Usage

```python
import trucktrack

trucktrack.fast_sum(list(range(1_000_000)))   # -> 499999500000
trucktrack.fibonacci(30)                      # -> 832040
trucktrack.word_count("hello world foo bar")  # -> 4
print(trucktrack.__version__)                 # -> 0.1.0
```

## Project Layout

```
.
├── pyproject.toml          # Build config (maturin backend)
├── Cargo.toml              # Rust crate manifest
├── src/
│   └── lib.rs              # Rust source
└── python/
    └── trucktrack/
        ├── __init__.py     # Public Python API
        ├── _core.pyi       # Type stubs
        └── py.typed        # PEP 561 marker
```

## Dev Workflow

| Task | Command |
|------|---------|
| Rebuild after Rust changes | `maturin develop` |
| Optimized build | `maturin develop --release` |
| Run tests | `pytest tests/ -v` |
| Type-check | `mypy python/trucktrack` |
| Lint Python | `ruff check python/ tests/` |
| Lint Rust | `cargo clippy` |
| Build wheel | `maturin build --release` |
| Publish | `maturin publish` |

## Adding New Rust Functions

1. Add a `#[pyfunction]` in `src/lib.rs`
2. Register it in the `#[pymodule]` block
3. Add a corresponding entry to `python/trucktrack/_core.pyi`
4. Re-export from `python/trucktrack/__init__.py`
5. Run `maturin develop` and add tests
