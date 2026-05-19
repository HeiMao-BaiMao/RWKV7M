# RWKV7M Implementation Progress 2026-05-19

- User goal: rebuild `RWKV7M.md` from `RWKV7M.paper.md`, complete JAX/Flax implementation, tests, inference/training, and installable `rwkv7m` library API.
- User requested Serena + CCC usage. Serena project activated; CCC initialized and indexed.
- `uv run pytest` initially failed before tests because safe-chain blocked freshly released `numpy==2.4.6` and `zipp==4.1.0` from `uv.lock`.
- Fixed uv reproducibility by constraining dependencies in `pyproject.toml`: `numpy>=2.2,<2.4`, `zipp>=3.20,<4.1`, bounded JAX/Flax/Optax, and `pytest>=8.4,<9`.
- `uv lock` succeeded and changed `numpy 2.4.6 -> 2.3.5`, `zipp 4.1.0 -> 3.23.1`, `pytest 9.0.3 -> 8.4.2`.
- `uv run pytest tests\test_screening_math.py tests\test_shapes.py -q` passed: 16 tests.
- Implemented so far:
  - `src/model/state.py`: added `LayerRWKVState` and real `init_rwkv_state`.
  - `src/model/rwkv_core.py`: TimeMix/ChannelMix now carry previous shifted hidden state; WKV recurrence state is threaded across chunks.
  - `src/model/screening.py`: added phase normalization; `read_only` alias maps to `read_screening_only`; unknown phase raises `ValueError`; scan carry is `(slots, ages)`; write params are always initialized when `use_write_screening=True`.
  - `src/model/screened_rwkv.py`: added `ModelConfig` validation; `v_first` now uses `zeros_like`; model returns updated RWKV state.
  - `src/model/__init__.py`: exports new state and phase helpers.
- Current remaining work:
  - Package restructure so `from rwkv7m import ...` works after `uv install --git` / `uv pip install git+...`; likely create `src/rwkv7m` and compatibility wrappers or move modules.
  - Add explicit tests requested by user: invalid phase, write-age accumulation, read_screening_only init then read_write apply, config validation, v_first safety.
  - Add library inference/training interface.
  - Update README, rebuild `RWKV7M.md`, optimize/replace `RWKV7M.paper.md`.
  - Run full `uv run pytest` and refresh CCC index after significant edits.
