# Overlay manifest

This overlay is based on the supplied Repomix snapshot of Sakana AI's `pc-alm` reference repository. The official synchronous implementation, including `pcalm/inference.py::run_pcalm`, is not modified.

## Included paths

| Path | Status | Reason |
|---|---|---|
| `ASYNC_EXPERIMENT.md` | NEW | Defines the pre-BEAM falsification question, scheduler semantics, work accounting, stale-read semantics, diagnostics, credit-front metric, and interpretation gate. |
| `OVERLAY_MANIFEST.md` | NEW | Lists overlay contents and validation actually performed. |
| `configs/async_smoke.yaml` | NEW | Small deterministic CPU dynamics experiment covering sync plus all required async modes. |
| `configs/async_dynamics.yaml` | NEW | Larger depth-32 synthetic experiment matrix spanning block size, rate heterogeneity, staleness, and fully-local scheduling. |
| `pcalm/async_inference.py` | NEW | Explicit seeded event simulator plus thin synchronous-oracle adapter; preserves the reference equations while changing scheduling semantics only. |
| `pcalm/async_metrics.py` | NEW | Residual/dual/activity metrics, BP gradient alignment, layerwise cosine, credit-front/dispersion metrics, and conservative propagation-exponent fitting. |
| `scripts/run_async_dynamics.py` | NEW | Fixed-weight dynamics harness writing config/summary/trace/layer/front/gradient/event machine-readable outputs and optional plots. |
| `tests/test_async_inference.py` | NEW | Tests determinism, zero-staleness equivalence, staleness bounds, mutation locality, fair accounting, local dual work, nonfinite rejection, and exact full-block reduction to the official reference. |
| `tests/test_async_metrics.py` | NEW | Tests credit-front orientation/dispersion and guarded propagation-exponent fitting. |

No file present in the supplied Repomix repository was modified.

## Validation actually run

Environment discovered in the execution sandbox:

- `uv 0.10.0`
- preinstalled `Python 3.13.5`
- preinstalled `jax 0.9.0.1`
- preinstalled `pytest 9.0.2`

The repository declares Python `>=3.10,<3.13`. `uv sync --extra test` was attempted first but **could not run** because the sandbox has no outbound DNS/network access and `uv` attempted to download CPython 3.11.14. No dependency or source-code failure was involved.

Commands/results:

| Command | Status |
|---|---|
| `uv sync --extra test` | **BLOCKED (environment)** — outbound DNS failure while downloading CPython 3.11.14. |
| `PYTHONPATH=. python3 -m pytest -q` before implementation | **PASS** — original 15 tests passed. |
| `PYTHONPATH=. python3 -m pytest -q tests/test_formulation.py tests/test_inputs.py` after implementation | **PASS** — 15/15 original tests. |
| `PYTHONPATH=. python3 -m pytest -q tests/test_async_inference.py tests/test_async_metrics.py` | **PASS** — 11/11 new tests. |
| `PYTHONPATH=. python3 -m pytest -q` | **PASS** — 26/26 total tests. |
| `PYTHONPATH=. python3 -m compileall -q pcalm scripts tests` | **PASS**. |
| `PYTHONPATH=. python3 scripts/run_async_dynamics.py --config configs/async_smoke.yaml` | **PASS** — all six modes reported `status: ok`; 30 local layer updates / 6 sweep equivalents per same-work mode. |
| Repeat smoke run to a second output directory plus byte comparison of `summary.json`, `trace.csv`, and `events.csv` | **PASS** — deterministic outputs. |
| Post-smoke accounting/stability QC script | **PASS** — all checkpoints finite; event counts sum to the requested budget; stale reads stayed within `tau_max`; every async mode had nonzero final activity-state distance from the same-work sync control. |

Smoke same-work final activity-state distances from the synchronous control were nonzero for every async mode (approximately `0.074` to `0.186` in this small run), confirming that the experimental schedulers exercise genuinely different trajectories. These numbers are QC evidence only, not scientific conclusions from a multi-seed study.
