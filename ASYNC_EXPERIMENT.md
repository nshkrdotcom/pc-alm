# Pre-BEAM asynchronous PC-ALM falsification experiment

## Research question

This experiment asks whether the fast backward credit propagation reported for the published, globally synchronous PC-ALM algorithm is robust to changes in **execution and scheduling semantics**. It deliberately does **not** implement BEAM, Elixir, actors, GenServers, Akka, distributed execution, or a new training system.

The official synchronous implementation in `pcalm/inference.py::run_pcalm` remains unchanged and is the control oracle. The new code reuses the same residual MLP, supervised loss, hidden-edge constraints, shifted augmented-Lagrangian energy, state variables, activity-step batch scaling, `alpha`, and `rho`. The manipulated variable is when a free-activity layer updates, which other activity versions it reads, and when dual state is advanced.

## Schedulers

Layer/activity indices follow the repository's natural order: **0 is closest to the input; `L_free - 1` is closest to the output**.

### `sync`

Calls the existing `run_pcalm` implementation. No synchronous primal solver is duplicated. A thin adapter reconstructs the actual post-update dual state when the configured weight-credit timing is `pre_dual_energy`.

### `random_coordinate`

One seeded uniformly selected free layer is updated per event. The partial derivative is obtained from the same complete augmented-Lagrangian objective, but only the selected activity is mutated. Each event observes state changes from all prior events, giving Gauss-Seidel/coordinate-style execution rather than the reference Jacobi-style simultaneous activity step.

Dual updates remain globally synchronized after the same amount of primal work as the reference: every `inner_steps * L_free` local layer updates, all hidden-edge duals advance once. This mode therefore isolates asynchronous **primal** scheduling.

### `random_block`

A seeded block of distinct free layers is selected per event. All selected layers use gradients from the same pre-event state and are mutated simultaneously; the next event sees those changes. Block size is configurable. Events are clipped at a dual-update boundary so the reference dual cadence remains exact in work units.

A useful invariant follows: with `block_size == L_free`, one block event updates every free layer from one common state. Repeating those events for `inner_steps` before each global dual update reduces to the official reference semantics. This is tested directly against `run_pcalm`.

### `heterogeneous_rates`

One layer updates per event, but selection probabilities are nonuniform. Rates are generated reproducibly from a seeded log-normal distribution and normalized to probabilities. `heterogeneous_rate_strength=0` is exactly uniform; larger values increase scheduler-rate skew. Numerical state learning rates are not randomized.

Global dual updates retain the reference work cadence, as in `random_coordinate`.

### `bounded_staleness`

One uniformly selected layer updates per event. The simulator keeps a versioned activity history. For an event at state version `v`, it samples a delay `d` in `[0, tau_max]` and reads the activity snapshot at version `v-d` (clipped only by available early history). The selected layer's **own** activity is always its current value; staleness applies to the neighboring activities used by its partial derivative. Duals are current in this experiment, so `tau_max` isolates stale **activity-neighbor** reads rather than mixing activity and dual staleness.

`events.csv` records requested delay, actual delay, read version, and current state version. `tau_max=0` is tested to be exactly equivalent to the nonstale `random_coordinate` schedule with the same seed.

### `fully_async_local`

This is the stronger event-driven approximation intended to probe whether a later actor experiment is scientifically justified. One local unit is selected per event, optionally under heterogeneous rates and bounded activity-neighbor staleness. It:

1. reads its available neighboring activity snapshot;
2. updates only its owned activity using the corresponding partial derivative of the same augmented-Lagrangian objective;
3. immediately updates only its owned hidden-edge dual `lambda_i` using its local constraint residual.

There is no global dual barrier. Other duals advance only when their owning layer receives an event. This is **not** claimed to be mathematically identical to published PC-ALM. It intentionally changes the execution semantics more strongly. In this first falsification experiment, dual values used by the primal partial derivative are current; only activity-neighbor reads can be stale.

## Work accounting

The primary unit is:

```text
layer_update_events
```

Updating one free activity layer once costs one event-unit. Updating a block of `k` layers costs `k` layer-update events. A synchronous reference activity step updating all `L_free` activities costs `L_free` layer-update events.

The common time axis is:

```text
sweep_equivalents = layer_update_events / L_free
```

With reference `inner_steps > 1`, one PC-ALM outer iteration consumes `inner_steps` sweep equivalents before its global dual update.

Dual work is tracked separately as `dual_update_events`: a global update of all `L_free` duals costs `L_free` dual-update events; one local dual update in `fully_async_local` costs one.

## Diagnostics

The dynamics harness holds model parameters fixed and records sparse checkpoints rather than computing an expensive full BP comparison at every local event.

`trace.csv` contains:

- local layer-update work and sweep-equivalent time;
- dual update work;
- total residual, dual, and activity norms;
- total weight-gradient cosine to ordinary BP;
- distance to the long synchronous reference terminal activity state;
- maximum observed staleness;
- finite-state and magnitude indicators.

`layer_trace.csv` contains per-free-layer activity, residual, and dual norms plus credit fraction. `gradient_trace.csv` contains per-weight-layer gradient cosine to BP. `events.csv` makes the realized schedule and stale-read versions auditable.

Nonfinite activity or dual state raises `FloatingPointError`; it is never silently reported as a successful trajectory. A configurable finite-magnitude divergence threshold marks an async run `diverged` and terminates it early.

## Credit front

The long synchronous reference run supplies the per-layer credit scale:

```text
credit_fraction_i(t) = ||lambda_i(t)|| / (||lambda_i_ref|| + eps)
```

For front calculations the natural input-to-output layer order is reversed so **distance 0 is the output-adjacent hidden-edge constraint**, and increasing distance moves backward toward the input.

For each threshold, `front.csv` records:

- `R_contiguous_layers`: number of consecutive layers reached starting at the output side;
- `R_any_layers`: deepest threshold crossing even if there are gaps;
- `dispersion_gap_layers = R_any_layers - R_contiguous_layers`;
- number, mean distance, and standard deviation of all threshold-active layers.

This intentionally distinguishes a compact propagating front from a smeared async pattern with isolated deeper crossings.

The harness also attempts a log-log fit `R(t) ~ t^beta` using `R_contiguous_layers`, but only when there are at least three positive points and at least a factor-of-two front dynamic range. Otherwise `summary.json` stores `beta: null` and a reason. Raw trajectories remain authoritative.

## Running

CPU smoke experiment (synthetic data, no download):

```bash
uv run python scripts/run_async_dynamics.py --config configs/async_smoke.yaml
```

In an environment where the repository's `uv` Python cannot be downloaded but JAX is already installed, the script can also be run directly from the checkout with an appropriate supported Python:

```bash
PYTHONPATH=. python scripts/run_async_dynamics.py --config configs/async_smoke.yaml
```

A larger synthetic depth-32 matrix is provided:

```bash
uv run python scripts/run_async_dynamics.py --config configs/async_dynamics.yaml
```

Important knobs are model depth/width/activation, deterministic data/model seed, `state_lr`, `rho`, `alpha`, `inner_steps`, work budget, scheduler seed, block size, rate heterogeneity, `tau_max`, diagnostic interval, front thresholds, and reference budget. The harness is not hard-coded to one depth.

## Outputs

Each run writes:

```text
results/async_dynamics/<run>/
  config.json
  summary.json
  trace.csv
  layer_trace.csv
  gradient_trace.csv
  front.csv
  events.csv
  [residual_norm.png]
  [credit_front.png]
```

`config.json` includes resolved work-accounting values. `summary.json` reports per-mode final work, stability status, gradient alignment, state/dual distance from the same-work synchronous control, realized layer rates, and conservative propagation-exponent fits.

## Interpretation / BEAM gate

A later BEAM/actor implementation is scientifically motivated only if these scheduling changes reveal reproducible structure that cannot be dismissed as a trivial numerical failure: for example, a robust front under coordinate/block asynchrony, a systematic slowdown or dispersion law with increasing `tau_max`, stable heterogeneous-rate regimes, or qualitatively distinct but coherent behavior under `fully_async_local`.

If modest asynchrony immediately destroys the credit front across seeds and sensible step sizes, or if all async variants collapse onto the synchronous trajectory after fair work normalization, an actor implementation has a much weaker scientific rationale. This experiment alone does not validate BEAM or actors.
