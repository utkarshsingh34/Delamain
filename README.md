# Delamain

**City-scale autonomous fleet orchestration powered by demand forecasting and reinforcement learning.**

Delamain ingests real NYC rideshare trip data, trains a zone-level demand forecaster and a
PPO reinforcement learning agent, and orchestrates a simulated fleet of 200 vehicles across
69 Manhattan zones. It accepts natural language commands from operators ("Yankees game ends
at 4pm, pre-position the Stadium area at 3:30") and responds with structured repositioning
decisions, expected impact, and tradeoffs. The Fleet API layer mirrors the Tesla Fleet API
schema, so the path from simulation to a real autonomous fleet is a configuration change,
not a rewrite.

> **Project Status:** Delamain is under active development following a 7-phase build plan.
> The architectural design is complete and documented in [CLAUDE.md](CLAUDE.md). Current progress:
>
> | Phase | Scope | Status |
> |-------|-------|--------|
> | 1 | NYC TLC data pipeline | Planned |
> | 2 | Gymnasium simulation environment | Planned |
> | 3 | LightGBM demand forecaster | Planned |
> | 4 | Reactive baseline + PPO RL agent | Planned |
> | 5 | FleetManager API (FastAPI + Redis) | Planned |
> | 6 | LangGraph orchestrator | Planned |
> | 7 | Streamlit comparison dashboard | Planned |

---

## What Delamain Does

**Demand-Driven Fleet Repositioning** — Instead of dispatching the nearest idle car to each
incoming ride request, Delamain predicts zone-level demand 15 minutes ahead and pre-positions
vehicles accordingly. During morning rush, cars are staged at Penn Station and Midtown before
the surge begins, not after.

**Autonomous Tick Loop** — Every 5 simulated minutes, the PPO agent computes a repositioning
plan across all 69 Manhattan zones using the demand forecast and current fleet state. The LLM
is never invoked for routine operation. Normal ticks cost zero API calls.

**Natural Language Operator Interface** — Operators issue commands in plain English via a chat
panel. The LangGraph orchestrator parses the command, resolves zone references, validates
feasibility, executes the repositioning, and explains the tradeoffs — "this will temporarily
reduce Midtown East coverage."

**Side-by-Side Benchmark** — The dashboard runs a reactive baseline (nearest-car FIFO dispatch)
and the Delamain agent simultaneously on the same demand stream. The divergence in wait times
and utilization is the demo.

**Real-to-Production Integration Path** — Every command issued in simulation goes through the
same `FleetManager` Protocol that would talk to Tesla's real Fleet API. Swapping
`SimulatedFleetAdapter` for `RealFleetAdapter` requires setting one environment variable.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   NYC TLC Trip Data (2023)                   │
│          Parquet, zone IDs only, 69 Manhattan zones          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                LightGBM Demand Forecaster                    │
│  Single global model, zone_id as categorical feature         │
│  Hour/weekday cyclic encoding, rolling demand windows        │
│  Output: predicted trip counts per zone, next 15 min         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              PPO Reinforcement Learning Agent                │
│  Obs: per-zone vehicle count + demand forecast +             │
│  supply-demand gap (69 × 3) + 4 cyclic time features         │
│  Act: 69-dim continuous Box → softmax → integer target       │
│  allocation per zone                                         │
│  Training: CPU-only, 8× SubprocVecEnv, ~5–15 min / 1M steps │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  LangGraph Orchestrator                      │
│  Ticking agent  → autonomous 5-min repositioning loop        │
│  Reactive agent → operator NL commands + anomaly response    │
│  Tools: reposition_vehicles, get_fleet_state, get_forecast,  │
│         dispatch_vehicle, get_zone_info                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FleetManager API (FastAPI + Redis)              │
│  Protocol: SimulatedFleetAdapter / RealFleetAdapter          │
│  REST + WebSocket + SSE streaming                            │
│  Tesla Fleet API schema — swap adapter via env variable      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                        │
│  Left: reactive baseline  |  Right: Delamain                 │
│  PyDeck Manhattan map, real-time KPI cards, operator chat    │
└──────────────────────────────────────────────────────────────┘
```

---

## The Simulation Environment

Delamain's Gymnasium environment (`FleetRepositioningEnv`) is the core against which both
the reactive baseline and the RL agent are evaluated. It runs against real historical
demand from NYC TLC trip data rather than synthetic patterns.

**Observation space (211 dims, `gymnasium.spaces.Box`):**

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Vehicle count | 69 | Normalized idle vehicles per zone |
| Demand forecast | 69 | Normalized predicted trips per zone (next 15 min) |
| Supply-demand gap | 69 | Normalized difference between the two |
| Time features | 4 | sin/cos(hour), sin/cos(day_of_week) |

All values normalized to [0, 1]. Normalization is mandatory for PPO convergence — unnormalized
observations cause gradient instability.

**Action space (69 dims, `gymnasium.spaces.Box`):** Raw continuous output in [-1, 1]. In
`step()`: apply softmax to get fractional allocations → multiply by idle vehicle count →
round to integer target per zone → compute repositioning moves as deltas.

`Discrete` and `MultiDiscrete` action spaces were ruled out: `Discrete(69^N)` causes
combinatorial explosion, and `MaskablePPO` has documented numerical instability above
~1,400 actions. The continuous space with softmax normalization is the proven approach for
fleet repositioning at this scale.

---

## Demand Forecasting

The forecaster predicts zone-level trip demand 15 minutes ahead and feeds those predictions
into both the PPO observation space and the orchestrator's planning tools.

**Single global model architecture:** One LightGBM model trained across all 69 zones, with
`zone_id` as a native categorical feature. Not 69 per-zone models — a single model shares
statistical strength across zones with similar demand profiles and generalizes better to
low-activity zones. LightGBM does not support multi-output regression, so the data is
stacked: each row is `(timestamp, zone_id, features) → demand_count`.

**Feature engineering:** Hour-of-day and day-of-week cyclical sin/cos encodings, rolling
demand averages at 15-min, 1-hour, and 24-hour lags, zone metadata (borough, service zone),
and optionally current weather conditions from the Open-Meteo API (zero-cost, no API key).

**Temporal cross-validation:** The model is trained with walk-forward cross-validation —
never using future data as training input. Evaluation metrics are reported per-zone (MAE,
RMSE) and in aggregate.

---

## Orchestrator Design

The LangGraph orchestrator runs two agents with distinct activation patterns:

**Ticking agent** runs every 5 simulated minutes on an autonomous loop. It calls the
forecaster, invokes the RL agent's repositioning recommendation, and executes the plan
via the FleetManager API. No LLM reasoning is involved — it calls `optimizer.plan()` and
dispatches the result. Zero API cost for normal operation.

**Reactive agent** activates on operator commands and demand anomalies (forecast errors
exceeding a configurable threshold). It receives the operator's natural language command,
resolves zone references and time specifications, checks current fleet state and the demand
forecast, generates a repositioning plan with explicit tradeoff explanation, and executes
via the same FleetManager tools.

Every decision — autonomous or operator-directed — is logged with: timestamp, trigger type,
zones affected, vehicles moved, forecast that motivated it, and expected impact. This audit
trail powers the dashboard's decision replay and explanation features.

---

## Operator Command Pipeline

```
Operator Input
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Parse      │────▶│   Plan      │────▶│  Validate   │────▶│  Execute +  │
│   Command    │     │   Moves     │     │  Feasibility│     │  Explain    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
 Resolve zones,      Query forecaster     Check vehicle       Dispatch via
 extract intent,     + RL optimizer,      availability,       FleetManager,
 parse time specs    compute target       flag tradeoffs      log decision
                     allocation
```

The orchestrator resolves zone references by name and borough before issuing any commands.
If a zone reference is ambiguous ("the bridge area", "near the park"), it flags for
clarification rather than guessing. If the requested repositioning would critically undersupply
another zone, it surfaces the tradeoff explicitly in its response before executing.

---

## Fleet API

The `FleetManager` Protocol defines a backend-agnostic interface for all fleet operations:

```python
class FleetManager(Protocol):
    def get_fleet_state(self) -> FleetState: ...
    def reposition_vehicles(self, plan: RepositioningPlan) -> RepositioningResult: ...
    def dispatch_vehicle(self, vehicle_id: str, pickup: ZoneID) -> DispatchResult: ...
    def get_vehicle_status(self, vehicle_id: str) -> VehicleStatus: ...
```

`SimulatedFleetAdapter` implements this against the Gymnasium simulation. `RealFleetAdapter`
implements the same interface against the Tesla Fleet API — using the same auth flow, Pydantic
schemas, and endpoint structure from Tesla's public Fleet API specification.

The LangGraph orchestrator calls these methods via HTTP. It never imports simulation internals.
Setting `USE_REAL_TESLA_API=true` in the environment is the complete migration path.

---

## Observability

Delamain uses **structlog** for structured logging throughout every module. All output
produces key-value pairs that are machine-parseable (JSON) or human-readable (pretty-print),
controlled by a single config variable.

Every pipeline run (simulation tick, operator command, forecaster inference) is tagged with
a **trace ID** — a 12-character hex identifier that propagates through every downstream log
event, enabling full reconstruction of any decision from log output alone.

Key things Delamain logs: repositioning decisions with full motivation context, LLM call
latencies for operator commands, forecast accuracy deltas, zone supply-demand gaps at each
tick, and operator command parse results.

```bash
# Normal operation
streamlit run interface/app.py

# Filter logs by trace ID (e.g., to replay a specific operator command)
cat delamain.log | jq 'select(.trace_id == "a1b2c3d4e5f6")'

# Watch live repositioning decisions
tail -f delamain.log | jq 'select(.event == "repositioning_decision")'
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RL Training | stable-baselines3 PPO | Fleet repositioning agent, CPU-optimized |
| Simulation | Gymnasium + SimPy | Environment and discrete-event simulation |
| Demand Forecasting | LightGBM | Zone-level trip demand prediction |
| Orchestration | LangGraph | Ticking agent + reactive operator interface |
| LLM | Claude via Anthropic API | Operator NL commands and anomaly response only |
| Fleet API | FastAPI + Redis | REST + WebSocket server, pub/sub state bus |
| Data Processing | Polars + PyArrow | Fast Parquet ingestion (5–10× over Pandas) |
| Geospatial | GeoPandas + Shapely | Zone polygon handling, EPSG:2263 → EPSG:4326 reprojection |
| Dashboard | Streamlit + PyDeck | Side-by-side comparison map and operator chat |
| Logging | structlog | Structured decision audit trail with trace ID propagation |

---

## Project Structure

```
delamain/
├── CLAUDE.md                    # Full build specification and technical notes
├── README.md
├── pyproject.toml
├── Makefile
├── docker-compose.yml           # Three-service architecture
├── .env.example
│
├── data/
│   ├── raw/                     # Downloaded Parquet files (gitignored)
│   ├── processed/               # Cleaned Parquet (gitignored)
│   ├── zones/                   # Taxi zone shapefile + lookup CSV
│   └── scripts/
│       ├── download.py          # Pull NYC TLC trip data + zone files
│       ├── preprocess.py        # Filter to Manhattan, output Parquet
│       └── eda.py               # Validate data quality before building
│
├── simulation/
│   ├── env.py                   # FleetRepositioningEnv (Gymnasium)
│   ├── demand.py                # HistoricalDemandModel + SyntheticDemandModel
│   ├── city.py                  # Zone graph, travel times, zone metadata
│   ├── metrics.py               # Wait time, utilization, deadhead tracking
│   └── runner.py                # SimPy simulation process entry point
│
├── forecasting/
│   ├── features.py              # Feature engineering pipeline
│   ├── model.py                 # LightGBM global model wrapper
│   ├── train.py                 # Training with temporal cross-validation
│   └── evaluate.py              # Per-zone MAE/RMSE + aggregate metrics
│
├── optimizer/
│   ├── baseline.py              # ReactiveDispatcher (nearest-car FIFO)
│   ├── heuristic.py             # ForecastPrePositioner (greedy)
│   └── rl_agent.py              # PPO agent (stable-baselines3, CPU)
│
├── api/
│   ├── protocol.py              # FleetManager Protocol + dataclasses
│   ├── schemas.py               # Pydantic v2 models (Tesla-compatible)
│   ├── sim_adapter.py           # SimulatedFleetAdapter(FleetManager)
│   ├── tesla_adapter.py         # RealFleetAdapter(FleetManager) — future
│   ├── server.py                # FastAPI server (REST + WebSocket + SSE)
│   └── redis_bus.py             # Redis pub/sub helpers
│
├── orchestrator/
│   ├── agent.py                 # LangGraph reactive agent (operator commands)
│   ├── ticker.py                # LangGraph ticking agent (5-min loop)
│   ├── tools.py                 # Tool definitions for both agents
│   ├── planner.py               # Integrates forecaster + optimizer
│   └── prompts.py               # System prompts
│
├── interface/
│   ├── app.py                   # Streamlit dashboard entry point
│   ├── components/
│   │   ├── map_view.py          # PyDeck map: zone heatmap + vehicle dots
│   │   ├── metrics_panel.py     # Real-time KPI cards
│   │   ├── chat.py              # Operator NL interface
│   │   └── comparison.py        # Side-by-side baseline vs Delamain
│   └── styles.py
│
├── eval/
│   ├── run_baseline.py
│   ├── run_heuristic.py
│   ├── run_rl.py
│   └── compare.py               # Benchmark charts (matplotlib)
│
└── tests/
    ├── test_data.py
    ├── test_env.py
    ├── test_forecasting.py
    ├── test_optimizer.py
    └── test_api.py
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Redis (for the API state bus)
- Anthropic API key (for operator NL commands only — not required to run the simulation)

```bash
docker compose up redis   # or: brew install redis && redis-server
```

### Installation

```bash
git clone https://github.com/your-username/delamain.git
cd delamain
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
```

### Download and Prepare Data

```bash
python data/scripts/download.py     # ~5 min, ~100 MB of 2023 TLC data
python data/scripts/preprocess.py   # filter to Manhattan, output Parquet
python data/scripts/eda.py          # validate before continuing
```

Do not proceed past this step until `eda.py` exits cleanly.

### Train Models

```bash
python forecasting/train.py    # ~5 min — outputs models/forecaster.pkl
python optimizer/rl_agent.py   # ~15 min (1M steps, CPU) — outputs models/rl_agent.zip
```

### Run the Stack

```bash
uvicorn api.server:app             # Fleet API on :8000
streamlit run interface/app.py     # Dashboard on :8501
```

Open the dashboard, press **Play**, and watch the comparison unfold.

---

## Configuration

All configuration lives in `pyproject.toml` and `.env`. Key settings:

```bash
# .env

ANTHROPIC_API_KEY=sk-ant-...

# Simulation
SIM_FLEET_SIZE=200
SIM_SPEED_MULTIPLIER=10       # 10× = 1 sim minute per 6 wall seconds
SIM_EPISODE_START_HOUR=6      # 6:00 AM
SIM_EPISODE_HOURS=18          # run until midnight
SIM_STEP_MINUTES=5            # decision frequency

# API
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379

# Tesla (future)
USE_REAL_TESLA_API=false
TESLA_FLEET_API_BASE=https://fleet-api.prd.na.vn.cloud.tesla.com
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required only for operator NL commands. Never hardcoded. |
| `USE_REAL_TESLA_API` | Set `true` to swap `SimulatedFleetAdapter` for `RealFleetAdapter`. |

---

## Design Decisions

**Why CPU-only PPO training?** stable-baselines3 PPO with an MLP policy runs 3–10× faster
on CPU than GPU for non-image environments. The official documentation states this explicitly.
GPU utilization never exceeds 5% for MLP policies — the CPU↔GPU data transfer overhead
exceeds any compute benefit. Set `device="cpu"` and use 8× `SubprocVecEnv` for parallelism.
Training 1M steps takes 5–15 minutes on a modern CPU, and 20–60 minutes if GPU is forced.

**Why a single global LightGBM model instead of one per zone?** A per-zone model cannot
share statistical patterns between zones with similar demand profiles — all Midtown zones
follow nearly identical weekday curves. A single model with `zone_id` as a categorical
feature learns these cross-zone patterns, generalizes better to low-activity zones, and
requires maintaining one artifact instead of 69. LightGBM's native categorical support
handles the zone feature efficiently without one-hot encoding.

**Why continuous action space over discrete?** `Discrete(69^N)` causes combinatorial
explosion at fleet scale. `MaskablePPO` has documented numerical instability above
~1,400 actions. `MultiDiscrete` cannot represent correlated repositioning decisions —
moving a car from zone A to zone B affects the optimal action for every other zone
simultaneously. A 69-dimensional continuous `Box` with softmax normalization is the
proven approach for fleet repositioning (Skordilis et al. 2021, Gammelli et al. 2024).

**Why mirror the Tesla Fleet API?** The architecture argument for this project rests on
the claim that deploying against a real autonomous fleet is tractable. Mirroring the actual
Fleet API schema — same Pydantic models, same auth flow, same endpoint structure — makes
that claim concrete rather than aspirational. `USE_REAL_TESLA_API=true` is the complete
integration path.

**Why LLM only for operator commands?** Routing every 5-minute tick through an LLM would
make the system slow, expensive, and opaque. The RL agent handles routine repositioning
optimally at machine speed. The LLM activates only for the things that require natural
language understanding: parsing operator intent and responding to demand anomalies the
forecast didn't predict. This keeps normal operation deterministic, auditable, and free.

**Why 2023 data only?** NYC TLC added a 20th column (`cbd_congestion_fee`) in January 2025
for the MTA Congestion Relief Zone toll. Using 2025 data causes schema mismatch errors
throughout the pipeline. 2023 provides a full year of data with a stable 19-column schema.

---

## Data Notes

Delamain uses NYC TLC Yellow Taxi trip data from 2023. Key facts every contributor
should know before touching the data pipeline:

- **Parquet only** — CSV was discontinued in May 2022
- **No GPS coordinates** — raw lat/lon was removed in July 2016 for privacy; only integer zone IDs remain (`PULocationID`, `DOLocationID`, values 1–263)
- **263 total zones** across all boroughs; Delamain filters to the 69 Manhattan zones
- **~2.5–4M trips/month**, ~40–55 MB per file
- **Zone shapefiles** are in EPSG:2263 (NAD83 / NY Long Island feet) and must be reprojected to EPSG:4326 before use in PyDeck

---

## Limitations

Delamain optimizes repositioning within the 69-zone Manhattan grid using historical demand
patterns. It does not handle cross-borough demand, airport routing, or demand spikes with
no historical precedent in the training data.

The RL agent learns policies against a SimPy simulation of historical demand — not against
a real fleet with mechanical failures, traffic incidents, or passenger no-shows. Performance
in a real deployment would require retraining against live telemetry.

The natural language operator interface resolves zone references by name and borough.
Ambiguous references that don't correspond to a known zone name are flagged for
clarification rather than guessed.
