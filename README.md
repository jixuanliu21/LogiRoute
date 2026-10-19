# LogiRoute — Multi-Vehicle Delivery Route Optimisation

A runnable prototype for **Capacitated Vehicle Routing (CVRP)** with an interactive Streamlit interface, CSV scenario input, three routing methods, feasibility validation, and downloadable vehicle assignments.

## Features

- Upload orders as CSV (`id,x,y,demand`; first row is depot with demand 0)
- Configure vehicle count and per-vehicle capacity
- Compare **capacity-aware nearest-neighbour**, **nearest-neighbour + intra-route 2-opt**, and **Google OR-Tools CVRP**
- Visualise routes, per-vehicle loads, total distance and runtime
- **Optional Ningbo OpenStreetMap road-network mode:** download the drivable road graph, snap stops to streets, compute directed shortest-road paths, and map resulting routes
- **Optional PyTorch integration:** train a small trip-duration regression model (synthetic examples by default, or user-uploaded observed trips) and use predicted trip times as optimisation costs
- Download optimised vehicle assignments as CSV
- Automated tests for feasibility, distance, input validation, and local-search non-worsening

## Quick start

Requires Python 3.11–3.13 (recommended). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Default example orders load automatically.

Run tests:

```bash
python -m pytest -q
```

## Data format

| id | x | y | demand |
|---|---:|---:|---:|
| DEPOT | 50 | 50 | 0 |
| C01 | 12 | 40 | 3 |

All coordinates are **planar synthetic units**. Distances are Euclidean, **not** actual road kilometres or travel times. All vehicles have identical capacities and return to the single depot. Each customer must be served exactly once; splitting orders is not supported.

## Algorithms

- **Nearest neighbour:** greedily select the closest unvisited customer whose demand fits the current vehicle's remaining capacity. It can fail to find a feasible allocation even if one exists.
- **2-opt:** improve the visit order *within each assigned vehicle route* by reversing subsequences; it does not reassign customers across vehicles.
- **OR-Tools:** use capacity constraints and guided local search with a configurable solve time limit. It seeks good feasible solutions; global optimality is not guaranteed.

Compare distance and runtime on identical scenarios. All returned solutions are validated against customer coverage and vehicle capacity. OR-Tools internally scales distances to integers; displayed metrics are recomputed from the original floating-point distance matrix.

## Limitations and next steps

This is a **prototype**, not a production routing platform. The optional OSM mode uses real road-network geometry and lengths, but is dependent on live OSM download/reachability and has not been validated against observed driving times. The optional PyTorch model is **not an independently validated real-world traffic forecast**: built-in training records are synthetic and its reported MAE is in-sample. No delivery time windows, time-dependent travel times, multi-depot operations, or live dispatch integration. Further work includes real trip observations, held-out time-series evaluation, realistic benchmarks, and deployment.

## Structure

```text
app.py                Streamlit dashboard
src/routing.py        Heuristics, 2-opt, OR-Tools and feasibility checks
src/io_utils.py       CSV validation
data/sample_orders.csv  Reproducible example
tests/               Automated tests
src/road_network.py  Optional OSMnx street network routing
src/traffic_ml.py    Optional PyTorch travel-duration model
data/sample_geo_orders.csv   Ningbo-area illustrative stop coordinates
requirements-advanced.txt  Extra OSMnx and PyTorch dependencies
```

## Optional advanced modules: OSM roads + PyTorch

Install the optional stack using Python 3.11–3.13:

```bash
python -m pip install -r requirements-advanced.txt
```

In Streamlit, switch **Distance source** to **OpenStreetMap roads**. The included `data/sample_geo_orders.csv` uses geographic lat/lon stops near Ningbo (synthetic stop positions), and the app obtains **real drivable road geometry and road lengths** through [OSMnx](https://osmnx.readthedocs.io/). Internet is required for the initial graph download; OSM coverage and one-way restrictions can affect reachability. The route objective uses **shortest road length**, not live road traffic or guaranteed road travel time. The map follows roads, not straight segments.

Enable the **Experimental PyTorch trip-time model** under the expander to train a small MLP with `distance_km`, `departure_hour` and `observed_minutes`, then use its estimated per-pair trip durations as the VRP cost matrix. The default data are **synthetically generated**, explicitly a demonstration of integration only. You may upload actual recorded trip observations. The displayed training MAE is *in-sample*, not an independently validated model score. No real traffic forecast performance is claimed. Time-dependent routing is not yet modelled: all legs use one assumed departure hour.

Do not present simulated-traffic performance as a real-world empirical improvement. For substantive traffic forecasting, obtain geographically aligned observed travel-time data and create chronological train/validation/test partitions before asserting generalisation.


## V2 evaluation tools

- `python -m experiments.benchmark` benchmarks three solvers on the **same five seeded synthetic scenarios**, recording feasibility, total planar distance, runtime and vehicles used. It saves `experiments/benchmark_results.csv`. Nearest-neighbour can legitimately fail on some feasible instances; failures are reported, not silently dropped.
- `src.traffic_ml.evaluate_duration_model()` reports train/validation/test MAE versus a training-only constant-minutes-per-km baseline on a seeded random split. **This split is intended only for the built-in IID synthetic demonstration**; real chronological trip records warrant temporal validation.
- The dashboard's displayed training MAE remains an in-sample diagnostic; use the separate function for test metrics. No real traffic accuracy or improvement is claimed from synthetic data.

**Important evaluation note:** Compare road distance with road distance, and predicted trip duration with predicted trip duration. OR-Tools solutions are limited by the user-specified search budget, not certified globally optimal. Different numbers of active vehicles may carry different operational fixed costs not included in the objective.


## V3: In-app evaluation dashboard

The Streamlit app now includes **Multi-scenario benchmark**. Click **Run 5-scenario benchmark** to compare solvers on five reproducible *synthetic planar* CVRP instances, review mean route distance, runtime and feasible rate, and download a CSV. These benchmark numbers are **not** Ningbo street-network distances. Methods can use different vehicle counts; mean route distance does not include a fixed cost per vehicle.

The optional PyTorch module now reports **training, validation and held-out test MAE** against a constant-speed (minutes/km) baseline. This split is random and suitable for the built-in IID *synthetic* demonstration; uploaded time-ordered real-world trips require chronological holdout evaluation before external forecasting claims. Training the PyTorch model uses training observations only; the test partition is reserved for evaluation.

The route panel includes **Solution feasibility checks** (customer coverage, fleet size, capacity). Benchmark results are shown inside the app after running, or loaded from `experiments/benchmark_results.csv` if that file exists.
