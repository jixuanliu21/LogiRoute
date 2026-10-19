"""Reproducible capacity-constrained VRP benchmark on synthetic planar instances.

Run: python -m experiments.benchmark
Writes experiments/benchmark_results.csv. No claims about real traffic.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.routing import distance_matrix, heuristic, ortools_solve, validate


def run_benchmark(seeds=(11, 22, 33, 44, 55), n_customers=18, vehicles=4, capacity=12):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 100, size=(n_customers + 1, 2))
        demands = [0] + rng.integers(1, 4, size=n_customers).tolist()
        distances = distance_matrix(pts)
        for label, fn in [
            ('Nearest neighbour', lambda:heuristic(distances,demands,capacity,vehicles,False)),
            ('Nearest neighbour + 2-opt', lambda:heuristic(distances,demands,capacity,vehicles,True)),
            ('OR-Tools CVRP', lambda:ortools_solve(distances,demands,capacity,vehicles,1))
        ]:
            try:
                sol = fn()
                validate(sol.routes, demands, capacity, vehicles)
                rows.append({'seed':seed,'customers':n_customers,'method':label,
                             'feasible':True,'distance':sol.total_distance,
                             'runtime_s':sol.runtime_seconds,'vehicles_used':len(sol.routes),'error':''})
            except (ValueError, ImportError) as exc:
                rows.append({'seed':seed,'customers':n_customers,'method':label,
                             'feasible':False,'distance':float('nan'),
                             'runtime_s':float('nan'),'vehicles_used':0,'error':str(exc)})
    return pd.DataFrame(rows)

if __name__=='__main__':
    result = run_benchmark()
    out = Path(__file__).with_name('benchmark_results.csv')
    result.to_csv(out,index=False)
    print(result.groupby('method').agg(feasible_rate=('feasible','mean'),
        mean_distance=('distance','mean'),mean_runtime_s=('runtime_s','mean')).round(3).to_string())
    print(f'\nSaved {out}')
