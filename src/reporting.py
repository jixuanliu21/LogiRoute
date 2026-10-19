"""Small pure-data utilities for benchmark reporting."""
import numpy as np
import pandas as pd


def summarize_benchmark(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only feasible runs; report sample counts so failures aren't hidden."""
    needed = {'seed', 'method', 'feasible', 'distance', 'runtime_s', 'vehicles_used'}
    if not needed.issubset(results.columns):
        raise ValueError(f'Benchmark results missing columns: {sorted(needed - set(results.columns))}')
    frame = results.copy()
    if frame.empty:
        raise ValueError('Benchmark results are empty')
    frame['feasible'] = frame['feasible'].map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).lower() == 'true')
    frame.loc[~frame['feasible'], ['distance', 'runtime_s', 'vehicles_used']] = np.nan
    return frame.groupby('method', as_index=False).agg(
        scenarios=('seed', 'size'),
        feasible_count=('feasible', 'sum'),
        feasible_rate=('feasible', 'mean'),
        mean_distance=('distance', 'mean'),
        mean_runtime_s=('runtime_s', 'mean'),
        mean_vehicles_used=('vehicles_used', 'mean'),
    )
