import pandas as pd
import pytest
from src.reporting import summarize_benchmark


def test_benchmark_summary_excludes_failures_from_mean():
    raw = pd.DataFrame([
        {'seed': 1, 'method': 'A', 'feasible': True, 'distance': 10, 'runtime_s': 0.5, 'vehicles_used': 2},
        {'seed': 2, 'method': 'A', 'feasible': False, 'distance': 999, 'runtime_s': 99, 'vehicles_used': 0},
    ])
    result = summarize_benchmark(raw).iloc[0]
    assert result['feasible_rate'] == .5
    assert result['mean_distance'] == 10
    assert result['mean_runtime_s'] == .5


def test_empty_benchmark_rejected():
    with pytest.raises(ValueError, match='missing columns|empty'):
        summarize_benchmark(pd.DataFrame())
