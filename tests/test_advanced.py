import numpy as np
import pandas as pd
import pytest
from src.traffic_ml import make_features, simulated_observations

def test_synthetic_data_is_well_formed():
    sample = simulated_observations(n=25)
    assert list(sample.columns) == ['distance_km','departure_hour','observed_minutes']
    assert len(sample) == 25
    assert (sample.observed_minutes > 0).all()

def test_feature_shape():
    assert make_features([1., 2.], [8, 17]).shape == (2, 4)

def test_training_if_torch_installed():
    pytest.importorskip('torch')
    from src.traffic_ml import train_duration_model, predict_duration_matrix
    m, mae = train_duration_model(simulated_observations(n=80), epochs=15)
    mat = predict_duration_matrix(m, np.array([[0.,2.],[2.,0.]]), 8)
    assert mat[0,0] == 0 and mat[0,1] > 0 and np.isfinite(mae)


def test_two_opt_directed_road_cost_never_worsens():
    from src.routing import two_opt, route_distance
    costs = np.array([[0,2,8,9],[7,0,2,8],[8,9,0,2],[2,8,9,0]], dtype=float)
    route = [0,1,2,3,0]
    updated = two_opt(route, costs)
    assert route_distance(updated, costs) <= route_distance(route, costs) + 1e-9
