import numpy as np
import pytest
from src.routing import validate, heuristic, distance_matrix


def test_route_rejects_duplicate_customer():
    with pytest.raises(ValueError, match='exactly once'):
        validate([[0,1,1,0]], [0,1,1], 5, 1)


def test_route_rejects_overcapacity():
    with pytest.raises(ValueError, match='Capacity'):
        validate([[0,1,2,0]], [0,4,4], 5, 1)


def test_two_opt_never_worsens_on_random_symmetric_costs():
    rng=np.random.default_rng(9)
    d=distance_matrix(rng.uniform(size=(12,2)))
    a=heuristic(d,[0]+[1]*11,12,1,False)
    b=heuristic(d,[0]+[1]*11,12,1,True)
    assert b.total_distance <= a.total_distance+1e-9


def test_ml_heldout_eval_if_available():
    pytest.importorskip('torch')
    pytest.importorskip('sklearn')
    from src.traffic_ml import evaluate_duration_model,simulated_observations
    _, metrics = evaluate_duration_model(simulated_observations(n=120), epochs=10)
    assert metrics['sizes']=={'train':84,'validation':18,'test':18}
    assert metrics['test']['model_mae']>=0
