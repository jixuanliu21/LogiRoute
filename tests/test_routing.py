import numpy as np
import pandas as pd
import pytest
from src.routing import distance_matrix, heuristic, ortools_solve, route_distance, two_opt, validate
from src.io_utils import parse_orders

@pytest.fixture
def sample():
    points = np.array([[0,0],[1,0],[2,0],[0,2],[2,2]], dtype=float)
    return distance_matrix(points), [0,2,2,2,2]

def test_distance(sample):
    d,_ = sample
    assert d[0,1] == 1
    assert np.allclose(d,d.T)

def test_heuristic_feasibility(sample):
    d,demands=sample
    for improve in (False,True):
        result=heuristic(d,demands,4,2,improve)
        validate(result.routes,demands,4,2)

def test_two_opt_nonworsening(sample):
    d,_=sample
    route=[0,1,3,2,4,0]
    assert route_distance(two_opt(route,d),d) <= route_distance(route,d) + 1e-8

def test_ortools_feasibility(sample):
    pytest.importorskip('ortools')
    d,demands=sample
    result=ortools_solve(d,demands,4,2,1)
    validate(result.routes,demands,4,2)

def test_insufficient_capacity(sample):
    d,demands=sample
    with pytest.raises(ValueError):
        heuristic(d,demands,3,2)

def test_csv_validation():
    df=pd.DataFrame({'id':['DEPOT','C1'],'x':[0,1],'y':[0,1],'demand':[0,2]})
    assert len(parse_orders(df)) == 2
    with pytest.raises(ValueError):
        parse_orders(df.assign(demand=[1,2]))
