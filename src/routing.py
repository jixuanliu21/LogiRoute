"""Capacitated vehicle routing: constructive heuristic, 2-opt, and OR-Tools."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass
class Solution:
    routes: list[list[int]]
    total_distance: float
    runtime_seconds: float
    method: str


def distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Coordinates must be a finite Nx2 array")
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def route_distance(route: Sequence[int], distances: np.ndarray) -> float:
    return sum(float(distances[a, b]) for a, b in zip(route[:-1], route[1:]))


def validate(routes: list[list[int]], demands: Sequence[int], capacity: int, vehicles: int) -> None:
    n = len(demands)
    if len(routes) > vehicles:
        raise ValueError("Too many vehicles")
    visited = []
    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            raise ValueError("Every route must start and end at depot 0")
        customers = route[1:-1]
        if any(i <= 0 or i >= n for i in customers):
            raise ValueError("Invalid customer index")
        if sum(demands[i] for i in customers) > capacity:
            raise ValueError("Capacity exceeded")
        visited.extend(customers)
    if sorted(visited) != list(range(1, n)):
        raise ValueError("Every customer must be served exactly once")


def two_opt(route: list[int], distances: np.ndarray) -> list[int]:
    """Improve a closed route without changing its assigned customers."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                # Check the *whole* tour: road networks can be directed, so
                # reversing a segment also changes all its internal arc costs.
                candidate = best[:i] + list(reversed(best[i:j+1])) + best[j+1:]
                if route_distance(candidate, distances) < route_distance(best, distances) - 1e-10:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def heuristic(distances: np.ndarray, demands: Sequence[int], capacity: int, vehicles: int, improve: bool = True) -> Solution:
    """Capacity-aware nearest feasible neighbour, followed by optional intra-route 2-opt."""
    start = time.perf_counter()
    n = len(demands)
    if capacity <= 0 or vehicles <= 0 or len(distances) != n or any(d < 0 for d in demands):
        raise ValueError("Invalid problem dimensions or capacity")
    if any(d > capacity for d in demands[1:]) or sum(demands[1:]) > capacity * vehicles:
        raise ValueError("Demand exceeds fleet capacity")
    remaining = set(range(1, n))
    routes = []
    for _ in range(vehicles):
        route = [0]
        load = 0
        current = 0
        while True:
            feasible = [i for i in remaining if load + demands[i] <= capacity]
            if not feasible:
                break
            nxt = min(feasible, key=lambda i: (distances[current, i], i))
            route.append(nxt)
            remaining.remove(nxt)
            load += demands[nxt]
            current = nxt
        route.append(0)
        if improve:
            route = two_opt(route, distances)
        routes.append(route)
        if not remaining:
            break
    if remaining:
        raise ValueError("Greedy heuristic could not find a feasible assignment; try OR-Tools")
    validate(routes, demands, capacity, vehicles)
    return Solution(routes, sum(route_distance(r, distances) for r in routes), time.perf_counter()-start, "Nearest neighbour + 2-opt" if improve else "Nearest neighbour")


def ortools_solve(distances: np.ndarray, demands: Sequence[int], capacity: int, vehicles: int, time_limit_seconds: int = 5) -> Solution:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    start = time.perf_counter()
    n = len(demands)
    if capacity <= 0 or vehicles <= 0 or n != len(distances) or any(d < 0 for d in demands):
        raise ValueError("Invalid problem dimensions or capacity")
    if any(d > capacity for d in demands[1:]) or sum(demands[1:]) > capacity * vehicles:
        raise ValueError("Demand exceeds fleet capacity")
    manager = pywrapcp.RoutingIndexManager(n, vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    # Integer millimetre-like scale for deterministic integer OR-Tools costs.
    scaled = np.rint(np.asarray(distances) * 1000).astype(np.int64)
    def distance_cb(from_index, to_index):
        return int(scaled[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])
    distance_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_idx)
    def demand_cb(index):
        return int(demands[manager.IndexToNode(index)])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(capacity)] * vehicles, True, "Capacity")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(max(1, int(time_limit_seconds)))
    assignment = routing.SolveWithParameters(params)
    if assignment is None:
        raise ValueError("OR-Tools found no feasible solution within the time limit")
    routes = []
    for vehicle in range(vehicles):
        idx = routing.Start(vehicle)
        route = [0]
        while not routing.IsEnd(idx):
            idx = assignment.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        if len(route) > 2:
            routes.append(route)
    validate(routes, demands, capacity, vehicles)
    return Solution(routes, sum(route_distance(r, distances) for r in routes), time.perf_counter()-start, "OR-Tools CVRP")
