"""Road-network routing using public OpenStreetMap data via OSMnx.
Requires internet on the first graph download. Coordinates: longitude, latitude.
"""
import numpy as np


def road_distance_matrix(orders, radius_m=3500):
    """Return directed shortest-road distances (km), and road polylines (lat, lon).

    The graph uses OSM road *length* (not live traffic or OSRM travel time).
    Raises clear errors for unsnappable/unreachable locations.
    """
    import osmnx as ox
    import networkx as nx

    coords = orders[['lat', 'lon']].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not ((-90 <= coords[:, 0]) & (coords[:, 0] <= 90)).all() or not ((-180 <= coords[:, 1]) & (coords[:, 1] <= 180)).all():
        raise ValueError('Latitude/longitude out of range')
    if len(coords) > 31:
        raise ValueError('Road mode limited to 30 customers to limit OSM downloads')
    # Increase radius if required to include all submitted stops.
    center = coords.mean(axis=0)
    approx_radius = max(
        np.hypot((coords[:, 0]-center[0])*111_200,
                 (coords[:, 1]-center[1])*111_200*np.cos(np.radians(center[0])))
    )
    graph = ox.graph_from_point(tuple(center), dist=int(max(radius_m, approx_radius + 1200)), network_type='drive')
    graph = ox.distance.add_edge_lengths(graph) if not all('length' in d for _, _, d in graph.edges(data=True)) else graph
    nodes = ox.distance.nearest_nodes(graph, X=coords[:, 1].tolist(), Y=coords[:, 0].tolist())
    n = len(nodes)
    mat = np.zeros((n, n), dtype=float)
    polylines = {}
    for i, start in enumerate(nodes):
        lengths, paths = nx.single_source_dijkstra(graph, start, weight='length')
        for j, end in enumerate(nodes):
            if end not in lengths:
                raise ValueError(f'No drivable route between {orders.iloc[i]["id"]} and {orders.iloc[j]["id"]}')
            mat[i, j] = lengths[end] / 1000.0
            if i != j:
                poly = [(graph.nodes[k]['y'], graph.nodes[k]['x']) for k in paths[end]]
                polylines[(i, j)] = poly
    return mat, polylines
