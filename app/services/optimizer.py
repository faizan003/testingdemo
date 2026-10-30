from typing import Dict, Tuple

import networkx as nx

from app.schemas import RouteMetrics
from app.services.grid import NodeId, edge_crosses_land, haversine_km


def apply_weighted_costs(
    graph: nx.Graph,
    node_risk: Dict[NodeId, float],
    w_distance: float,
    w_risk: float,
    w_fuel: float,
    risk_block_threshold: float,
) -> None:
    """
    Baseline uses distance_km only. Optimized uses a nonlinear weather multiplier so
    small risk gradients can prefer lateral detours instead of collapsing to the same path.
    """
    for u, v, data in graph.edges(data=True):
        distance = data["distance_km"]
        risk = (node_risk[u] + node_risk[v]) / 2
        fuel_penalty = distance * (1 + 1.8 * risk)
        # Multiplicative sea-state factor (stronger than additive for tie-breaking)
        weather_multiplier = 1.0 + 2.6 * (risk**1.28)

        if risk >= risk_block_threshold:
            data["optimized_cost"] = 1e9
        else:
            data["optimized_cost"] = (
                w_distance * distance * weather_multiplier
                + w_risk * (risk**1.35) * 140.0
                + w_fuel * fuel_penalty * (1.0 + 0.45 * risk)
            )


def shortest_path(graph: nx.Graph, source: NodeId, destination: NodeId) -> list[NodeId]:
    return nx.shortest_path(graph, source, destination, weight="distance_km")


def optimized_path(graph: nx.Graph, source: NodeId, destination: NodeId, coords: Dict[NodeId, tuple[float, float]]) -> list[NodeId]:
    def heuristic(a: NodeId, b: NodeId) -> float:
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]
        return haversine_km(lat1, lon1, lat2, lon2)

    return nx.astar_path(graph, source, destination, heuristic=heuristic, weight="optimized_cost")


def calculate_metrics(path: list[NodeId], graph: nx.Graph, node_risk: Dict[NodeId, float]) -> RouteMetrics:
    total_distance = 0.0
    edge_risk_sum = 0.0
    edge_count = max(1, len(path) - 1)

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_distance += graph[u][v]["distance_km"]
        edge_risk_sum += (node_risk[u] + node_risk[v]) / 2

    avg_risk = edge_risk_sum / edge_count
    fuel_proxy = total_distance * (1 + 1.6 * avg_risk)
    eta_hours = total_distance / 28.0  # Demo proxy at ~28 km/h

    return RouteMetrics(
        distance_km=round(total_distance, 2),
        fuel_proxy=round(fuel_proxy, 2),
        average_risk=round(avg_risk, 3),
        estimated_eta_hours=round(eta_hours, 2),
    )


def to_coordinate_list(path: list[NodeId], coords: Dict[NodeId, tuple[float, float]]) -> list[Tuple[float, float]]:
    return [coords[node] for node in path]


def extend_path_polyline_with_ports(
    path: list[NodeId],
    coords: Dict[NodeId, tuple[float, float]],
    node_risk: Dict[NodeId, float],
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> tuple[list[Tuple[float, float]], list[float]]:
    """
    Prepend/append true ports only when the stub to the first/last sea node does not
    cross land (avoids drawing a chord over coast/islands).
    """
    inner_coords = to_coordinate_list(path, coords)
    inner_risks = [round(node_risk[node], 3) for node in path]
    if not inner_coords:
        return [], []

    full_coords = list(inner_coords)
    full_risks = list(inner_risks)
    first_lat, first_lon = full_coords[0]
    last_lat, last_lon = full_coords[-1]

    if not edge_crosses_land(source_lat, source_lon, first_lat, first_lon):
        full_coords = [(source_lat, source_lon)] + full_coords
        full_risks = [full_risks[0]] + full_risks
    if not edge_crosses_land(last_lat, last_lon, dest_lat, dest_lon):
        full_coords = full_coords + [(dest_lat, dest_lon)]
        full_risks = full_risks + [full_risks[-1]]
    return full_coords, full_risks

