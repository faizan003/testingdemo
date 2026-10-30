from app.services.grid import build_navigation_grid, edge_crosses_land
from app.services.optimizer import (
    apply_weighted_costs,
    calculate_metrics,
    extend_path_polyline_with_ports,
    optimized_path,
    shortest_path,
    to_coordinate_list,
)


def test_baseline_and_optimized_paths_exist():
    graph, coords, source, destination = build_navigation_grid(
        13.0827, 80.2707, 1.3521, 103.8198, resolution=18
    )
    node_risk = {node: 0.2 for node in coords}
    baseline = shortest_path(graph, source, destination)
    apply_weighted_costs(graph, node_risk, w_distance=1.0, w_risk=2.0, w_fuel=1.2, risk_block_threshold=0.95)
    optimized = optimized_path(graph, source, destination, coords)
    assert len(baseline) > 1
    assert len(optimized) > 1


def test_extend_path_polyline_connects_requested_ports():
    coords = {(0, 0): (10.0, 80.0), (1, 1): (11.0, 81.0)}
    node_risk = {(0, 0): 0.1, (1, 1): 0.2}
    path = [(0, 0), (1, 1)]
    full, risks = extend_path_polyline_with_ports(
        path,
        coords,
        node_risk,
        9.5,
        79.5,
        11.5,
        81.5,
    )
    assert full[0] == (9.5, 79.5)
    assert full[-1] == (11.5, 81.5)
    assert len(full) == len(path) + 2
    assert len(risks) == len(full)


def test_chennai_singapore_prefers_short_sea_corridor():
    """Baseline should use Malacca-scale distance, not a Borneo-scale ocean loop."""
    graph, coords, source, destination = build_navigation_grid(
        13.0827, 80.2707, 1.3521, 103.8198, resolution=18
    )
    path = shortest_path(graph, source, destination)
    metrics = calculate_metrics(path, graph, {n: 0.2 for n in coords})
    assert metrics.distance_km < 5500
    pts = to_coordinate_list(path, coords)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        assert not edge_crosses_land(a[0], a[1], b[0], b[1], strict=True)


def test_metrics_are_non_negative():
    graph, coords, source, destination = build_navigation_grid(10.0, 60.0, 11.0, 62.0, resolution=10)
    node_risk = {node: 0.3 for node in coords}
    path = shortest_path(graph, source, destination)
    metrics = calculate_metrics(path, graph, node_risk)
    assert metrics.distance_km >= 0
    assert metrics.fuel_proxy >= 0
    assert 0 <= metrics.average_risk <= 1
