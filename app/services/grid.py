import math
from functools import lru_cache
from typing import Dict, Tuple

import networkx as nx
from global_land_mask import globe


NodeId = Tuple[int, int]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _great_circle_intermediate(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> tuple[float, float]:
    """Point at fraction f in [0, 1] along the great circle from (lat1,lon1) to (lat2,lon2)."""
    if f <= 0.0:
        return lat1, lon1
    if f >= 1.0:
        return lat2, lon2
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    d = 2 * math.asin(
        min(
            1.0,
            math.sqrt(
                max(0.0, math.sin((φ2 - φ1) / 2) ** 2)
                + math.cos(φ1) * math.cos(φ2) * math.sin((λ2 - λ1) / 2) ** 2
            ),
        )
    )
    if d < 1e-12:
        return lat1, lon1
    a = math.sin((1 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(φ1) * math.cos(λ1) + b * math.cos(φ2) * math.cos(λ2)
    y = a * math.cos(φ1) * math.sin(λ1) + b * math.cos(φ2) * math.sin(λ2)
    z = a * math.sin(φ1) + b * math.sin(φ2)
    φi = math.atan2(z, math.hypot(x, y))
    λi = math.atan2(y, x)
    return math.degrees(φi), math.degrees(λi)


def edge_crosses_land(lat1: float, lon1: float, lat2: float, lon2: float, strict: bool = False) -> bool:
    """
    Reject chords that cut through land bridges (Sumatra, Sri Lanka, etc.).

    Soft mode (orthogonal neighbours): tolerate brief coast clips so narrow straits
    like Malacca stay connected.
    Strict mode (diagonals): tighter — diagonals are the usual peninsula shortcuts.
    """
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    # Short orthogonal hops: keep Malacca / Palk open (both ends already sea)
    if not strict and dist_km < 110:
        return False
    if dist_km < 120:
        n = 8 if strict else 6
    else:
        n = max(10, min(36, int(dist_km / 30) + 6))
    land_hits = 0
    consecutive = 0
    max_consecutive = 0
    for k in range(1, n):
        lat, lon = _great_circle_intermediate(lat1, lon1, lat2, lon2, k / n)
        if _is_land(lat, lon):
            land_hits += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    interior = n - 1
    if interior <= 0:
        return False
    if strict:
        return max_consecutive >= 2 or (land_hits / interior) > 0.16
    # Soft ortho: only hard land bridges (not a coast clip in a strait)
    return max_consecutive >= 4 or (land_hits / interior) > 0.45


def ensure_sea_point(lat: float, lon: float, step_deg: float = 0.12, rings: int = 30) -> tuple[float, float]:
    """
    If a geocoded 'port' lands inland (city centre), walk outward until we hit water.
    Custom place names often resolve inland and then break sea routing.
    """
    if not _is_land(lat, lon):
        return lat, lon
    for ring in range(1, rings + 1):
        samples = max(8, ring * 8)
        for k in range(samples):
            ang = (2.0 * math.pi * k) / samples
            tlat = lat + ring * step_deg * math.cos(ang)
            tlon = lon + ring * step_deg * math.sin(ang)
            if -90.0 <= tlat <= 90.0 and -180.0 <= tlon <= 180.0 and not _is_land(tlat, tlon):
                return tlat, tlon
    return lat, lon


def _gc_land_fraction(lat1: float, lon1: float, lat2: float, lon2: float, samples: int = 24) -> float:
    """How much of the straight great-circle is land — high means we must sail around."""
    hits = 0
    for k in range(1, samples):
        lat, lon = _great_circle_intermediate(lat1, lon1, lat2, lon2, k / samples)
        if _is_land(lat, lon):
            hits += 1
    return hits / max(1, samples - 1)


def _navigation_grid_single(
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
    resolution: int,
    allow_diagonals: bool,
    margin_scale: float = 1.0,
) -> tuple[nx.Graph, Dict[NodeId, tuple[float, float]], NodeId, NodeId]:
    """Sea grid; land-bridge edges dropped. Box must allow going *around* land barriers."""
    span_lat = abs(dest_lat - source_lat)
    span_lon = abs(dest_lon - source_lon)
    around = max(span_lat, span_lon)
    margin_lat = max(3.5, span_lat * 0.55) * margin_scale
    margin_lon = max(3.5, span_lon * 0.45) * margin_scale
    if margin_scale > 1.0:
        margin_lat = max(margin_lat, around * 0.75 * margin_scale, 9.0 * margin_scale)
        margin_lon = max(margin_lon, around * 0.4 * margin_scale, 5.0 * margin_scale)
    min_lat = min(source_lat, dest_lat) - margin_lat
    max_lat = max(source_lat, dest_lat) + margin_lat
    min_lon = min(source_lon, dest_lon) - margin_lon
    max_lon = max(source_lon, dest_lon) + margin_lon
    min_lat = max(-85.0, min_lat)
    max_lat = min(85.0, max_lat)

    lat_step = (max_lat - min_lat) / (resolution - 1)
    lon_step = (max_lon - min_lon) / (resolution - 1)

    graph = nx.Graph()
    coords: Dict[NodeId, tuple[float, float]] = {}
    sea_nodes: set[NodeId] = set()

    for i in range(resolution):
        for j in range(resolution):
            lat = min_lat + i * lat_step
            lon = min_lon + j * lon_step
            node_id = (i, j)
            coords[node_id] = (lat, lon)
            if not _is_land(lat, lon):
                graph.add_node(node_id)
                sea_nodes.add(node_id)

    ortho = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = ortho + (diag if allow_diagonals else [])

    for i in range(resolution):
        for j in range(resolution):
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if not (0 <= ni < resolution and 0 <= nj < resolution):
                    continue
                a, b = (i, j), (ni, nj)
                if a not in sea_nodes or b not in sea_nodes or a > b:
                    continue
                la, loa = coords[a]
                lb, lob = coords[b]
                is_diag = di != 0 and dj != 0
                # Ortho edges stay (Malacca needs them). Only drop diagonal land chords.
                if is_diag and edge_crosses_land(la, loa, lb, lob, strict=True):
                    continue
                graph.add_edge(a, b, distance_km=haversine_km(la, loa, lb, lob))

    source_node = nearest_node(coords, source_lat, source_lon, sea_nodes)
    dest_node = nearest_node(coords, dest_lat, dest_lon, sea_nodes)
    return graph, coords, source_node, dest_node


def build_navigation_grid(
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
    resolution: int = 18,
) -> tuple[nx.Graph, Dict[NodeId, tuple[float, float]], NodeId, NodeId]:
    """
    Build a sea-only graph. Prefer a tight corridor when the straight line is mostly water
    (Malacca). If the great-circle crosses land (peninsula / continent), expand margins
    immediately so custom ports like Mumbai → Visakhapatnam do not hang or fail.
    """
    source_lat, source_lon = ensure_sea_point(source_lat, source_lon)
    dest_lat, dest_lon = ensure_sea_point(dest_lat, dest_lon)

    leg_km = haversine_km(source_lat, source_lon, dest_lat, dest_lon)
    short_enough_km = max(leg_km * 1.85, leg_km + 500.0)
    land_frac = _gc_land_fraction(source_lat, source_lon, dest_lat, dest_lon)
    needs_detour = land_frac >= 0.22

    if needs_detour:
        # Around-peninsula: wider box, keep resolution modest for speed
        margin_scales = (1.55, 2.1, 2.6)
        base_res = min(24, max(resolution, 20))
    else:
        margin_scales = (1.0, 1.55, 2.1)
        # Strait corridors (Malacca) need denser sea nodes — res 28 often disconnects
        base_res = min(36, max(resolution, 32, int(20 + leg_km / 260)))

    for margin_scale in margin_scales:
        if needs_detour:
            attempt_res = (min(26, base_res + (2 if margin_scale > 1.6 else 0)),)
        else:
            attempt_res = (min(36, base_res), min(34, max(32, base_res - 2)))

        scale_best = None
        scale_best_len = float("inf")

        for use_res in attempt_res:
            graph, coords, source_node, dest_node = _navigation_grid_single(
                source_lat,
                source_lon,
                dest_lat,
                dest_lon,
                use_res,
                True,
                margin_scale=margin_scale,
            )
            if source_node not in graph or dest_node not in graph:
                continue
            if not nx.has_path(graph, source_node, dest_node):
                continue
            path_len = nx.shortest_path_length(
                graph, source_node, dest_node, weight="distance_km"
            )
            if path_len < scale_best_len:
                scale_best_len = path_len
                scale_best = (graph, coords, source_node, dest_node)
            if margin_scale <= 1.0 and scale_best_len <= short_enough_km:
                return scale_best

        if scale_best is not None:
            return scale_best

    raise ValueError(
        "No sea-only route between these places. Use coastal ports (add “Port” to the name), "
        "not inland cities, or pick a demo scenario."
    )


def nearest_node(coords: Dict[NodeId, tuple[float, float]], lat: float, lon: float, candidates: set[NodeId]) -> NodeId:
    if not candidates:
        raise ValueError("No sea nodes available in navigation grid.")
    return min(candidates, key=lambda node: haversine_km(lat, lon, coords[node][0], coords[node][1]))


@lru_cache(maxsize=400_000)
def _is_land_cached(lat_r: float, lon_r: float) -> bool:
    return bool(globe.is_land(lat_r, lon_r))


def _is_land(lat: float, lon: float) -> bool:
    # ~0.005° cache bins — fine enough for Malacca / Palk Strait
    return _is_land_cached(round(lat, 3), round(lon, 3))
