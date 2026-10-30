import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from networkx import NetworkXNoPath

from app.schemas import (
    Coordinate,
    GeocodeResponse,
    MissionAlert,
    MlPrediction,
    OptimizeRouteRequest,
    OptimizeRouteResponse,
    RiskHeatPoint,
)
from app.services.grid import build_navigation_grid, ensure_sea_point
from app.services.optimizer import (
    apply_weighted_costs,
    calculate_metrics,
    extend_path_polyline_with_ports,
    optimized_path,
    shortest_path,
)
from app.services.predictor import build_mission_alerts, build_risk_heatmap, predict_voyage_outcomes
from app.services.weather import fetch_weather_risk_for_points

app = FastAPI(title="Smart Ship Route Optimization API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOMINATIM_UA = "SmartShipRouteOptimization/1.0 (college-demo; contact=local)"


def _rank_geocode_hit(item: dict) -> int:
    text = f"{item.get('display_name') or ''}".lower()
    score = 0
    if "port" in text:
        score += 3
    if "harbour" in text or "harbor" in text:
        score += 3
    if "bay" in text or "coast" in text or "anchorage" in text:
        score += 1
    return score


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/geocode", response_model=GeocodeResponse)
async def geocode(q: str = Query(..., min_length=2, max_length=120)) -> GeocodeResponse:
    """Resolve a port/place name via Nominatim, then snap inland hits out to nearby sea."""
    query = q.strip()
    variants = [f"{query} port", f"{query} harbour", f"{query} harbor", query]
    headers = {"User-Agent": NOMINATIM_UA, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
            for variant in variants:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"format": "json", "q": variant, "limit": 5},
                )
                if response.status_code != 200:
                    continue
                data = response.json()
                if not data:
                    continue
                hit = max(data, key=_rank_geocode_hit)
                lat = float(hit["lat"])
                lon = float(hit["lon"])
                sea_lat, sea_lon = ensure_sea_point(lat, lon)
                return GeocodeResponse(
                    lat=sea_lat,
                    lon=sea_lon,
                    display_name=str(hit.get("display_name") or variant),
                    query=query,
                )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Port lookup failed: {exc}") from exc

    raise HTTPException(
        status_code=404,
        detail='Port not found. Try clearer names like "Mumbai Port, India".',
    )


@app.post("/optimize-route", response_model=OptimizeRouteResponse)
async def optimize_route(payload: OptimizeRouteRequest) -> OptimizeRouteResponse:
    # Snap inland custom-port pins to water before routing / drawing stubs
    src_lat, src_lon = ensure_sea_point(payload.source.lat, payload.source.lon)
    dst_lat, dst_lon = ensure_sea_point(payload.destination.lat, payload.destination.lon)
    try:
        graph, coords, source_node, dest_node = build_navigation_grid(
            src_lat,
            src_lon,
            dst_lat,
            dst_lon,
            payload.grid_resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    node_risk = await fetch_weather_risk_for_points(coords, use_live_weather=payload.use_live_weather)
    try:
        baseline = shortest_path(graph, source_node, dest_node)
    except NetworkXNoPath as exc:
        raise HTTPException(status_code=400, detail="No valid sea route found for selected points.") from exc

    apply_weighted_costs(
        graph=graph,
        node_risk=node_risk,
        w_distance=payload.w_distance,
        w_risk=payload.w_risk,
        w_fuel=payload.w_fuel,
        risk_block_threshold=payload.risk_block_threshold,
    )
    try:
        optimized = optimized_path(graph, source_node, dest_node, coords)
    except NetworkXNoPath:
        optimized = baseline

    bl_latlons, baseline_risk_profile = extend_path_polyline_with_ports(
        baseline,
        coords,
        node_risk,
        src_lat,
        src_lon,
        dst_lat,
        dst_lon,
    )
    opt_latlons, optimized_risk_profile = extend_path_polyline_with_ports(
        optimized,
        coords,
        node_risk,
        src_lat,
        src_lon,
        dst_lat,
        dst_lon,
    )
    baseline_coords = [Coordinate(lat=lat, lon=lon) for lat, lon in bl_latlons]
    optimized_coords = [Coordinate(lat=lat, lon=lon) for lat, lon in opt_latlons]
    weather_values = list(node_risk.values())
    weather_report = {
        "average_risk": round(sum(weather_values) / max(1, len(weather_values)), 3),
        "max_risk": round(max(weather_values) if weather_values else 0.0, 3),
        "severe_points": sum(1 for risk in weather_values if risk >= 0.75),
        "sampled_points": len(weather_values),
    }
    baseline_metrics = calculate_metrics(baseline, graph, node_risk)
    optimized_metrics = calculate_metrics(optimized, graph, node_risk)

    heatmap = [RiskHeatPoint(**p) for p in build_risk_heatmap(coords, node_risk)]
    ml_raw = predict_voyage_outcomes(
        source_lat=src_lat,
        source_lon=src_lon,
        dest_lat=dst_lat,
        dest_lon=dst_lon,
        distance_km=optimized_metrics.distance_km,
        avg_risk=optimized_metrics.average_risk,
        max_risk=float(weather_report["max_risk"]),
        w_distance=payload.w_distance,
        w_risk=payload.w_risk,
        w_fuel=payload.w_fuel,
        optimizer_eta=optimized_metrics.estimated_eta_hours,
        optimizer_fuel=optimized_metrics.fuel_proxy,
    )
    alerts = [
        MissionAlert(**a)
        for a in build_mission_alerts(
            weather_mode="live" if payload.use_live_weather else "synthetic",
            avg_risk=float(weather_report["average_risk"]),
            max_risk=float(weather_report["max_risk"]),
            severe_points=int(weather_report["severe_points"]),
            baseline_distance=baseline_metrics.distance_km,
            optimized_distance=optimized_metrics.distance_km,
            baseline_risk=baseline_metrics.average_risk,
            optimized_risk=optimized_metrics.average_risk,
        )
    ]

    return OptimizeRouteResponse(
        baseline_route=baseline_coords,
        optimized_route=optimized_coords,
        baseline_risk_profile=baseline_risk_profile,
        optimized_risk_profile=optimized_risk_profile,
        weather_mode="live" if payload.use_live_weather else "synthetic",
        weather_report=weather_report,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        risk_heatmap=heatmap,
        ml_prediction=MlPrediction(**ml_raw),
        alerts=alerts,
    )
