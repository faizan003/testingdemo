import math
import os
import asyncio
from typing import Dict, Tuple

import httpx

from app.services.grid import haversine_km


def synthetic_weather_risk(lat: float, lon: float) -> float:
    base = 0.35 + 0.2 * math.sin(math.radians(lat * 1.7)) + 0.15 * math.cos(math.radians(lon * 1.1))
    storm_band = 0.35 if (8 <= lat <= 22 and 60 <= lon <= 90) else 0.0
    risk = base + storm_band
    return max(0.0, min(1.0, risk))


async def fetch_weather_risk_for_points(
    points: Dict[Tuple[int, int], tuple[float, float]],
    use_live_weather: bool = False,
) -> Dict[Tuple[int, int], float]:
    if not use_live_weather:
        return {node: synthetic_weather_risk(lat, lon) for node, (lat, lon) in points.items()}

    # Uses Open-Meteo marine weather for a lightweight keyless demo path.
    # To keep UI responsive, sample points and project sampled risks to nearby nodes.
    timeout = float(os.getenv("WEATHER_TIMEOUT_SECONDS", "6"))
    max_live_points = int(os.getenv("WEATHER_MAX_LIVE_POINTS", "80"))
    point_items = list(points.items())
    sample_step = max(1, math.ceil(len(point_items) / max_live_points))
    sampled_items = point_items[::sample_step]

    sampled_risks: Dict[Tuple[int, int], float] = {}
    semaphore = asyncio.Semaphore(int(os.getenv("WEATHER_CONCURRENCY", "10")))

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def fetch_one(node: Tuple[int, int], lat: float, lon: float) -> None:
            async with semaphore:
                url = (
                    "https://marine-api.open-meteo.com/v1/marine"
                    f"?latitude={lat}&longitude={lon}"
                    "&hourly=wave_height,wind_wave_height,wind_wave_period"
                    "&forecast_days=1"
                )
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    payload = response.json()
                    hourly = payload.get("hourly", {})
                    wave = _safe_mean(hourly.get("wave_height", []))
                    wind_wave = _safe_mean(hourly.get("wind_wave_height", []))
                    period = _safe_mean(hourly.get("wind_wave_period", []))
                    risk = min(1.0, (wave / 5.0) * 0.55 + (wind_wave / 3.0) * 0.35 + (period / 12.0) * 0.1)
                    sampled_risks[node] = max(0.0, risk)
                except Exception:
                    sampled_risks[node] = synthetic_weather_risk(lat, lon)

        await asyncio.gather(*(fetch_one(node, lat, lon) for node, (lat, lon) in sampled_items))

    # Project sampled risks to all nodes via nearest sampled node in grid index space.
    if not sampled_risks:
        return {node: synthetic_weather_risk(lat, lon) for node, (lat, lon) in points.items()}

    # Inverse-distance weighted blend from marine samples so risk varies smoothly
    # across the grid (nearest-neighbor alone makes entire corridors identical → no real optimization).
    sample_list = [(n, sampled_risks[n], points[n][0], points[n][1]) for n in sampled_risks]
    full_risks: Dict[Tuple[int, int], float] = {}
    k_neighbors = int(os.getenv("WEATHER_IDW_NEIGHBORS", "8"))
    power = float(os.getenv("WEATHER_IDW_POWER", "2"))

    for node, (lat, lon) in points.items():
        dists: list[tuple[float, float]] = []
        for _s_node, s_risk, slat, slon in sample_list:
            d = haversine_km(lat, lon, slat, slon) + 1e-4
            dists.append((d, s_risk))
        dists.sort(key=lambda x: x[0])
        top = dists[: min(k_neighbors, len(dists))]
        weight_sum = sum(1.0 / (d**power) for d, _ in top)
        blended = sum((1.0 / (d**power)) * r for d, r in top) / weight_sum
        full_risks[node] = max(0.0, min(1.0, blended))

    # Blend with synthetic spatial risk so corridors are not perfectly uniform.
    # Pure marine-IDW fields are often flat along the shortest sea lane → baseline == optimized.
    synth_weight = float(os.getenv("WEATHER_SYNTH_BLEND", "0.55"))
    for node, (lat, lon) in points.items():
        syn = synthetic_weather_risk(lat, lon)
        merged = (1.0 - synth_weight) * full_risks[node] + synth_weight * syn
        full_risks[node] = max(0.0, min(1.0, merged))

    return full_risks


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

