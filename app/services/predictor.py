"""Lightweight ML ETA / fuel predictor for demo (scikit-learn RandomForest)."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _feature_row(
    distance_km: float,
    avg_risk: float,
    max_risk: float,
    w_distance: float,
    w_risk: float,
    w_fuel: float,
    lat_span: float,
    lon_span: float,
    great_circle_km: float,
) -> list[float]:
    return [
        distance_km,
        avg_risk,
        max_risk,
        w_distance,
        w_risk,
        w_fuel,
        lat_span,
        lon_span,
        great_circle_km,
        distance_km / max(great_circle_km, 1.0),
    ]


@lru_cache(maxsize=1)
def _trained_models() -> tuple[RandomForestRegressor, RandomForestRegressor]:
    """Train once on synthetic voyage samples (no external dataset required)."""
    rng = np.random.default_rng(42)
    n = 420
    X: list[list[float]] = []
    y_eta: list[float] = []
    y_fuel: list[float] = []
    for _ in range(n):
        gc = float(rng.uniform(400, 9000))
        detour = float(rng.uniform(1.02, 1.55))
        distance = gc * detour
        avg_risk = float(rng.uniform(0.05, 0.85))
        max_risk = min(1.0, avg_risk + float(rng.uniform(0.0, 0.35)))
        w_d = float(rng.uniform(0.4, 2.5))
        w_r = float(rng.uniform(0.4, 3.5))
        w_f = float(rng.uniform(0.4, 2.5))
        lat_span = float(rng.uniform(2, 40))
        lon_span = float(rng.uniform(2, 55))
        speed = 26.0 + 6.0 * (1.0 - avg_risk)  # slower in rough seas
        eta = distance / speed * (1.0 + 0.18 * avg_risk + 0.04 * w_r)
        fuel = distance * (1.0 + 1.55 * avg_risk) * (1.0 + 0.08 * w_f)
        eta += float(rng.normal(0, 1.2))
        fuel += float(rng.normal(0, 8.0))
        X.append(_feature_row(distance, avg_risk, max_risk, w_d, w_r, w_f, lat_span, lon_span, gc))
        y_eta.append(max(1.0, eta))
        y_fuel.append(max(10.0, fuel))

    eta_model = RandomForestRegressor(n_estimators=48, max_depth=8, random_state=42, n_jobs=1)
    fuel_model = RandomForestRegressor(n_estimators=48, max_depth=8, random_state=42, n_jobs=1)
    eta_model.fit(np.asarray(X), np.asarray(y_eta))
    fuel_model.fit(np.asarray(X), np.asarray(y_fuel))
    return eta_model, fuel_model


def predict_voyage_outcomes(
    *,
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
    distance_km: float,
    avg_risk: float,
    max_risk: float,
    w_distance: float,
    w_risk: float,
    w_fuel: float,
    optimizer_eta: float,
    optimizer_fuel: float,
) -> dict[str, Any]:
    eta_model, fuel_model = _trained_models()
    gc = _haversine_km(source_lat, source_lon, dest_lat, dest_lon)
    lat_span = abs(dest_lat - source_lat)
    lon_span = abs(dest_lon - source_lon)
    row = np.asarray(
        [
            _feature_row(
                distance_km,
                avg_risk,
                max_risk,
                w_distance,
                w_risk,
                w_fuel,
                lat_span,
                lon_span,
                gc,
            )
        ]
    )
    pred_eta = float(eta_model.predict(row)[0])
    pred_fuel = float(fuel_model.predict(row)[0])
    return {
        "model": "RandomForestRegressor",
        "trained_samples": 420,
        "predicted_eta_hours": round(pred_eta, 2),
        "predicted_fuel_proxy": round(pred_fuel, 2),
        "optimizer_eta_hours": round(optimizer_eta, 2),
        "optimizer_fuel_proxy": round(optimizer_fuel, 2),
        "eta_abs_error_hours": round(abs(pred_eta - optimizer_eta), 2),
        "fuel_abs_error": round(abs(pred_fuel - optimizer_fuel), 2),
        "features_used": [
            "distance_km",
            "avg_risk",
            "max_risk",
            "w_distance",
            "w_risk",
            "w_fuel",
            "lat_span",
            "lon_span",
            "great_circle_km",
            "detour_ratio",
        ],
    }


def build_risk_heatmap(
    coords: dict[tuple[int, int], tuple[float, float]],
    node_risk: dict[tuple[int, int], float],
    max_points: int = 140,
) -> list[dict[str, float]]:
    items = list(node_risk.items())
    if not items:
        return []
    step = max(1, math.ceil(len(items) / max_points))
    heat: list[dict[str, float]] = []
    for node, risk in items[::step]:
        lat, lon = coords[node]
        heat.append({"lat": round(lat, 5), "lon": round(lon, 5), "risk": round(float(risk), 3)})
    return heat


def build_mission_alerts(
    *,
    weather_mode: str,
    avg_risk: float,
    max_risk: float,
    severe_points: int,
    baseline_distance: float,
    optimized_distance: float,
    baseline_risk: float,
    optimized_risk: float,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if max_risk >= 0.75:
        alerts.append(
            {
                "level": "critical",
                "title": "Severe sea state detected",
                "detail": f"Peak corridor risk {max_risk:.2f} — optimized path prioritizes safer lanes.",
            }
        )
    elif avg_risk >= 0.45:
        alerts.append(
            {
                "level": "warning",
                "title": "Elevated marine risk",
                "detail": f"Average field risk {avg_risk:.2f} under {weather_mode} weather mode.",
            }
        )
    else:
        alerts.append(
            {
                "level": "info",
                "title": "Corridor conditions favorable",
                "detail": f"Average risk {avg_risk:.2f}; baseline and optimized paths remain navigable.",
            }
        )

    detour = optimized_distance - baseline_distance
    if detour > 25:
        alerts.append(
            {
                "level": "warning",
                "title": "Weather detour engaged",
                "detail": f"Optimized route adds {detour:.1f} km versus pure shortest sea path.",
            }
        )
    elif detour < -5:
        alerts.append(
            {
                "level": "info",
                "title": "Shorter optimized corridor",
                "detail": f"Optimized path saves {abs(detour):.1f} km while balancing risk weights.",
            }
        )

    risk_drop = baseline_risk - optimized_risk
    if risk_drop > 0.02:
        alerts.append(
            {
                "level": "ok",
                "title": "Risk reduced",
                "detail": f"Average path risk improved by {risk_drop:.3f} versus baseline.",
            }
        )
    if severe_points >= 8:
        alerts.append(
            {
                "level": "critical",
                "title": "Multiple severe grid cells",
                "detail": f"{severe_points} severe samples in the weather field — review heatmap overlay.",
            }
        )
    return alerts
