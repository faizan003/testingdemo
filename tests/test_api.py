from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_optimize_route_contract():
    payload = {
        "source": {"lat": 13.0827, "lon": 80.2707},
        "destination": {"lat": 1.3521, "lon": 103.8198},
        "grid_resolution": 12,
        "use_live_weather": False,
    }
    response = client.post("/optimize-route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline_route" in data
    assert "optimized_route" in data
    assert "baseline_metrics" in data
    assert "optimized_metrics" in data
    assert "risk_heatmap" in data
    assert "ml_prediction" in data
    assert "alerts" in data
    assert data["ml_prediction"]["model"] == "RandomForestRegressor"
