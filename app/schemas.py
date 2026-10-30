from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class GeocodeResponse(BaseModel):
    lat: float
    lon: float
    display_name: str
    query: str


class RiskHeatPoint(BaseModel):
    lat: float
    lon: float
    risk: float


class MissionAlert(BaseModel):
    level: str
    title: str
    detail: str


class MlPrediction(BaseModel):
    model: str
    trained_samples: int
    predicted_eta_hours: float
    predicted_fuel_proxy: float
    optimizer_eta_hours: float
    optimizer_fuel_proxy: float
    eta_abs_error_hours: float
    fuel_abs_error: float
    features_used: list[str]


class OptimizeRouteRequest(BaseModel):
    source: Coordinate
    destination: Coordinate
    grid_resolution: int = Field(default=18, ge=8, le=50)
    w_distance: float = Field(default=1.0, ge=0)
    w_risk: float = Field(default=2.0, ge=0)
    w_fuel: float = Field(default=1.2, ge=0)
    risk_block_threshold: float = Field(default=0.78, ge=0, le=1.0)
    use_live_weather: bool = False


class RouteMetrics(BaseModel):
    distance_km: float
    fuel_proxy: float
    average_risk: float
    estimated_eta_hours: float


class OptimizeRouteResponse(BaseModel):
    baseline_route: list[Coordinate]
    optimized_route: list[Coordinate]
    baseline_risk_profile: list[float]
    optimized_risk_profile: list[float]
    weather_mode: str
    weather_report: dict[str, float | int]
    baseline_metrics: RouteMetrics
    optimized_metrics: RouteMetrics
    risk_heatmap: list[RiskHeatPoint] = Field(default_factory=list)
    ml_prediction: MlPrediction | None = None
    alerts: list[MissionAlert] = Field(default_factory=list)
