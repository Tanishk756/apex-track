"""
APEX-Track Weather Telemetry & Atmospheric Compensation Engine
================================================================
Ingests live weather data via OpenWeatherMap API to compute
optical atmospheric compensation (fog/visibility contrast gain) and
crosswind drift vectors for trajectory prediction.
"""

from __future__ import annotations

import os
import time
import requests
from typing import Any, Dict, Optional
import structlog

from apex.engine.config.security import SecurityManager, mask_key

log = structlog.get_logger(__name__)


class WeatherEngine:
    """OpenWeatherMap Live Telemetry Engine."""

    _instance: Optional[WeatherEngine] = None

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or SecurityManager.instance().weather_key
        self.location_name: str = "Defense Tactical Sector Alpha"
        self.latitude: float = 28.6139  # Default lat
        self.longitude: float = 77.2090 # Default lon
        self.last_fetch_time: float = 0.0
        self.cached_weather: Dict[str, Any] = self._default_weather()

    @classmethod
    def instance(cls) -> WeatherEngine:
        if cls._instance is None:
            cls._instance = WeatherEngine()
        return cls._instance

    def _default_weather(self) -> Dict[str, Any]:
        return {
            "condition": "CLEAR",
            "temperature_c": 24.5,
            "humidity_pct": 45,
            "visibility_m": 10000,
            "wind_speed_ms": 3.5,
            "wind_deg": 140,
            "pressure_hpa": 1013,
            "atmospheric_attenuation": 1.0,
            "recommended_vision_mode": "EO_DAYLIGHT",
        }

    def fetch_live_weather(self, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """
        Fetches live meteorological data from OpenWeatherMap API.
        Caches result for 60 seconds to avoid hitting API rate limits.
        """
        now = time.time()
        if now - self.last_fetch_time < 60.0 and self.cached_weather:
            return self.cached_weather

        target_lat = lat if lat is not None else self.latitude
        target_lon = lon if lon is not None else self.longitude

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={target_lat}&lon={target_lon}&appid={self.api_key}&units=metric"

        try:
            res = requests.get(url, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                weather_main = data.get("weather", [{}])[0].get("main", "CLEAR").upper()
                vis_m = data.get("visibility", 10000)
                temp = data.get("main", {}).get("temp", 24.5)
                humidity = data.get("main", {}).get("humidity", 45)
                wind_speed = data.get("wind", {}).get("speed", 3.5)
                wind_deg = data.get("wind", {}).get("deg", 140)

                # Compute optical attenuation factor (low visibility -> higher CLAHE gain requirement)
                attenuation = max(1.0, min(3.0, 10000.0 / max(vis_m, 500.0)))
                rec_mode = "THERMAL_FLIR" if vis_m < 2000 or weather_main in ["FOG", "MIST", "HAZE", "RAIN"] else "EO_DAYLIGHT"

                self.cached_weather = {
                    "condition": weather_main,
                    "temperature_c": temp,
                    "humidity_pct": humidity,
                    "visibility_m": vis_m,
                    "wind_speed_ms": wind_speed,
                    "wind_deg": wind_deg,
                    "pressure_hpa": data.get("main", {}).get("pressure", 1013),
                    "atmospheric_attenuation": round(attenuation, 2),
                    "recommended_vision_mode": rec_mode,
                    "city": data.get("name", "Tactical Sector"),
                }
                self.last_fetch_time = now
                log.info("openweather_fetch_success", condition=weather_main, temp=temp, wind=wind_speed)
            else:
                log.warning("openweather_http_error", status_code=res.status_code)
        except Exception as exc:
            log.warning("openweather_fetch_failed", error=str(exc))

        return self.cached_weather

    def get_status(self) -> Dict[str, Any]:
        w = self.fetch_live_weather()
        return {
            "status": "ONLINE" if self.api_key else "OFFLINE",
            "api_key_configured": bool(self.api_key),
            "api_key_masked": mask_key(self.api_key),
            "weather": w,
        }
