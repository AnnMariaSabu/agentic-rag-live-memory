"""Weather tool: current weather for a city, from Open-Meteo (free, no key).

Two calls: city name -> coordinates (geocoding), coordinates -> weather.
Like the currency tool, it NEVER raises: failures return {"error": "..."}.
"""

from datetime import datetime, timezone

import requests
from pydantic import BaseModel, Field

import cache
import config

SOURCE = "Open-Meteo"

# WMO weather codes -> words (https://open-meteo.com/en/docs, "WMO Weather interpretation codes")
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Light snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with light hail", 97: "Thunderstorm with heavy hail",
}


class CityNotFound(Exception):
    """Geocoding found no match for the city name."""


class Weather(BaseModel):
    """Shape of a successful answer. Pydantic checks it for us."""

    city: str
    country: str
    latitude: float
    longitude: float
    temperature_c: float
    feels_like_c: float
    humidity_pct: float = Field(ge=0, le=100)
    wind_kmh: float = Field(ge=0)
    conditions: str
    summary: str
    observed_at: str  # local time at the weather station
    source: str
    as_of: str  # when WE fetched it (UTC)


def _find_city(name: str) -> dict:
    """Call 1: 'Tokyo' -> {name, country, latitude, longitude}."""
    r = requests.get(
        config.GEOCODING_URL,
        params={"name": name, "count": 1, "language": "en"},
        timeout=config.TIMEOUT,
    )
    r.raise_for_status()
    results = r.json().get("results")  # missing key = no match (API still answers 200)
    if not results:
        raise CityNotFound(name)
    return results[0]


def _current_weather(lat: float, lon: float) -> dict:
    """Call 2: coordinates -> current weather values."""
    r = requests.get(
        config.OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "wind_speed_10m,weather_code",
            "timezone": "auto",
        },
        timeout=config.TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["current"]


def get_weather(city: str) -> dict:
    """Current weather for a city, with a one-line summary in words."""
    if not isinstance(city, str) or not city.strip():
        return {"error": "Please give a city name, e.g. 'Tokyo'."}
    city = city.strip()

    # Step 7: reuse weather fetched less than WEATHER_TTL_MIN minutes ago.
    # Key is lower-case, so "Tokyo" and "tokyo" share one cache entry.
    result, freshness = cache.cached_call(
        f"weather:{city.lower()}", config.WEATHER_TTL_MIN, lambda: _fetch_weather(city))
    return result if "error" in result else {**result, **freshness}


def _fetch_weather(city: str) -> dict:
    """Both API calls + translation into words. Never raises."""
    try:
        place = _find_city(city)
        now = _current_weather(place["latitude"], place["longitude"])

        conditions = WEATHER_CODES.get(now["weather_code"], "Unknown conditions")
        name, country = place["name"], place.get("country", "")
        where = f"{name}, {country}" if country else name
        summary = (
            f"{where}: {conditions}, {now['temperature_2m']} °C "
            f"(feels like {now['apparent_temperature']} °C), wind {now['wind_speed_10m']} km/h"
        )
        result = Weather(
            city=name, country=country,
            latitude=place["latitude"], longitude=place["longitude"],
            temperature_c=now["temperature_2m"], feels_like_c=now["apparent_temperature"],
            humidity_pct=now["relative_humidity_2m"], wind_kmh=now["wind_speed_10m"],
            conditions=conditions, summary=summary,
            observed_at=now["time"], source=SOURCE,
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        return result.model_dump()

    except CityNotFound:
        return {"error": f"City '{city}' not found. Check the spelling, e.g. 'Tokyo' or 'Paris'."}
    except requests.Timeout:
        return {"error": "The weather service took too long to answer. Please try again later."}
    except requests.ConnectionError:
        return {"error": "Could not reach the weather service. Check the internet connection."}
    except requests.HTTPError as e:
        return {"error": f"The weather service returned an error ({e}). Please try again later."}
    except (KeyError, ValueError, TypeError) as e:  # unexpected reply shape / failed validation
        return {"error": f"The weather service sent an unexpected answer ({type(e).__name__})."}


if __name__ == "__main__":
    # Quick demo: python -m tools.weather
    for c in ["Tokyo", "paris", "Atlantisxyz", "  "]:
        print(f"{c!r}: {get_weather(c)}\n")
