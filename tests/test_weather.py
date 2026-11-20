"""Tests for tools/weather.py.

Mocked tests use a fake requests.get that answers the geocoding call first,
then the weather call. One 'live' test calls the real API.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.weather import get_weather


def fake_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data if json_data is not None else {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} error", response=resp)
    return resp


GEO_TOKYO = {"results": [{"name": "Tokyo", "country": "Japan",
                          "latitude": 35.6895, "longitude": 139.69171}]}
GEO_NOT_FOUND = {"generationtime_ms": 0.1}  # real API: 200 OK, no "results" key


def weather_reply(code=0):
    return {"current": {"time": "2026-11-20T17:45", "temperature_2m": 10.1,
                        "apparent_temperature": 7.0, "relative_humidity_2m": 47,
                        "wind_speed_10m": 5.8, "weather_code": code}}


@patch("tools.weather.requests.get")
def test_success_shape(mock_get):
    mock_get.side_effect = [fake_response(200, GEO_TOKYO), fake_response(200, weather_reply(3))]
    result = get_weather("tokyo")
    assert result["city"] == "Tokyo" and result["country"] == "Japan"
    assert result["temperature_c"] == 10.1
    assert result["conditions"] == "Overcast"
    assert result["summary"] == "Tokyo, Japan: Overcast, 10.1 °C (feels like 7.0 °C), wind 5.8 km/h"
    assert result["source"] == "Open-Meteo" and "as_of" in result
    # Rule: every API call uses a timeout
    for call in mock_get.call_args_list:
        assert "timeout" in call.kwargs


@patch("tools.weather.requests.get")
def test_weather_call_uses_city_coordinates(mock_get):
    mock_get.side_effect = [fake_response(200, GEO_TOKYO), fake_response(200, weather_reply())]
    get_weather("Tokyo")
    params = mock_get.call_args_list[1].kwargs["params"]
    assert (params["latitude"], params["longitude"]) == (35.6895, 139.69171)


@patch("tools.weather.requests.get")
def test_city_not_found(mock_get):
    mock_get.return_value = fake_response(200, GEO_NOT_FOUND)
    result = get_weather("Atlantisxyz")
    assert "not found" in result["error"]
    assert mock_get.call_count == 1  # no weather call for a missing city


@pytest.mark.parametrize("bad", ["", "   ", None])
@patch("tools.weather.requests.get")
def test_empty_city_needs_no_api(mock_get, bad):
    assert "give a city name" in get_weather(bad)["error"]
    mock_get.assert_not_called()


@patch("tools.weather.requests.get")
def test_unknown_weather_code(mock_get):
    mock_get.side_effect = [fake_response(200, GEO_TOKYO), fake_response(200, weather_reply(999))]
    assert get_weather("Tokyo")["conditions"] == "Unknown conditions"


@patch("tools.weather.requests.get", side_effect=requests.Timeout)
def test_timeout(mock_get):
    assert "too long" in get_weather("Tokyo")["error"]


@patch("tools.weather.requests.get", side_effect=requests.ConnectionError)
def test_no_connection(mock_get):
    assert "Could not reach" in get_weather("Tokyo")["error"]


@patch("tools.weather.requests.get")
def test_server_error_500(mock_get):
    mock_get.side_effect = [fake_response(200, GEO_TOKYO), fake_response(500)]
    assert "returned an error" in get_weather("Tokyo")["error"]


@patch("tools.weather.requests.get")
def test_weird_reply_shape(mock_get):
    mock_get.side_effect = [fake_response(200, GEO_TOKYO), fake_response(200, {"oops": 1})]
    assert "unexpected answer" in get_weather("Tokyo")["error"]


@pytest.mark.live
def test_live_tokyo():
    """Real API call. Skip offline with: pytest -m "not live" """
    result = get_weather("Tokyo")
    assert "error" not in result, result
    assert result["country"] == "Japan"
