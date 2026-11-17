"""Open-Meteo geocoding and forecast access."""

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from math import isfinite
from threading import Lock
import time
from typing import Any

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = (5, 20)
FORECAST_CACHE_TTL_SECONDS = 300
GEOCODING_CACHE_TTL_SECONDS = 3600
CACHE_MAX_ENTRIES = 512
MAX_429_RETRIES = 2
MAX_RETRY_WAIT_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 2
FORECAST_CACHE_COORDINATE_PRECISION = 3
RATE_LIMIT_MESSAGE = (
    "The weather provider is receiving too many requests. Please try again shortly."
)

_response_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_in_flight: dict[str, Future[dict[str, Any]]] = {}
_cache_lock = Lock()

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """An upstream weather API error safe to report to the application layer."""


class WeatherRateLimitError(WeatherServiceError):
    """The upstream weather API remained rate-limited after bounded retries."""


class LocationNotFoundError(WeatherServiceError):
    """No geocoding result was found for the requested place."""


class InvalidCoordinatesError(ValueError):
    """Latitude or longitude is outside the valid geographic range."""


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise WeatherServiceError(f"The weather provider returned invalid {field} data.")
    return float(value)


def _weather_code(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WeatherServiceError("The weather provider returned an invalid condition code.")
    return value


def _percentage(value: Any, field: str) -> int:
    percentage = _number(value, field)
    if not 0 <= percentage <= 100:
        raise WeatherServiceError(f"The weather provider returned invalid {field} data.")
    return int(percentage)


def _weather_condition(value: Any) -> str:
    code = _weather_code(value)
    return WEATHER_CODES.get(code, "Conditions unavailable")


def _cache_ttl(url: str) -> int:
    if url == FORECAST_URL:
        return FORECAST_CACHE_TTL_SECONDS
    if url == GEOCODING_URL:
        return GEOCODING_CACHE_TTL_SECONDS
    return 0


def _cache_key(url: str, params: dict[str, Any]) -> str:
    cache_params = params
    if url == FORECAST_URL:
        cache_params = params.copy()
        for coordinate in ("latitude", "longitude"):
            value = cache_params.get(coordinate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rounded = round(float(value), FORECAST_CACHE_COORDINATE_PRECISION)
                cache_params[coordinate] = 0.0 if rounded == 0 else rounded
    return json.dumps((url, cache_params), sort_keys=True, separators=(",", ":"))


def _retry_delay(response: requests.Response, attempt: int, waited: float) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            delay = max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                delay = min(
                    INITIAL_RETRY_DELAY_SECONDS * (2**attempt),
                    MAX_RETRY_DELAY_SECONDS,
                )
    else:
        delay = min(
            INITIAL_RETRY_DELAY_SECONDS * (2**attempt),
            MAX_RETRY_DELAY_SECONDS,
        )

    if delay > MAX_RETRY_WAIT_SECONDS - waited:
        return None
    return delay


def _fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    waited = 0.0
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.Timeout as exc:
            raise WeatherServiceError("The weather provider took too long to respond.") from exc
        except requests.RequestException as exc:
            raise WeatherServiceError("The weather provider is temporarily unavailable.") from exc

        if response.status_code == 429:
            delay = (
                _retry_delay(response, attempt, waited)
                if attempt < MAX_429_RETRIES
                else None
            )
            if delay is None:
                raise WeatherRateLimitError(RATE_LIMIT_MESSAGE)
            time.sleep(delay)
            waited += delay
            continue

        try:
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise WeatherServiceError("The weather provider is temporarily unavailable.") from exc
        except ValueError as exc:
            raise WeatherServiceError("The weather provider returned an unreadable response.") from exc

        if not isinstance(payload, dict):
            raise WeatherServiceError("The weather provider returned an invalid response.")
        return payload

    raise WeatherRateLimitError(RATE_LIMIT_MESSAGE)


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    key = _cache_key(url, params)
    now = time.monotonic()
    with _cache_lock:
        cached = _response_cache.get(key)
        if cached is not None:
            expires_at, payload = cached
            if expires_at > now:
                _response_cache.move_to_end(key)
                return deepcopy(payload)
            del _response_cache[key]

        future = _in_flight.get(key)
        is_owner = future is None
        if is_owner:
            future = Future()
            _in_flight[key] = future

    if not is_owner:
        return deepcopy(future.result())

    try:
        payload = _fetch_json(url, params)
    except BaseException as exc:
        with _cache_lock:
            future.set_exception(exc)
            del _in_flight[key]
        raise

    ttl = _cache_ttl(url)
    with _cache_lock:
        if ttl:
            _response_cache[key] = (time.monotonic() + ttl, payload)
            _response_cache.move_to_end(key)
            while len(_response_cache) > CACHE_MAX_ENTRIES:
                _response_cache.popitem(last=False)
        future.set_result(payload)
        del _in_flight[key]
    return deepcopy(payload)


def resolve_location(query: str) -> dict[str, Any]:
    payload = _get_json(
        GEOCODING_URL,
        {"name": query, "count": 1, "language": "en", "format": "json"},
    )
    results = payload.get("results")
    if not isinstance(results, list):
        if "results" not in payload:
            raise WeatherServiceError("The geocoding provider returned an invalid response.")
        raise LocationNotFoundError("I couldn't find that location. Try a nearby city or town.")
    if not results:
        raise LocationNotFoundError("I couldn't find that location. Try a nearby city or town.")

    result = results[0]
    if not isinstance(result, dict):
        raise WeatherServiceError("The geocoding provider returned an invalid location.")

    name = result.get("name")
    country = result.get("country")
    latitude = result.get("latitude")
    longitude = result.get("longitude")
    if not isinstance(name, str) or not name.strip() or not isinstance(country, str) or not country.strip():
        raise WeatherServiceError("The geocoding provider returned incomplete location details.")

    return {
        "name": name,
        "region": result.get("admin1") if isinstance(result.get("admin1"), str) else "",
        "country": country,
        "latitude": _number(latitude, "latitude"),
        "longitude": _number(longitude, "longitude"),
    }


def resolve_coordinates(latitude: Any, longitude: Any) -> dict[str, Any]:
    """Validate browser coordinates and create a location for the forecast API."""

    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or not isfinite(latitude)
        or not -90 <= latitude <= 90
    ):
        raise InvalidCoordinatesError("Latitude must be a number from -90 to 90.")
    if (
        isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
        or not isfinite(longitude)
        or not -180 <= longitude <= 180
    ):
        raise InvalidCoordinatesError("Longitude must be a number from -180 to 180.")

    return {
        "name": "Your location",
        "region": "",
        "country": "Browser location",
        "latitude": float(latitude),
        "longitude": float(longitude),
    }


def _current_rain_probability(payload: dict[str, Any]) -> int:
    hourly = payload.get("hourly")
    current = payload.get("current")
    if not isinstance(hourly, dict) or not isinstance(current, dict):
        raise WeatherServiceError("The forecast provider returned incomplete hourly data.")

    times = hourly.get("time")
    probabilities = hourly.get("precipitation_probability")
    current_time = current.get("time")
    if not isinstance(times, list) or not isinstance(probabilities, list) or len(times) != len(probabilities):
        raise WeatherServiceError("The forecast provider returned invalid hourly data.")
    if not isinstance(current_time, str) or not times:
        raise WeatherServiceError("The forecast provider returned no current observation time.")

    try:
        current_datetime = datetime.fromisoformat(current_time)
        hourly_datetimes = [datetime.fromisoformat(item) for item in times if isinstance(item, str)]
    except ValueError as exc:
        raise WeatherServiceError("The forecast provider returned invalid timestamps.") from exc
    if len(hourly_datetimes) != len(times):
        raise WeatherServiceError("The forecast provider returned invalid hourly timestamps.")

    closest_index = min(
        range(len(hourly_datetimes)),
        key=lambda index: abs((hourly_datetimes[index] - current_datetime).total_seconds()),
    )
    probability = probabilities[closest_index]
    if probability is None:
        raise WeatherServiceError("The forecast provider did not return current rain probability.")
    return int(_number(probability, "rain probability"))


def get_forecast(location: dict[str, Any]) -> dict[str, Any]:
    payload = _get_json(
        FORECAST_URL,
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "is_day,precipitation,weather_code,wind_speed_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "hourly": (
                "temperature_2m,precipitation_probability,precipitation,"
                "wind_speed_10m,weather_code"
            ),
            "forecast_days": 7,
            "timezone": "auto",
        },
    )
    current = payload.get("current")
    daily = payload.get("daily")
    hourly = payload.get("hourly")
    timezone = payload.get("timezone")
    if (
        not isinstance(current, dict)
        or not isinstance(daily, dict)
        or not isinstance(hourly, dict)
        or not isinstance(timezone, str)
        or not timezone.strip()
    ):
        raise WeatherServiceError("The forecast provider returned incomplete weather data.")

    required_current = (
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "wind_speed_10m",
        "weather_code",
    )
    if any(field not in current for field in required_current):
        raise WeatherServiceError("The forecast provider returned incomplete current conditions.")

    daily_fields = (
        "time",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
    )
    if any(not isinstance(daily.get(field), list) for field in daily_fields):
        raise WeatherServiceError("The forecast provider returned incomplete daily data.")
    daily_count = len(daily["time"])
    if daily_count < 7 or any(len(daily[field]) != daily_count for field in daily_fields[1:]):
        raise WeatherServiceError("The forecast provider returned inconsistent daily data.")

    forecast = []
    for index in range(daily_count):
        forecast_date = daily["time"][index]
        if not isinstance(forecast_date, str):
            raise WeatherServiceError("The forecast provider returned an invalid forecast date.")
        forecast.append(
            {
                "date": forecast_date,
                "high_c": _number(daily["temperature_2m_max"][index], "daily high"),
                "low_c": _number(daily["temperature_2m_min"][index], "daily low"),
                "condition": _weather_condition(daily["weather_code"][index]),
                "rain_probability_pct": _percentage(
                    daily["precipitation_probability_max"][index],
                    "daily rain probability",
                ),
            }
        )

    current_time = current.get("time")
    if not isinstance(current_time, str):
        raise WeatherServiceError("The forecast provider returned an invalid current observation time.")
    try:
        current_datetime = datetime.fromisoformat(current_time)
    except ValueError as exc:
        raise WeatherServiceError("The forecast provider returned an invalid current observation time.") from exc

    hourly_fields = (
        "time",
        "temperature_2m",
        "precipitation_probability",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    )
    if any(not isinstance(hourly.get(field), list) for field in hourly_fields):
        raise WeatherServiceError("The forecast provider returned incomplete hourly data.")
    hourly_count = len(hourly["time"])
    if hourly_count < 24:
        raise WeatherServiceError("The forecast provider returned incomplete hourly data.")
    if any(len(hourly[field]) != hourly_count for field in hourly_fields[1:]):
        raise WeatherServiceError("The forecast provider returned inconsistent hourly data.")

    hourly_forecast = []
    for index, timestamp in enumerate(hourly["time"]):
        if not isinstance(timestamp, str):
            raise WeatherServiceError("The forecast provider returned an invalid hourly timestamp.")
        try:
            forecast_datetime = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise WeatherServiceError("The forecast provider returned an invalid hourly timestamp.") from exc
        if forecast_datetime < current_datetime:
            continue
        hourly_forecast.append(
            {
                "time": forecast_datetime.strftime("%Y-%m-%dT%H:%M"),
                "temperature_c": _number(hourly["temperature_2m"][index], "hourly temperature"),
                "rain_probability_pct": _percentage(
                    hourly["precipitation_probability"][index],
                    "hourly rain probability",
                ),
                "precipitation_mm": _number(hourly["precipitation"][index], "hourly precipitation"),
                "wind_kmh": _number(hourly["wind_speed_10m"][index], "hourly wind speed"),
                "condition": _weather_condition(hourly["weather_code"][index]),
            }
        )
    if len(hourly_forecast) < 24:
        raise WeatherServiceError("The forecast provider returned fewer than 24 upcoming hourly forecasts.")

    return {
        "timezone": timezone,
        "current": {
            "temperature_c": _number(current["temperature_2m"], "temperature"),
            "feels_like_c": _number(current["apparent_temperature"], "feels-like temperature"),
            "condition": _weather_condition(current["weather_code"]),
            "humidity_pct": _percentage(current["relative_humidity_2m"], "humidity"),
            "wind_kmh": _number(current["wind_speed_10m"], "wind speed"),
            "rain_probability_pct": _current_rain_probability(payload),
            "observed_at": current_time,
            "is_day": current.get("is_day") == 1,
        },
        "forecast": forecast,
        "hourly": hourly_forecast[:24],
        "hourly_forecast": hourly_forecast,
    }
