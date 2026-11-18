"""Open-Meteo geocoding and MET Norway forecast access."""

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from math import isfinite
import re
from threading import Lock
import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
REVERSE_GEOCODING_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
FORECAST_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MET_USER_AGENT = "WeatherGPT/1.0 (https://github.com/syntaxelite/WeatherGPT)"
REQUEST_TIMEOUT = (5, 20)
FORECAST_CACHE_TTL_SECONDS = 3 * 60 * 60
GEOCODING_CACHE_TTL_SECONDS = 3600
CACHE_MAX_ENTRIES = 512
MAX_429_RETRIES = 2
MAX_RETRY_WAIT_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 2
FORECAST_CACHE_COORDINATE_PRECISION = 4
RATE_LIMIT_MESSAGE = (
    "The weather provider is receiving too many requests. Please try again shortly."
)

_response_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_in_flight: dict[str, Future[dict[str, Any]]] = {}
_cache_lock = Lock()

WEATHER_SYMBOLS = {
    "clearsky": "Clear sky",
    "fair": "Mainly clear",
    "partlycloudy": "Partly cloudy",
    "cloudy": "Overcast",
    "fog": "Fog",
    "lightrain": "Light rain",
    "rain": "Rain",
    "heavyrain": "Heavy rain",
    "lightrainshowers": "Light rain showers",
    "rainshowers": "Rain showers",
    "heavyrainshowers": "Heavy rain showers",
    "lightsnow": "Light snow",
    "snow": "Snow",
    "heavysnow": "Heavy snow",
    "lightsnowshowers": "Light snow showers",
    "lightssnowshowers": "Light snow showers",
    "snowshowers": "Snow showers",
    "heavysnowshowers": "Heavy snow showers",
    "lightsleet": "Light sleet",
    "sleet": "Sleet",
    "heavysleet": "Heavy sleet",
    "lightsleetshowers": "Light sleet showers",
    "lightssleetshowers": "Light sleet showers",
    "sleetshowers": "Sleet showers",
    "heavysleetshowers": "Heavy sleet showers",
    "lightrainandthunder": "Light rain and thunder",
    "rainandthunder": "Rain and thunder",
    "heavyrainandthunder": "Heavy rain and thunder",
    "rainshowersandthunder": "Rain showers and thunder",
    "lightrainshowersandthunder": "Light rain showers and thunder",
    "heavyrainshowersandthunder": "Heavy rain showers and thunder",
    "lightsnowandthunder": "Light snow and thunder",
    "snowandthunder": "Snow and thunder",
    "heavysnowandthunder": "Heavy snow and thunder",
    "snowshowersandthunder": "Snow showers and thunder",
    "lightssnowshowersandthunder": "Light snow showers and thunder",
    "heavysnowshowersandthunder": "Heavy snow showers and thunder",
    "lightsleetandthunder": "Light sleet and thunder",
    "sleetandthunder": "Sleet and thunder",
    "heavysleetandthunder": "Heavy sleet and thunder",
    "lightssleetshowersandthunder": "Light sleet showers and thunder",
    "sleetshowersandthunder": "Sleet showers and thunder",
    "heavysleetshowersandthunder": "Heavy sleet showers and thunder",
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


def _percentage(value: Any, field: str) -> int:
    percentage = _number(value, field)
    if not 0 <= percentage <= 100:
        raise WeatherServiceError(f"The weather provider returned invalid {field} data.")
    return int(percentage)


def _weather_condition(symbol: Any, cloud_fraction: Any) -> str:
    if isinstance(symbol, str):
        symbol = re.sub(r"_(day|night|polartwilight)$", "", symbol)
        condition = WEATHER_SYMBOLS.get(symbol)
        if condition:
            return condition
    if isinstance(cloud_fraction, (int, float)) and not isinstance(cloud_fraction, bool):
        if cloud_fraction <= 20:
            return "Clear sky"
        if cloud_fraction <= 60:
            return "Partly cloudy"
        if cloud_fraction <= 90:
            return "Cloudy"
        return "Overcast"
    return "Conditions unavailable"


def _cache_ttl(url: str, cache_control: str = "") -> int:
    if url == FORECAST_URL:
        max_age = re.search(r"(?:^|,)\s*max-age=(\d+)", cache_control, re.IGNORECASE)
        server_ttl = int(max_age.group(1)) if max_age else 0
        return max(FORECAST_CACHE_TTL_SECONDS, server_ttl)
    if url in (GEOCODING_URL, REVERSE_GEOCODING_URL):
        return GEOCODING_CACHE_TTL_SECONDS
    return 0


def _cache_key(url: str, params: dict[str, Any]) -> str:
    cache_params = params
    if url == FORECAST_URL:
        cache_params = params.copy()
        for coordinate in ("lat", "lon"):
            value = cache_params.get(coordinate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                truncated = int(float(value) * 10**FORECAST_CACHE_COORDINATE_PRECISION) / (
                    10**FORECAST_CACHE_COORDINATE_PRECISION
                )
                cache_params[coordinate] = 0.0 if truncated == 0 else truncated
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


def _fetch_json(url: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    waited = 0.0
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            request_options: dict[str, Any] = {
                "params": params,
                "timeout": REQUEST_TIMEOUT,
            }
            if url in (FORECAST_URL, REVERSE_GEOCODING_URL):
                request_options["headers"] = {"User-Agent": MET_USER_AGENT}
            response = requests.get(url, **request_options)
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
        cache_control = response.headers.get("Cache-Control", "")
        return payload, cache_control if isinstance(cache_control, str) else ""

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
        payload, cache_control = _fetch_json(url, params)
    except BaseException as exc:
        with _cache_lock:
            future.set_exception(exc)
            del _in_flight[key]
        raise

    ttl = _cache_ttl(url, cache_control)
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
        **(
            {"timezone": result["timezone"]}
            if isinstance(result.get("timezone"), str)
            else {}
        ),
    }


def reverse_geocode(latitude: float, longitude: float) -> str:
    """Return a city or locality name for validated coordinates."""

    payload = _get_json(
        REVERSE_GEOCODING_URL,
        {"latitude": latitude, "longitude": longitude, "localityLanguage": "en"},
    )
    for key in ("city", "locality"):
        name = payload.get(key)
        if isinstance(name, str) and name.strip():
            return name.strip()
    raise WeatherServiceError("The reverse-geocoding provider returned no city or locality.")


def resolve_coordinates(
    latitude: Any,
    longitude: Any,
    timezone_name: Any = None,
) -> dict[str, Any]:
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

    location = {
        "name": "Your location",
        "region": "",
        "country": "Browser location",
        "latitude": float(latitude),
        "longitude": float(longitude),
    }
    if isinstance(timezone_name, str) and timezone_name.strip():
        location["timezone"] = timezone_name.strip()
    return location


def _forecast_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise WeatherServiceError("The forecast provider returned an invalid timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WeatherServiceError("The forecast provider returned an invalid timestamp.") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _location_timezone(value: Any) -> tuple[str, Any]:
    if isinstance(value, str) and value.strip():
        try:
            return value.strip(), ZoneInfo(value.strip())
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise WeatherServiceError("The location returned an invalid timezone.") from exc
    return "UTC", timezone.utc


def _period_data(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    for period_name, period_hours in (
        ("next_1_hours", 1),
        ("next_6_hours", 6),
        ("next_12_hours", 12),
    ):
        period = data.get(period_name)
        if isinstance(period, dict):
            details = period.get("details")
            if isinstance(details, dict):
                return period, details, period_hours
    return {}, {}, 1


def _condition_for_entry(entry: dict[str, Any], cloud_fraction: Any) -> str:
    period, _, _ = _period_data(entry["data"])
    summary = period.get("summary")
    symbol = summary.get("symbol_code") if isinstance(summary, dict) else None
    return _weather_condition(symbol, cloud_fraction)


def get_forecast(location: dict[str, Any]) -> dict[str, Any]:
    coordinate_scale = 10**FORECAST_CACHE_COORDINATE_PRECISION
    latitude = int(float(location["latitude"]) * coordinate_scale) / coordinate_scale
    longitude = int(float(location["longitude"]) * coordinate_scale) / coordinate_scale
    payload = _get_json(FORECAST_URL, {"lat": latitude, "lon": longitude})
    properties = payload.get("properties")
    timeseries = properties.get("timeseries") if isinstance(properties, dict) else None
    if not isinstance(timeseries, list) or len(timeseries) < 24:
        raise WeatherServiceError("The forecast provider returned incomplete timeseries data.")

    timezone_name, local_timezone = _location_timezone(location.get("timezone"))
    rows = []
    for entry in timeseries:
        if not isinstance(entry, dict) or not isinstance(entry.get("data"), dict):
            raise WeatherServiceError("The forecast provider returned an invalid forecast step.")
        data = entry["data"]
        instant = data.get("instant")
        details = instant.get("details") if isinstance(instant, dict) else None
        if not isinstance(details, dict):
            raise WeatherServiceError("The forecast provider returned incomplete forecast details.")
        required_fields = (
            "air_temperature",
            "relative_humidity",
            "wind_speed",
            "wind_from_direction",
            "cloud_area_fraction",
        )
        if any(field not in details for field in required_fields):
            raise WeatherServiceError("The forecast provider returned incomplete forecast details.")

        forecast_time = _forecast_datetime(entry.get("time"))
        period, _, period_hours = _period_data(data)
        precipitation = None
        for period_name in ("next_1_hours", "next_6_hours", "next_12_hours"):
            candidate = data.get(period_name)
            candidate_details = candidate.get("details") if isinstance(candidate, dict) else None
            if (
                isinstance(candidate_details, dict)
                and candidate_details.get("precipitation_amount") is not None
            ):
                precipitation = candidate_details["precipitation_amount"]
                break
        if precipitation is not None:
            precipitation = _number(precipitation, "precipitation")
        cloud_fraction = _percentage(details["cloud_area_fraction"], "cloud cover")
        local_time = forecast_time.astimezone(local_timezone)
        rows.append(
            {
                "local_time": local_time,
                "temperature": _number(details["air_temperature"], "temperature"),
                "humidity": _percentage(details["relative_humidity"], "humidity"),
                "wind": _number(details["wind_speed"], "wind speed") * 3.6,
                "wind_direction": _number(
                    details["wind_from_direction"],
                    "wind direction",
                ),
                "cloud_fraction": cloud_fraction,
                "precipitation": precipitation,
                "period_hours": period_hours,
                "condition": _condition_for_entry(entry, cloud_fraction),
                "symbol": (
                    period.get("summary", {}).get("symbol_code")
                    if isinstance(period.get("summary"), dict)
                    else None
                ),
                "apparent_temperature": details.get("apparent_air_temperature"),
            }
        )
    if not rows:
        raise WeatherServiceError("The forecast provider returned no forecast steps.")

    def format_local_time(value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M")

    day_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        day_groups.setdefault(row["local_time"].date().isoformat(), []).append(row)
    forecast = []
    for forecast_date, day_rows in list(day_groups.items())[:7]:
        precipitation_values = [
            row["precipitation"]
            for row in day_rows
            if row["precipitation"] is not None
        ]
        condition_counts: dict[str, int] = {}
        for row in day_rows:
            condition_counts[row["condition"]] = condition_counts.get(row["condition"], 0) + 1
        forecast.append(
            {
                "date": forecast_date,
                "high_c": max(row["temperature"] for row in day_rows),
                "low_c": min(row["temperature"] for row in day_rows),
                "condition": max(condition_counts, key=condition_counts.get),
                "precipitation_mm": (
                    round(sum(precipitation_values), 2)
                    if len(precipitation_values) == len(day_rows)
                    else None
                ),
            }
        )
    if len(forecast) < 7:
        raise WeatherServiceError("The forecast provider returned fewer than seven forecast days.")

    current = rows[0]
    hourly_forecast = [
        {
            "time": format_local_time(row["local_time"]),
            "temperature_c": row["temperature"],
            "precipitation_mm": row["precipitation"],
            "period_hours": row["period_hours"],
            "wind_kmh": row["wind"],
            "wind_direction_degrees": row["wind_direction"],
            "cloud_cover_pct": row["cloud_fraction"],
            "condition": row["condition"],
        }
        for row in rows
    ]
    if len(hourly_forecast) < 24:
        raise WeatherServiceError("The forecast provider returned fewer than 24 upcoming forecasts.")

    symbol = current["symbol"]
    is_day = not isinstance(symbol, str) or not symbol.endswith("_night")
    apparent_temperature = current["apparent_temperature"]
    if apparent_temperature is not None:
        apparent_temperature = _number(apparent_temperature, "feels-like temperature")
    return {
        "timezone": timezone_name,
        "current": {
            "temperature_c": current["temperature"],
            "feels_like_c": apparent_temperature,
            "condition": current["condition"],
            "humidity_pct": current["humidity"],
            "wind_kmh": current["wind"],
            "wind_direction_degrees": current["wind_direction"],
            "cloud_cover_pct": current["cloud_fraction"],
            "precipitation_mm": current["precipitation"],
            "period_hours": current["period_hours"],
            "forecast_at": format_local_time(current["local_time"]),
            "is_day": is_day,
        },
        "forecast": forecast,
        "hourly": hourly_forecast[:24],
        "hourly_forecast": hourly_forecast,
    }
