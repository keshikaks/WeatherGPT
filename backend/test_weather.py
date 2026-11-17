import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Event
from unittest.mock import Mock, patch

import requests
import weather

from weather import (
    FORECAST_URL,
    GEOCODING_URL,
    InvalidCoordinatesError,
    LocationNotFoundError,
    RATE_LIMIT_MESSAGE,
    WeatherRateLimitError,
    WeatherServiceError,
    _current_rain_probability,
    _get_json,
    get_forecast,
    resolve_coordinates,
    resolve_location,
)


class OpenMeteoErrorTests(unittest.TestCase):
    def setUp(self):
        with weather._cache_lock:
            weather._response_cache.clear()

    def test_resolves_valid_coordinates(self):
        self.assertEqual(
            resolve_coordinates(12.5, -45.25),
            {
                "name": "Your location",
                "region": "",
                "country": "Browser location",
                "latitude": 12.5,
                "longitude": -45.25,
            },
        )

    def test_rejects_invalid_coordinates(self):
        for latitude, longitude in (
            (90.1, 0),
            (-90.1, 0),
            (0, 180.1),
            (0, -180.1),
            (float("nan"), 0),
            (0, float("inf")),
            (True, 0),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaises(InvalidCoordinatesError):
                    resolve_coordinates(latitude, longitude)

    @patch("weather.requests.get", side_effect=requests.Timeout)
    def test_timeout_is_reported_as_weather_service_error(self, get):
        with self.assertRaisesRegex(WeatherServiceError, "too long"):
            resolve_location("Somewhere")
        get.assert_called_once()

    @patch("weather.requests.get")
    def test_empty_geocoding_results_are_not_invented(self, get):
        response = Mock()
        response.json.return_value = {"results": []}
        get.return_value = response

        with self.assertRaises(LocationNotFoundError):
            resolve_location("Somewhere")

    @patch("weather.requests.get")
    def test_malformed_geocoding_result_is_reported(self, get):
        response = Mock()
        response.json.return_value = {"results": [{"name": "Somewhere"}]}
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "incomplete location"):
            resolve_location("Somewhere")

    @patch("weather.requests.get")
    def test_identical_provider_requests_use_cached_response(self, get):
        response = Mock()
        response.json.return_value = {"results": [{"name": "London"}]}
        get.return_value = response
        params = {"name": "London", "count": 1}

        first = _get_json(GEOCODING_URL, params)
        first["results"].clear()
        second = _get_json(GEOCODING_URL, params)

        self.assertEqual(second, {"results": [{"name": "London"}]})
        get.assert_called_once()

    @patch("weather.requests.get")
    def test_nearby_forecast_coordinates_share_cache_without_rounding_request(self, get):
        response = Mock()
        response.json.return_value = {"forecast": "nearby"}
        get.return_value = response
        first_params = {"latitude": 60.17123, "longitude": 24.94123, "timezone": "auto"}
        nearby_params = {"latitude": 60.17134, "longitude": 24.94134, "timezone": "auto"}

        first = _get_json(FORECAST_URL, first_params)
        nearby = _get_json(FORECAST_URL, nearby_params)
        farther = _get_json(
            FORECAST_URL,
            {"latitude": 60.18123, "longitude": 24.94123, "timezone": "auto"},
        )

        self.assertEqual(nearby, first)
        self.assertEqual(farther, first)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"], first_params)
        self.assertEqual(
            get.call_args_list[1].kwargs["params"]["latitude"],
            60.18123,
        )

    @patch("weather.requests.get")
    def test_concurrent_identical_provider_requests_share_one_call(self, get):
        request_started = Event()
        waiter_joined = Event()
        release_response = Event()
        response = Mock()
        response.json.return_value = {"results": [{"name": "London"}]}

        class ObservableFuture(Future):
            def result(self, timeout=None):
                waiter_joined.set()
                return super().result(timeout)

        def delayed_response(*args, **kwargs):
            request_started.set()
            if not release_response.wait(timeout=5):
                raise TimeoutError("Test did not release the mocked response.")
            return response

        get.side_effect = delayed_response
        params = {"name": "London", "count": 1}

        with patch("weather.Future", ObservableFuture):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(_get_json, GEOCODING_URL, params)
                self.assertTrue(request_started.wait(timeout=5))
                second = executor.submit(_get_json, GEOCODING_URL, params)
                self.assertTrue(waiter_joined.wait(timeout=5))
                release_response.set()
                self.assertEqual(first.result(timeout=5), second.result(timeout=5))

        get.assert_called_once()

    @patch("weather.time.sleep")
    @patch("weather.requests.get")
    def test_retries_429_using_retry_after(self, get, sleep):
        rate_limited = Mock(status_code=429, headers={"Retry-After": "0.25"})
        success = Mock(status_code=200)
        success.json.return_value = {"forecast": "ok"}
        get.side_effect = [rate_limited, success]

        payload = _get_json(FORECAST_URL, {"latitude": 1, "longitude": 2})

        self.assertEqual(payload, {"forecast": "ok"})
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0.25)

    @patch("weather.time.sleep")
    @patch("weather.requests.get")
    def test_repeated_429_uses_bounded_backoff_and_clear_error(self, get, sleep):
        get.return_value = Mock(status_code=429, headers={})

        with self.assertRaisesRegex(WeatherRateLimitError, RATE_LIMIT_MESSAGE):
            _get_json(FORECAST_URL, {"latitude": 1, "longitude": 2})

        self.assertEqual(get.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.5, 1.0],
        )

    @patch("weather.time.sleep")
    @patch("weather.requests.get")
    def test_does_not_retry_before_long_retry_after(self, get, sleep):
        get.return_value = Mock(status_code=429, headers={"Retry-After": "30"})

        with self.assertRaisesRegex(WeatherRateLimitError, RATE_LIMIT_MESSAGE):
            _get_json(FORECAST_URL, {"latitude": 1, "longitude": 2})

        get.assert_called_once()
        sleep.assert_not_called()

    def test_missing_current_rain_probability_is_not_reported_as_zero(self):
        payload = {
            "current": {"time": "2026-10-05T12:00"},
            "hourly": {
                "time": ["2026-10-05T12:00"],
                "precipitation_probability": [None],
            },
        }
        with self.assertRaisesRegex(WeatherServiceError, "did not return"):
            _current_rain_probability(payload)

    @staticmethod
    def _forecast_payload(hourly_count=48, daily_count=7):
        start = datetime.fromisoformat("2026-10-05T12:00")
        return {
            "timezone": "Europe/Helsinki",
            "current": {
                "time": start.isoformat(),
                "temperature_2m": 14.5,
                "apparent_temperature": 13.0,
                "relative_humidity_2m": 63,
                "wind_speed_10m": 12.0,
                "weather_code": 2,
                "is_day": 1,
            },
            "hourly": {
                "time": [(start + timedelta(hours=index)).isoformat() for index in range(hourly_count)],
                "temperature_2m": [14.5] * hourly_count,
                "precipitation_probability": [20] * hourly_count,
                "precipitation": [0.0] * hourly_count,
                "wind_speed_10m": [12.0] * hourly_count,
                "weather_code": [2] * hourly_count,
            },
            "daily": {
                "time": [
                    (start.date() + timedelta(days=index)).isoformat()
                    for index in range(daily_count)
                ],
                "weather_code": [2] * daily_count,
                "temperature_2m_max": [18.0] * daily_count,
                "temperature_2m_min": [10.0] * daily_count,
                "precipitation_probability_max": [20] * daily_count,
            },
        }

    @patch("weather.requests.get")
    def test_forecast_includes_timezone_and_next_24_hour_insights(self, get):
        response = Mock()
        response.json.return_value = self._forecast_payload()
        get.return_value = response
        location = {"latitude": 60.17, "longitude": 24.94}

        forecast = get_forecast(location)
        cached_forecast = get_forecast(location)

        self.assertEqual(forecast["timezone"], "Europe/Helsinki")
        self.assertEqual(len(forecast["forecast"]), 7)
        self.assertEqual(len(forecast["hourly"]), 24)
        self.assertEqual(len(forecast["hourly_forecast"]), 48)
        self.assertEqual(forecast["hourly"][0]["time"], "2026-10-05T12:00")
        self.assertEqual(forecast["hourly"][0]["rain_probability_pct"], 20)
        self.assertEqual(forecast["hourly"][0]["wind_kmh"], 12.0)
        self.assertEqual(forecast["hourly_forecast"][-1]["time"], "2026-10-07T11:00")
        self.assertEqual(get.call_args.kwargs["params"]["timezone"], "auto")
        self.assertEqual(cached_forecast, forecast)
        get.assert_called_once()

    @patch("weather.requests.get")
    def test_rejects_forecast_with_fewer_than_seven_days(self, get):
        response = Mock()
        response.json.return_value = self._forecast_payload(daily_count=6)
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "inconsistent daily data"):
            get_forecast({"latitude": 0, "longitude": 0})

    @patch("weather.requests.get")
    def test_rejects_forecast_with_fewer_than_24_upcoming_hours(self, get):
        response = Mock()
        response.json.return_value = self._forecast_payload(hourly_count=23)
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "incomplete hourly data"):
            get_forecast({"latitude": 0, "longitude": 0})

    @patch("weather.requests.get")
    def test_rejects_invalid_hourly_rain_probability(self, get):
        payload = self._forecast_payload()
        payload["hourly"]["precipitation_probability"][0] = 101
        response = Mock()
        response.json.return_value = payload
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "invalid hourly rain probability"):
            get_forecast({"latitude": 0, "longitude": 0})


if __name__ == "__main__":
    unittest.main()
