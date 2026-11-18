import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import Mock, patch

import requests
import weather

from weather import (
    FORECAST_CACHE_TTL_SECONDS,
    FORECAST_URL,
    GEOCODING_URL,
    InvalidCoordinatesError,
    LocationNotFoundError,
    RATE_LIMIT_MESSAGE,
    WeatherRateLimitError,
    WeatherServiceError,
    _get_json,
    get_forecast,
    resolve_coordinates,
    resolve_location,
)


class WeatherProviderTests(unittest.TestCase):
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

    @patch("weather.requests.get")
    def test_geocoding_keeps_open_meteo_timezone(self, get):
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "name": "Helsinki",
                    "country": "Finland",
                    "latitude": 60.17,
                    "longitude": 24.94,
                    "timezone": "Europe/Helsinki",
                }
            ]
        }
        get.return_value = response
        location = resolve_location("Helsinki")

        self.assertEqual(location["timezone"], "Europe/Helsinki")
        self.assertEqual(get.call_args.args[0], GEOCODING_URL)

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
    def test_forecast_cache_key_truncates_coordinates_to_four_decimal_places(self, get):
        response = Mock()
        response.json.return_value = {"forecast": "nearby"}
        get.return_value = response
        first_params = {"lat": 60.17123, "lon": 24.94123}
        nearby_params = {"lat": 60.17129, "lon": 24.94129}

        first = _get_json(FORECAST_URL, first_params)
        nearby = _get_json(FORECAST_URL, nearby_params)
        farther = _get_json(
            FORECAST_URL,
            {"lat": 60.18123, "lon": 24.94123},
        )

        self.assertEqual(nearby, first)
        self.assertEqual(farther, first)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"], first_params)
        self.assertEqual(
            get.call_args_list[1].kwargs["params"]["lat"],
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

    @staticmethod
    def _forecast_payload(step_count=216):
        start = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        return {
            "properties": {
                "meta": {"updated_at": "2026-10-05T10:00:00Z"},
                "timeseries": [
                    {
                        "time": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
                        "data": {
                            "instant": {
                                "details": {
                                    "air_temperature": 14.5 + (index % 12),
                                    "relative_humidity": 63,
                                    "wind_speed": 10 / 3.6,
                                    "wind_from_direction": 225,
                                    "cloud_area_fraction": 45,
                                }
                            },
                            "next_1_hours": {
                                "summary": {"symbol_code": "partlycloudy_day"},
                                "details": {
                                    "precipitation_amount": 0.1 if index == 0 else 0,
                                },
                            },
                        },
                    }
                    for index in range(step_count)
                ],
            }
        }

    @patch("weather.requests.get")
    def test_met_norway_response_maps_to_forecast_features(self, get):
        response = Mock()
        response.headers = {"Cache-Control": "max-age=900"}
        response.json.return_value = self._forecast_payload()
        get.return_value = response
        location = {
            "latitude": 60.17329,
            "longitude": 24.94129,
            "timezone": "Europe/Helsinki",
        }

        forecast = get_forecast(location)
        cached_forecast = get_forecast(location)

        self.assertEqual(forecast["timezone"], "Europe/Helsinki")
        self.assertEqual(len(forecast["forecast"]), 7)
        self.assertEqual(len(forecast["hourly"]), 24)
        self.assertEqual(len(forecast["hourly_forecast"]), 216)
        self.assertEqual(forecast["hourly"][0]["time"], "2026-10-05T15:00")
        self.assertEqual(forecast["hourly"][0]["precipitation_mm"], 0.1)
        self.assertEqual(forecast["hourly"][0]["wind_kmh"], 10.0)
        self.assertEqual(forecast["current"]["humidity_pct"], 63)
        self.assertEqual(forecast["current"]["wind_direction_degrees"], 225)
        self.assertEqual(forecast["current"]["cloud_cover_pct"], 45)
        self.assertIsNone(forecast["current"]["feels_like_c"])
        self.assertEqual(forecast["current"]["condition"], "Partly cloudy")
        self.assertEqual(forecast["forecast"][0]["high_c"], 22.5)
        self.assertEqual(get.call_args.args[0], FORECAST_URL)
        self.assertEqual(get.call_args.kwargs["params"], {"lat": 60.1732, "lon": 24.9412})
        self.assertEqual(
            get.call_args.kwargs["headers"]["User-Agent"],
            "WeatherGPT/1.0 (https://github.com/syntaxelite/WeatherGPT)",
        )
        self.assertEqual(cached_forecast, forecast)
        get.assert_called_once()

    @patch("weather.requests.get")
    def test_missing_precipitation_is_not_fabricated_as_zero(self, get):
        response = Mock()
        response.headers = {}
        payload = self._forecast_payload()
        del payload["properties"]["timeseries"][0]["data"]["next_1_hours"]["details"][
            "precipitation_amount"
        ]
        response.json.return_value = payload
        get.return_value = response

        forecast = get_forecast(
            {"latitude": 0, "longitude": 0, "timezone": "Europe/Helsinki"}
        )
        self.assertIsNone(forecast["hourly"][0]["precipitation_mm"])
        self.assertNotIn("rain_probability_pct", forecast["hourly"][0])

    @patch("weather.requests.get")
    def test_rejects_forecast_with_fewer_than_seven_days(self, get):
        response = Mock()
        response.headers = {}
        response.json.return_value = self._forecast_payload(step_count=5 * 24)
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "fewer than seven forecast days"):
            get_forecast({"latitude": 0, "longitude": 0})

    @patch("weather.requests.get")
    def test_rejects_incomplete_met_norway_response(self, get):
        payload = self._forecast_payload()
        del payload["properties"]["timeseries"][0]["data"]["instant"]["details"][
            "relative_humidity"
        ]
        response = Mock()
        response.headers = {}
        response.json.return_value = payload
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "incomplete forecast details"):
            get_forecast({"latitude": 0, "longitude": 0})

    @patch("weather.requests.get")
    def test_forecast_cache_honors_provider_cache_control(self, get):
        response = Mock()
        response.headers = {"Cache-Control": "public, max-age=14400"}
        response.json.return_value = self._forecast_payload()
        get.return_value = response

        get_forecast({"latitude": 1, "longitude": 2})
        cache_key = weather._cache_key(FORECAST_URL, {"lat": 1.0, "lon": 2.0})

        self.assertGreaterEqual(
            weather._response_cache[cache_key][0] - weather.time.monotonic(),
            4 * 60 * 60 - 1,
        )
        self.assertEqual(weather._cache_ttl(FORECAST_URL, "max-age=900"), FORECAST_CACHE_TTL_SECONDS)
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
