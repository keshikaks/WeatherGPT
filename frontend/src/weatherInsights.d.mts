export interface HourlyWeather {
  time: string;
  temperature_c: number;
  precipitation_mm: number | null;
  wind_kmh: number;
  wind_direction_degrees: number;
  cloud_cover_pct: number;
  condition: string;
}

export interface DailyWeather {
  date: string;
  high_c: number;
  low_c: number;
  condition: string;
  precipitation_mm: number | null;
}

export interface ActivityWindow {
  time: string;
  temperature_c: number;
  precipitation_mm: number | null;
  wind_kmh: number;
  condition: string;
}

export function getActivityPlans(hours: HourlyWeather[]): Array<{
  activity: string;
  criteria: {
    maxPrecipitationMm: number;
    maxWindKmh: number;
    minTemperatureC: number;
    maxTemperatureC: number;
  };
  windows: ActivityWindow[];
}>;
export function getDailyScale(days: DailyWeather[]): { min: number; max: number };
export function formatLocationDateTime(
  value: string,
  timezone?: string,
  options?: Intl.DateTimeFormatOptions,
): string;
export function formatLocationDate(value: string): { weekday: string; monthDay: string } | null;
