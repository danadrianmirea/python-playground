#!/usr/bin/env python3
"""
forecast.py — Displays a calendar for the current week and the next week
with temperatures for each day.

Uses the Open-Meteo API (no API key required).

Usage:
    python forecast.py [location]

If no location is given, reads from ~/.weather_cli_config.json (shared with weather_cli.py).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Fix Unicode rendering on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import aiohttp
import asyncio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CONFIG_PATH = Path.home() / ".weather_cli_config.json"

# WMO weather-interpretation codes → (emoji, description)
WMO_CODES: dict[int, tuple[str, str]] = {
    0:  ("☀️",  "Clear sky"),
    1:  ("🌤️",  "Mainly clear"),
    2:  ("⛅",  "Partly cloudy"),
    3:  ("☁️",  "Overcast"),
    45: ("🌫️",  "Fog"),
    48: ("🌫️",  "Depositing rime fog"),
    51: ("🌦️",  "Light drizzle"),
    53: ("🌦️",  "Moderate drizzle"),
    55: ("🌧️",  "Dense drizzle"),
    56: ("🌧️",  "Light freezing drizzle"),
    57: ("🌧️",  "Dense freezing drizzle"),
    61: ("🌧️",  "Slight rain"),
    63: ("🌧️",  "Moderate rain"),
    65: ("🌧️",  "Heavy rain"),
    66: ("❄️",  "Light freezing rain"),
    67: ("❄️",  "Heavy freezing rain"),
    71: ("🌨️",  "Slight snowfall"),
    73: ("🌨️",  "Moderate snowfall"),
    75: ("❄️",  "Heavy snowfall"),
    77: ("🌨️",  "Snow grains"),
    80: ("🌦️",  "Slight rain showers"),
    81: ("🌧️",  "Moderate rain showers"),
    82: ("⛈️",  "Violent rain showers"),
    85: ("🌨️",  "Slight snow showers"),
    86: ("❄️",  "Heavy snow showers"),
    95: ("⛈️",  "Thunderstorm"),
    96: ("⛈️",  "Thunderstorm with slight hail"),
    99: ("⛈️",  "Thunderstorm with heavy hail"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_week_days() -> list[date]:
    """Return a list of 7 dates from Monday to Sunday of the current week."""
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


def get_next_week_days() -> list[date]:
    """Return a list of 7 dates from next Monday to next Sunday."""
    today = date.today()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # if today is Monday, go to next Monday
    next_monday = today + timedelta(days=days_until_monday)
    return [next_monday + timedelta(days=i) for i in range(7)]


def load_config_location() -> str:
    """Try to load a default location from the shared config file."""
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return data.get("location", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


async def geocode(session: aiohttp.ClientSession, city: str) -> Optional[tuple[float, float, str]]:
    """Resolve city name → (lat, lon, display_name)."""
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    async with session.get(GEOCODING_URL, params=params) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
        results = data.get("results")
        if not results:
            return None
        r = results[0]
        name_parts = [r.get("name", "")]
        if r.get("admin1"):
            name_parts.append(r["admin1"])
        if r.get("country"):
            name_parts.append(r["country"])
        display = ", ".join(filter(None, name_parts))
        return float(r["latitude"]), float(r["longitude"]), display


async def fetch_forecast(
    session: aiohttp.ClientSession, lat: float, lon: float
) -> Optional[dict]:
    """Fetch daily forecast for the next 14 days from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "temperature_unit": "celsius",
        "timezone": "auto",
        "forecast_days": 14,
    }
    async with session.get(FORECAST_URL, params=params) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


def _weather_emoji(code: int) -> str:
    return WMO_CODES.get(code, ("❓", "Unknown"))[0]


# ---------------------------------------------------------------------------
# Calendar display
# ---------------------------------------------------------------------------

def build_forecast_map(forecast: Optional[dict]) -> dict[str, tuple[float, float, int]]:
    """Build a lookup dict: date_str -> (high, low, weather_code)."""
    forecast_map: dict[str, tuple[float, float, int]] = {}
    if forecast is None:
        return forecast_map
    daily = forecast.get("daily", {})
    dates_api = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    codes = daily.get("weather_code", [])
    for i, d in enumerate(dates_api):
        forecast_map[d] = (highs[i], lows[i], codes[i])
    return forecast_map


def print_week_calendar(
    week_days: list[date],
    forecast_map: dict[str, tuple[float, float, int]],
    title: str,
    today: date,
) -> None:
    """Print a single week calendar grid."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cell_w = 7

    def render_row(cells: list[str]) -> str:
        return "│ " + " │ ".join(c.ljust(cell_w - 2) for c in cells) + " │"

    def top_border() -> str:
        return "┌" + "┬".join("─" * cell_w for _ in range(7)) + "┐"

    def mid_border() -> str:
        return "├" + "┼".join("─" * cell_w for _ in range(7)) + "┤"

    def bot_border() -> str:
        return "└" + "┴".join("─" * cell_w for _ in range(7)) + "┘"

    # Row 1: Day names
    day_row = render_row(day_names)

    # Row 2: Day numbers (highlight today)
    day_num_cells: list[str] = []
    for d in week_days:
        label = str(d.day)
        if d == today:
            label = f"*{d.day}"
        day_num_cells.append(label)
    num_row = render_row(day_num_cells)

    # Rows 3-5: Emoji, High, Low
    emoji_cells: list[str] = []
    high_cells: list[str] = []
    low_cells: list[str] = []
    for d in week_days:
        ds = d.isoformat()
        if ds in forecast_map:
            h, lo, c = forecast_map[ds]
            emoji_cells.append(_weather_emoji(c))
            high_cells.append(f"{h:.0f}°C")
            low_cells.append(f"{lo:.0f}°C")
        else:
            emoji_cells.append(" ")
            high_cells.append("--")
            low_cells.append("--")

    emoji_row = render_row(emoji_cells)
    high_row = render_row(high_cells)
    low_row = render_row(low_cells)

    print(f"  {title}")
    print(top_border())
    print(day_row)
    print(mid_border())
    print(num_row)
    print(mid_border())
    print(emoji_row)
    print(high_row)
    print(low_row)
    print(bot_border())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(location: str) -> None:
    # Resolve location
    if not location:
        location = load_config_location()
    if not location:
        print("Usage: python forecast.py [location]")
        print("       (or set a default location with weather_cli.py config --location CITY)")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        geo = await geocode(session, location)
        if geo is None:
            print(f"Location not found: {location}")
            sys.exit(1)
        lat, lon, display = geo

        forecast = await fetch_forecast(session, lat, lon)
        if forecast is None:
            print("Failed to fetch forecast data.")
            sys.exit(1)

    today = date.today()
    forecast_map = build_forecast_map(forecast)

    current_week = get_current_week_days()
    next_week = get_next_week_days()

    # Header
    header = f"Weather Forecast — {display}"
    sep = "=" * len(header)
    print()
    print(header)
    print(sep)
    print()

    print_week_calendar(current_week, forecast_map, "This Week", today)
    print_week_calendar(next_week, forecast_map, "Next Week", today)

    print(f"  * = today ({today.isoformat()})")
    print()


def main() -> None:
    location = sys.argv[1] if len(sys.argv) > 1 else ""
    asyncio.run(main_async(location))


if __name__ == "__main__":
    main()