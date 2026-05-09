#!/usr/bin/env python3
"""
weather_cli.py — Terminal weather app powered by Open-Meteo (no API key).

Usage:
    weather now [location]              Current conditions
    weather forecast [location] [days]  Multi-day forecast (1–16 days)
    weather config --location CITY      Set default city
    weather config --unit   UNIT        Set temperature unit (metric / imperial)
    weather config --show               Display current config

Examples:
    python weather_cli.py now London
    python weather_cli.py forecast Tokyo 5
    python weather_cli.py config --location "New York" --unit imperial

Requires:  rich, aiohttp   (pip install rich aiohttp)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Fix Unicode rendering on Windows terminals (e.g. CP1252 can't handle emoji).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
# Config
# ---------------------------------------------------------------------------

@dataclass
class WeatherConfig:
    location: str = ""
    unit: str = "metric"  # metric | imperial

    @classmethod
    def load(cls) -> WeatherConfig:
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return cls(
                    location=data.get("location", ""),
                    unit=data.get("unit", "metric"),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps({"location": self.location, "unit": self.unit}, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# IP geolocation
# ---------------------------------------------------------------------------

IPINFO_URL = "https://ipinfo.io/json"


async def detect_location_from_ip(session: aiohttp.ClientSession) -> Optional[str]:
    """Detect the user's city from their IP address using ipinfo.io (free, no API key)."""
    try:
        async with session.get(IPINFO_URL, timeout=5) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            city = data.get("city", "")
            if city:
                return city
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def geocode(session: aiohttp.ClientSession, city: str) -> Optional[tuple[float, float, str]]:
    """Resolve city name → (lat, lon, display_name).  Returns None if not found."""
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


def _build_current_params(lat: float, lon: float, unit: str) -> dict:
    temp_key = "temperature_2m"
    wind_key = "wind_speed_10m"
    return {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            temp_key,
            "relative_humidity_2m",
            "apparent_temperature",
            "weather_code",
            wind_key,
            "wind_direction_10m",
        ]),
        "temperature_unit": "celsius" if unit == "metric" else "fahrenheit",
        "wind_speed_unit": "kmh" if unit == "metric" else "mph",
        "timezone": "auto",
        "forecast_days": 1,
    }


async def fetch_current(
    session: aiohttp.ClientSession, lat: float, lon: float, unit: str
) -> Optional[dict]:
    """Fetch current-conditions block from Open-Meteo."""
    params = _build_current_params(lat, lon, unit)
    async with session.get(FORECAST_URL, params=params) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


def _build_forecast_params(lat: float, lon: float, days: int, unit: str) -> dict:
    temp_key = "temperature_2m_max" if unit == "metric" else "temperature_2m_max"
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code",
            "wind_speed_10m_max",
        ]),
        "temperature_unit": "celsius" if unit == "metric" else "fahrenheit",
        "wind_speed_unit": "kmh" if unit == "metric" else "mph",
        "timezone": "auto",
        "forecast_days": days,
    }


async def fetch_forecast(
    session: aiohttp.ClientSession, lat: float, lon: float, days: int, unit: str
) -> Optional[dict]:
    """Fetch daily-forecast block from Open-Meteo."""
    params = _build_forecast_params(lat, lon, days, unit)
    async with session.get(FORECAST_URL, params=params) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

console = Console(force_terminal=True)


def _weather_emoji_desc(code: int) -> tuple[str, str]:
    return WMO_CODES.get(code, ("❓", "Unknown"))


def _unit_suffix(unit: str) -> str:
    return "°C" if unit == "metric" else "°F"


def _wind_unit(unit: str) -> str:
    return "km/h" if unit == "metric" else "mph"


def _wind_direction_arrow(degrees: int) -> str:
    """Return a compass arrow for a wind direction in degrees."""
    arrows = "↑↗→↘↓↙←↖"
    idx = round(degrees / 45) % 8
    return arrows[idx]


def show_current(city_display: str, data: dict, unit: str) -> None:
    """Render current conditions as a rich Panel."""
    c = data.get("current", {})
    temp = c.get("temperature_2m", "?")
    feels = c.get("apparent_temperature", "?")
    humidity = c.get("relative_humidity_2m", "?")
    wind_speed = c.get("wind_speed_10m", "?")
    wind_dir = c.get("wind_direction_10m", 0)
    code = c.get("weather_code", 0)

    emoji, desc = _weather_emoji_desc(int(code))
    dir_arrow = _wind_direction_arrow(int(wind_dir))
    u_temp = _unit_suffix(unit)
    u_wind = _wind_unit(unit)

    body = Text()
    body.append(f"{emoji}  {desc}\n", style="bold")
    body.append(f"Temperature:        {temp}{u_temp}\n")
    body.append(f"Feels like:         {feels}{u_temp}\n")
    body.append(f"Humidity:           {humidity}%\n")
    body.append(f"Wind:               {wind_speed} {u_wind}  {dir_arrow} ({wind_dir}°)\n")

    panel = Panel(
        body,
        title=f"[bold]{city_display}[/]",
        subtitle="Open-Meteo",
        border_style="cyan",
    )
    console.print(panel)


def show_forecast(city_display: str, data: dict, unit: str) -> None:
    """Render a multi-day forecast as a rich Table."""
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        console.print("[red]No forecast data returned.[/]")
        return

    u_temp = _unit_suffix(unit)
    u_wind = _wind_unit(unit)

    table = Table(
        title=f"[bold]{city_display}[/] — Forecast",
        box=None,
        header_style="bold cyan",
        title_style="bold",
    )
    table.add_column("Date", style="cyan")
    table.add_column("Weather", justify="center")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Rain", justify="right")
    table.add_column("Wind", justify="right")

    for i, date in enumerate(dates):
        code = daily.get("weather_code", [0] * len(dates))[i]
        tmax = daily.get("temperature_2m_max", ["?"] * len(dates))[i]
        tmin = daily.get("temperature_2m_min", ["?"] * len(dates))[i]
        precip = daily.get("precipitation_sum", [0] * len(dates))[i]
        wind = daily.get("wind_speed_10m_max", ["?"] * len(dates))[i]

        emoji, desc = _weather_emoji_desc(int(code))
        table.add_row(
            str(date),
            f"{emoji} {desc}",
            f"{tmax}{u_temp}",
            f"{tmin}{u_temp}",
            f"{precip} mm",
            f"{wind} {u_wind}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

async def cmd_now(args: argparse.Namespace, config: WeatherConfig) -> None:
    location = args.location or config.location
    if not location:
        async with aiohttp.ClientSession() as session:
            detected = await detect_location_from_ip(session)
            if detected:
                location = detected
                console.print(f"[dim]Auto-detected location: {location}[/]")
            else:
                console.print(
                    "[red]No location provided.  Use [bold]weather now CITY[/] or set a default with "
                    "[bold]weather config --location CITY[/].[/]"
                )
                sys.exit(1)

    unit = config.unit

    async with aiohttp.ClientSession() as session:
        geo = await geocode(session, location)
        if geo is None:
            console.print(f"[red]Location not found:[/] {location}")
            sys.exit(1)
        lat, lon, display = geo

        data = await fetch_current(session, lat, lon, unit)
        if data is None:
            console.print("[red]Failed to fetch weather data.[/]")
            sys.exit(1)

    show_current(display, data, unit)


async def cmd_forecast(args: argparse.Namespace, config: WeatherConfig) -> None:
    location = args.location or config.location
    if not location:
        async with aiohttp.ClientSession() as session:
            detected = await detect_location_from_ip(session)
            if detected:
                location = detected
                console.print(f"[dim]Auto-detected location: {location}[/]")
            else:
                console.print(
                    "[red]No location provided.  Use [bold]weather forecast CITY [DAYS][/] or set a default "
                    "with [bold]weather config --location CITY[/].[/]"
                )
                sys.exit(1)

    days = args.days
    if not 1 <= days <= 16:
        console.print("[red]Days must be between 1 and 16.[/]")
        sys.exit(1)

    unit = config.unit

    async with aiohttp.ClientSession() as session:
        geo = await geocode(session, location)
        if geo is None:
            console.print(f"[red]Location not found:[/] {location}")
            sys.exit(1)
        lat, lon, display = geo

        data = await fetch_forecast(session, lat, lon, days, unit)
        if data is None:
            console.print("[red]Failed to fetch forecast data.[/]")
            sys.exit(1)

    show_forecast(display, data, unit)


def cmd_config(args: argparse.Namespace, config: WeatherConfig) -> None:
    changed = False

    if args.location is not None:
        config.location = args.location
        changed = True
        console.print(f"[green]Default location set to:[/] {config.location}")

    if args.unit is not None:
        if args.unit not in ("metric", "imperial"):
            console.print("[red]Unit must be 'metric' or 'imperial'.[/]")
            sys.exit(1)
        config.unit = args.unit
        changed = True
        console.print(f"[green]Unit set to:[/] {config.unit}")

    if changed:
        config.save()

    if args.show or not changed:
        unit_label = "°C, km/h" if config.unit == "metric" else "°F, mph"
        console.print()
        console.print(Panel(
            f"[bold]Location:[/] {config.location or '(not set)'}\n"
            f"[bold]Unit:[/]     {config.unit}  ({unit_label})",
            title="[bold]Weather CLI Config[/]",
            border_style="green",
        ))


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weather",
        description="Terminal weather app — Open-Meteo, no API key needed.",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # ---- now ----
    p_now = sub.add_parser("now", help="Current conditions")
    p_now.add_argument("location", nargs="?", default=None, help="City name")

    # ---- forecast ----
    p_fc = sub.add_parser("forecast", help="Multi-day forecast")
    p_fc.add_argument("location", nargs="?", default=None, help="City name")
    p_fc.add_argument("days", nargs="?", type=int, default=7, help="Number of days (1–16, default 7)")

    # ---- config ----
    p_cfg = sub.add_parser("config", help="View or change defaults")
    p_cfg.add_argument("--location", "-l", default=None, help="Set default city")
    p_cfg.add_argument("--unit", "-u", default=None, choices=("metric", "imperial"), help="Temperature unit")
    p_cfg.add_argument("--show", "-s", action="store_true", default=False, help="Show current config")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default to "forecast" when no subcommand is given
    if args.command is None:
        args.command = "forecast"
        args.location = None
        args.days = 7

    config = WeatherConfig.load()

    if args.command == "now":
        asyncio.run(cmd_now(args, config))
    elif args.command == "forecast":
        asyncio.run(cmd_forecast(args, config))
    elif args.command == "config":
        cmd_config(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
