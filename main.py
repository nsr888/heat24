#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch last 7 days of hourly temperature & humidity for multiple locations,
compute per-hour averages across the week, derive Heat Index, and plot 24h charts
with activity windows, sunrise/sunset, and a "avoid the sun" band.

Notes:
- Comments are in English (as requested).
- Uses Open-Meteo's openmeteo-requests client with caching & retry.
"""

import datetime as dt
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import openmeteo_requests
import requests_cache
from retry_requests import retry


# -----------------------------
# Configuration
# -----------------------------
# Define locations as a dictionary: name -> (latitude, longitude)
LOCATIONS = {
    "Paphos": (34.7768, 32.4245),
    "Palemi": (34.88593, 32.50657),
    "Pana Panagia": (34.91901721271778, 32.630531461579665),
    "Limmasol": (34.7071, 33.0226),
    "Nicosia": (35.1856, 33.3823),
    "Larnaca": (34.9190, 33.6232),
    "Famagusta": (35.1264, 33.9197),
    "Troodos": (34.9886, 32.8662),
}

# Local storage file path template (will be formatted with location name)
STORAGE_FILE_TEMPLATE = "heat_data_cache_{}.json"


# -----------------------------
# Heat Index (NOAA) in °C
# -----------------------------
def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Compute Heat Index in Celsius from air temperature in Celsius and RH in %.
    Formula adapted from NOAA/Steadman's regression, operating in Fahrenheit internally.
    """
    # Convert to Fahrenheit
    T = temp_c * 9.0 / 5.0 + 32.0

    # Core regression (vectorized)
    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * rh
        - 0.22475541 * T * rh
        - 6.83783e-3 * T * T
        - 5.481717e-2 * rh * rh
        + 1.22874e-3 * T * T * rh
        + 8.5282e-4 * T * rh * rh
        - 1.99e-6 * T * T * rh * rh
    )

    # Adjustment terms (apply where conditions hold)
    adj = np.zeros_like(HI)

    mask_low_rh = (rh < 13) & (T >= 80) & (T <= 112)
    adj[mask_low_rh] -= ((13 - rh[mask_low_rh]) / 4.0) * np.sqrt(
        (17.0 - np.abs(T[mask_low_rh] - 95.0)) / 17.0
    )

    mask_high_rh = (rh > 85) & (T >= 80) & (T <= 87)
    adj[mask_high_rh] += ((rh[mask_high_rh] - 85.0) / 10.0) * (
        (87.0 - T[mask_high_rh]) / 5.0
    )

    HI += adj

    # Convert back to Celsius
    return (HI - 32.0) * 5.0 / 9.0


def save_to_storage(data: dict, location_name: str) -> None:
    """Save data to local storage with current timestamp."""
    storage_file = STORAGE_FILE_TEMPLATE.format(location_name)
    storage_data = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data": data,
    }
    with open(storage_file, "w") as f:
        json.dump(storage_data, f)


def load_from_storage(location_name: str) -> dict | None:
    """Load data from local storage if it exists and is less than 24 hours old."""
    storage_file = STORAGE_FILE_TEMPLATE.format(location_name)
    if not os.path.exists(storage_file):
        return None

    try:
        with open(storage_file, "r") as f:
            storage_data = json.load(f)

        # Check if data is less than 24 hours old
        timestamp = dt.datetime.fromisoformat(storage_data["timestamp"])
        if dt.datetime.now(dt.timezone.utc) - timestamp < dt.timedelta(hours=24):
            return storage_data["data"]
        else:
            # Delete expired cache file
            os.remove(storage_file)
            return None
    except (json.JSONDecodeError, KeyError, ValueError):
        # If there's an error reading the file, remove it
        if os.path.exists(storage_file):
            os.remove(storage_file)
        return None


# -----------------------------
# Fetch data from Open-Meteo
# -----------------------------
def fetch_data(lat: float, lon: float) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """Fetch hourly temperature & RH for last 7 days + sunrise/sunset daily; return df (local time) and tz name."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["sunrise", "sunset"],
        "past_days": 7,
        "forecast_days": 0,
        "timezone": "auto",  # local timezone
    }

    responses = client.weather_api(url, params=params)
    resp = responses[0]

    tz_name = resp.Timezone().decode("utf-8")  # e.g., "Asia/Nicosia"

    # Hourly block
    hourly = resp.Hourly()
    t_values = hourly.Variables(0).ValuesAsNumpy()  # temperature_2m
    rh_values = hourly.Variables(1).ValuesAsNumpy()  # relative_humidity_2m

    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(tz=tz_name)

    hourly_df = pd.DataFrame(
        {
            "time": time_index,
            "temperature_c": t_values,
            "rh_pct": rh_values,
        }
    ).set_index("time")

    # Daily sunrise/sunset (for median over last week)
    daily = resp.Daily()
    # Use ValuesInt64AsNumpy for sunrise/sunset as they are unix timestamps
    sunrise_times = daily.Variables(0).ValuesInt64AsNumpy()  # sunrise
    sunset_times = daily.Variables(1).ValuesInt64AsNumpy()  # sunset

    # Convert to datetime objects
    sunrise_dates = pd.to_datetime(sunrise_times, unit="s", utc=True).tz_convert(
        tz_name
    )
    sunset_dates = pd.to_datetime(sunset_times, unit="s", utc=True).tz_convert(tz_name)

    daily_df = pd.DataFrame({"sunrise": sunrise_dates, "sunset": sunset_dates})

    # Keep only the last full 7 days (exclude any future hours if present)
    now_local = pd.Timestamp.now(tz_name)
    hourly_df = hourly_df[hourly_df.index <= now_local]

    return hourly_df, tz_name, daily_df


# -----------------------------
# Aggregate per hour-of-day
# -----------------------------
def per_hour_means(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean temperature and RH for each hour-of-day across the last 7 days."""
    df = hourly_df.copy()
    df["hod"] = df.index.hour
    agg = df.groupby("hod")[["temperature_c", "rh_pct"]].mean()
    # For plotting, map HOD to a reference date (today local) to get a continuous 24h axis
    return agg


# -----------------------------
# Automatic determination of activity windows
# -----------------------------
def determine_activity_windows(
    agg: pd.DataFrame, sunrise_hour: int, sunset_hour: int
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Determine optimal walking times and avoid period based on heat index data.
    Walking times are suggested after sunrise and before sunset, during cooler periods.

    Returns:
        Tuple of (morning_walk, evening_walk, avoid_period)
        Each is a tuple of (label, color, start_time, duration)
    """
    # Calculate heat index for each hour
    heat_index_values = heat_index_celsius(
        agg["temperature_c"].values, agg["rh_pct"].values
    )

    # Create a dataframe with hours and heat index
    hourly_heat = pd.DataFrame({"hour": range(24), "heat_index": heat_index_values})

    # Define safe heat index threshold (in Celsius)
    # According to NOAA:
    # - Caution: 26.7-32.2°C (80-90°F)
    # - Extreme caution: 32.2-40.6°C (90-105°F)
    # - Danger: 40.6-54.4°C (105-130°F)
    # We'll set our "too hot" threshold at 32.0°C
    TOO_HOT_THRESHOLD = 32.0

    # Calculate rolling average over 2 hours to determine "too hot" periods
    # This considers the current hour and the next hour for a more robust assessment
    heat_index_extended = np.concatenate(
        [heat_index_values, [heat_index_values[0]]]
    )  # Wrap around
    rolling_avg = (heat_index_extended[:-1] + heat_index_extended[1:]) / 2

    # Find hours that are "too hot" based on the rolling average
    too_hot_mask = rolling_avg >= TOO_HOT_THRESHOLD
    too_hot_hours = hourly_heat[too_hot_mask]

    # Initialize avoid period
    avoid_start = "12:00"
    avoid_end = "16:59"

    # Determine avoid period as the continuous block of hottest hours
    if len(too_hot_hours) > 0:
        # Find consecutive blocks of too hot hours
        too_hot_hours_sorted = too_hot_hours.sort_values("hour").reset_index(drop=True)
        too_hot_hours_sorted["block"] = (
            too_hot_hours_sorted["hour"].diff() > 1
        ).cumsum()

        # Find the longest consecutive unsafe block
        if not too_hot_hours_sorted.empty:
            longest_block = too_hot_hours_sorted.groupby("block").size().idxmax()
            avoid_block = too_hot_hours_sorted[
                too_hot_hours_sorted["block"] == longest_block
            ]

            # Determine avoid period bounds
            if not avoid_block.empty:
                avoid_start_hour = avoid_block["hour"].min()
                avoid_end_hour = avoid_block["hour"].max()

                # Handle wraparound if needed
                if avoid_end_hour < avoid_start_hour:
                    avoid_end_hour = 23

                # Round to full hours
                avoid_start = f"{int(round(avoid_start_hour)):02d}:00"
                avoid_end = f"{int(round(avoid_end_hour)):02d}:59"

    # Ensure sunrise/sunset are within reasonable bounds
    sunrise_hour = max(5, sunrise_hour)  # Prevent early morning suggestions
    sunset_hour = min(21, sunset_hour)  # Prevent late evening suggestions

    # Find all hours within daylight and below safe threshold
    safe_hours = hourly_heat[
        (hourly_heat["hour"] >= sunrise_hour)
        & (hourly_heat["hour"] <= sunset_hour)
        & (hourly_heat["heat_index"] < TOO_HOT_THRESHOLD)
    ].sort_values("heat_index")

    # Select morning window (earliest safe hours after sunrise)
    morning_candidates = safe_hours[safe_hours["hour"] <= 11]
    if not morning_candidates.empty:
        # Find earliest consecutive 1-2 hour block
        morning_start = morning_candidates.iloc[0]["hour"]
        morning_duration = min(2, len(morning_candidates))
    else:
        # Fallback to default if no safe hours found
        morning_start = sunrise_hour + 1
        morning_duration = 1

    # Select evening window (latest safe hours before sunset)
    evening_candidates = safe_hours[safe_hours["hour"] >= 14]
    if not evening_candidates.empty:
        # Allow evening walk to extend closer to sunset
        # Evening walk can start at the latest safe hour, even if it's close to sunset
        evening_start = evening_candidates.iloc[-1]["hour"]
        evening_duration = 1  # Simplify to 1 hour

        # Allow evening walk to start as close to sunset as possible
        evening_start = min(evening_start, sunset_hour)
    else:
        # Fallback to default if no safe hours found
        evening_start = sunset_hour - 1  # Start closer to sunset
        evening_duration = 1

    # Round to full hours
    morning_start = int(round(morning_start))
    evening_start = int(round(evening_start))

    # Ensure windows respect daylight constraints
    morning_start = max(sunrise_hour, morning_start)
    evening_start = min(sunset_hour - evening_duration, evening_start)

    # Format start times as HH:00
    morning_walk_start_str = f"{morning_start:02d}:00"
    evening_walk_start_str = f"{evening_start:02d}:00"

    # Create the tuples for the activity windows
    morning_walk = (
        "Walk Morning",
        "green",
        morning_walk_start_str,
        float(morning_duration),
    )
    evening_walk = (
        "Walk Evening",
        "green",
        evening_walk_start_str,
        float(evening_duration),
    )
    avoid_period = (avoid_start, avoid_end)

    return morning_walk, evening_walk, avoid_period


def mk_dt_on(day: pd.Timestamp, hhmm: str, tz: str) -> pd.Timestamp:
    """Create a timezone-aware timestamp on the given date with HH:MM local time."""
    hour, minute = map(int, hhmm.split(":"))
    return pd.Timestamp(
        year=day.year, month=day.month, day=day.day, hour=hour, minute=minute, tz=tz
    )


def plot_chart(
    agg: pd.DataFrame, tz: str, daily_df: pd.DataFrame, location_name: str
) -> plt.Figure:
    """Plot 24h Heat Index curve with heat index level segments and sunrise/sunset. Returns the figure."""
    import matplotlib.dates as mdates

    # Build a reference local date (today in that tz)
    ref_day = pd.Timestamp.now(tz).normalize()

    # Build a 24h time axis on the reference day with higher resolution for smooth filling
    # Use 10-minute intervals for smoother color transitions
    times_hourly = [ref_day + pd.Timedelta(hours=h) for h in range(24)]
    times_smooth = [ref_day + pd.Timedelta(minutes=m) for m in range(0, 24 * 60, 10)]

    # Get hourly heat index values
    hi_hourly = heat_index_celsius(agg["temperature_c"].values, agg["rh_pct"].values)

    # Interpolate to get smooth heat index values for filling
    hi_smooth = np.interp(
        [t.hour + t.minute / 60.0 for t in times_smooth],
        [t.hour + t.minute / 60.0 for t in times_hourly],
        hi_hourly,
    )

    # Calculate median sunrise/sunset hours for display
    sunrise_mins = [(t.hour * 60 + t.minute) for t in daily_df["sunrise"]]
    sunset_mins = [(t.hour * 60 + t.minute) for t in daily_df["sunset"]]
    sunrise_med = int(np.median(sunrise_mins))
    sunset_med = int(np.median(sunset_mins))

    sunrise_ts = mk_dt_on(
        ref_day, f"{sunrise_med // 60:02d}:{sunrise_med % 60:02d}", tz
    )
    sunset_ts = mk_dt_on(ref_day, f"{sunset_med // 60:02d}:{sunset_med % 60:02d}", tz)

    # Ensure the timestamps are in the same timezone as the plot
    sunrise_ts = sunrise_ts.tz_convert(tz)
    sunset_ts = sunset_ts.tz_convert(tz)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        times_hourly,
        hi_hourly,
        label="Heat Index (°C)",
        color="black",
        linewidth=2,
        marker="o",
    )
    ax.set_ylabel("Heat Index (°C)")
    ax.set_xlabel(f"Local time ({tz})")
    ax.set_title(f"24h Heat Index (7-day hourly averages) — {location_name}")

    # Set x-axis limits to show full 24-hour period
    ax.set_xlim(ref_day, ref_day + pd.Timedelta(hours=24))

    # Set the x-axis to use the local timezone
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H", tz=tz))

    # Define heat index thresholds (in Celsius) based on user specification:
    # Green: Below 27°C (Suitable for activity)
    # Yellow: 27-32°C (Caution)
    # Deep Yellow: 32-41°C (Extreme Caution)
    # Orange: 41-54°C (Danger)
    # Red: Above 54°C (Extreme Danger)

    caution_threshold = 27.0
    extreme_caution_threshold = 32.0
    danger_threshold = 41.0
    extreme_danger_threshold = 54.0

    # Create time points for filling (smooth points)
    time_points = np.array(times_smooth)

    # Fill areas based on heat index levels using smooth interpolated values
    # Green: Suitable for activity (below caution level)
    ax.fill_between(
        time_points,
        np.min(hi_smooth) - 5,
        hi_smooth,
        where=(hi_smooth < caution_threshold),
        color="green",
        alpha=0.3,
        label="Suitable for activity",
    )

    # Yellow: Caution (27-32°C)
    ax.fill_between(
        time_points,
        np.min(hi_smooth) - 5,
        hi_smooth,
        where=(hi_smooth >= caution_threshold)
        & (hi_smooth < extreme_caution_threshold),
        color="yellow",
        alpha=0.5,
        label="Caution",
    )

    # Deep Yellow: Extreme Caution (32-41°C)
    ax.fill_between(
        time_points,
        np.min(hi_smooth) - 5,
        hi_smooth,
        where=(hi_smooth >= extreme_caution_threshold) & (hi_smooth < danger_threshold),
        color="#FFD700",
        alpha=0.6,
        label="Extreme Caution",
    )  # Deep yellow/gold

    # Orange: Danger (41-54°C)
    ax.fill_between(
        time_points,
        np.min(hi_smooth) - 5,
        hi_smooth,
        where=(hi_smooth >= danger_threshold) & (hi_smooth < extreme_danger_threshold),
        color="orange",
        alpha=0.7,
        label="Danger",
    )

    # Red: Extreme Danger (above 54°C)
    ax.fill_between(
        time_points,
        np.min(hi_smooth) - 5,
        hi_smooth,
        where=(hi_smooth >= extreme_danger_threshold),
        color="red",
        alpha=0.8,
        label="Extreme Danger",
    )

    # Sunrise / Sunset lines
    ax.axvline(sunrise_ts, linestyle="--", linewidth=1.5, color="gold", label="Sunrise")
    ax.axvline(sunset_ts, linestyle="--", linewidth=1.5, color="purple", label="Sunset")

    # X axis formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate()

    # Enable autoscaling for y-axis
    ax.relim()
    ax.autoscale_view()

    # Legend
    ax.legend(loc="upper left")

    # Remove the tight layout and save here; return the figure
    plt.tight_layout()
    return fig


def process_location(location_name: str, lat: float, lon: float) -> plt.Figure:
    """Process a single location: fetch data, compute averages, and plot."""
    print(f"\nProcessing location: {location_name} ({lat}, {lon})")

    # Try to load data from storage first
    cached_data = load_from_storage(location_name)

    if cached_data:
        print("Using cached data from local storage")
        # Reconstruct data from cached JSON
        hourly_data = pd.DataFrame(
            {
                "time": pd.to_datetime(cached_data["hourly_data"]["time"]),
                "temperature_c": cached_data["hourly_data"]["temperature_c"],
                "rh_pct": cached_data["hourly_data"]["rh_pct"],
            }
        ).set_index("time")

        daily_data = pd.DataFrame(
            {
                "sunrise": [pd.to_datetime(cached_data["daily_data"]["sunrise"])],
                "sunset": [pd.to_datetime(cached_data["daily_data"]["sunset"])],
            }
        )

        tz_name = cached_data["tz_name"]
    else:
        print("Fetching fresh data from API")
        hourly_df, tz_name, daily_df = fetch_data(lat, lon)

        # Save to storage for future use
        cache_data = {
            "hourly_data": {
                "time": hourly_df.index.strftime("%Y-%m-%d %H:%M:%S%z").tolist(),
                "temperature_c": hourly_df["temperature_c"].tolist(),
                "rh_pct": hourly_df["rh_pct"].tolist(),
            },
            "daily_data": {
                "sunrise": daily_df["sunrise"].iloc[0].strftime("%Y-%m-%d %H:%M:%S%z"),
                "sunset": daily_df["sunset"].iloc[0].strftime("%Y-%m-%d %H:%M:%S%z"),
            },
            "tz_name": tz_name,
        }
        save_to_storage(cache_data, location_name)

        # Use fresh data
        hourly_data = hourly_df
        daily_data = daily_df

    agg = per_hour_means(hourly_data)
    
    # Calculate sunrise/sunset for printing
    sunrise_mins = [(t.hour * 60 + t.minute) for t in daily_data["sunrise"]]
    sunset_mins = [(t.hour * 60 + t.minute) for t in daily_data["sunset"]]
    sunrise_med = int(np.median(sunrise_mins))
    sunset_med = int(np.median(sunset_mins))
    
    # Print sunrise/sunset times for user information
    print(f"Sunrise/sunset information for {location_name}:")
    print(f"  Sunrise: {sunrise_med // 60:02d}:{sunrise_med % 60:02d}")
    print(f"  Sunset: {sunset_med // 60:02d}:{sunset_med % 60:02d}")
    
    fig = plot_chart(agg, tz_name, daily_data, location_name)
    return fig


def main():
    """Process all configured locations."""
    figures = []
    for location_name, (lat, lon) in LOCATIONS.items():
        fig = process_location(location_name, lat, lon)
        figures.append(fig)
    
    # Save all figures to a single PDF
    pdf_filename = "heat_index_plots.pdf"
    with PdfPages(pdf_filename) as pdf:
        for fig in figures:
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
    
    print(f"\nAll plots saved to {pdf_filename}")
    
    # Close all figures to free memory
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
