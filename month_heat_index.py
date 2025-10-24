#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch one full past month's hourly temperature & humidity for multiple locations,
compute per-hour averages, derive Heat Index, and plot 24h charts
with heat index levels, median sunrise/sunset.

Notes:
- Comments are in English.
- Uses Open-Meteo's openmeteo-requests client with caching & retry.
- Targets previous full calendar month (e.g., September if running in October).
"""

import datetime as dt
import json
import os
import calendar
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
locations = [
    {"name": "Paphos", "lat": 34.7768, "lon": 32.4245},
    {"name": "Karakol", "lat": 42.4901, "lon": 78.3958},
    # {"name": "Kampos", "lat": 35.0403, "lon": 32.7324},
    # {"name": "Limassol", "lat": 34.7071, "lon": 33.0226},
    # {"name": "Nicosia", "lat": 35.1856, "lon": 33.3823},
    {"name": "Belgrade", "lat": 44.804, "lon": 20.465},
    {"name": "Budva", "lat": 42.2864, "lon": 18.8419},
    {"name": "Sevan", "lat": 40.5556, "lon": 45.0034},
    {"name": "Yerevan", "lat": 40.1792, "lon": 44.4991},
    {"name": "Girona", "lat": 41.9794, "lon": 2.8214},
    {"name": "Bilbao", "lat": 43.263, "lon": -2.935},
]


# -----------------------------
# Heat Index (NOAA) in °C
# -----------------------------
def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Compute Heat Index in Celsius from air temperature in Celsius and RH in %.
    Formula adapted from NOAA/Steadman's regression, operating in Fahrenheit internally.
    Only applies formula when temp >= 26.7°C (80°F), as per NOAA applicability.
    For colder temperatures, returns the actual temperature (HI not defined for cold weather).
    """
    # Threshold for HI applicability (80°F = 26.666...°C)
    HI_THRESHOLD_C = 26.7

    # Initialize output with temperature
    result = temp_c.copy()

    # Mask for where to apply HI formula
    hot_mask = temp_c >= HI_THRESHOLD_C

    if np.any(hot_mask):
        # Convert to Fahrenheit only for hot conditions
        T = temp_c[hot_mask] * 9.0 / 5.0 + 32.0
        rh_hot = rh[hot_mask]

        # Core regression (vectorized)
        HI = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * rh_hot
            - 0.22475541 * T * rh_hot
            - 6.83783e-3 * T * T
            - 5.481717e-2 * rh_hot * rh_hot
            + 1.22874e-3 * T * T * rh_hot
            + 8.5282e-4 * T * rh_hot * rh_hot
            - 1.99e-6 * T * T * rh_hot * rh_hot
        )

        # Adjustment terms (apply where conditions hold)
        adj = np.zeros_like(HI)

        mask_low_rh = (rh_hot < 13) & (T >= 80) & (T <= 112)
        if np.any(mask_low_rh):
            adj[mask_low_rh] -= ((13 - rh_hot[mask_low_rh]) / 4.0) * np.sqrt(
                (17.0 - np.abs(T[mask_low_rh] - 95.0)) / 17.0
            )

        mask_high_rh = (rh_hot > 85) & (T >= 80) & (T <= 87)
        if np.any(mask_high_rh):
            adj[mask_high_rh] += ((rh_hot[mask_high_rh] - 85.0) / 10.0) * (
                (87.0 - T[mask_high_rh]) / 5.0
            )

        HI += adj

        # Convert back to Celsius
        result[hot_mask] = (HI - 32.0) * 5.0 / 9.0

    return result


def save_to_storage(data: dict, location_name: str, start_date: str) -> None:
    """Save data to local storage with current timestamp."""
    storage_file = f"heat_data_cache_month_{location_name}_{start_date.replace('-', '')}.json"
    storage_data = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data": data,
    }
    with open(storage_file, "w") as f:
        json.dump(storage_data, f)


def load_from_storage(location_name: str, start_date: str) -> dict | None:
    """Load data from local storage if it exists and is less than 24 hours old."""
    storage_file = f"heat_data_cache_month_{location_name}_{start_date.replace('-', '')}.json"
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
# Fetch data from Open-Meteo Archive API
# -----------------------------
def fetch_data(lat: float, lon: float, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """Fetch hourly temperature & RH for given past month + sunrise/sunset daily; return df (local time) and tz name."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["sunrise", "sunset"],
        "timezone": "auto",  # local timezone
    }

    responses = client.weather_api(url, params=params)
    resp = responses[0]

    tz_name = resp.Timezone().decode("utf-8")  # e.g., "Asia/Nicosia"

    # Hourly block
    hourly = resp.Hourly()
    t_values = hourly.Variables(0).ValuesAsNumpy()  # temperature_2m
    rh_values = hourly.Variables(1).ValuesAsNumpy()  # relative_humidity_2m

    # Build time index from unix timestamps array
    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(tz=tz_name)

    hourly_df = pd.DataFrame(
        {
            "temperature_c": t_values,
            "rh_pct": rh_values,
        },
        index=time_index
    )

    # Daily sunrise/sunset
    daily = resp.Daily()
    sunrise_times = daily.Variables(0).ValuesInt64AsNumpy()  # sunrise
    sunset_times = daily.Variables(1).ValuesInt64AsNumpy()  # sunset

    # Convert to datetime objects
    sunrise_dates = pd.to_datetime(sunrise_times, unit="s", utc=True).tz_convert(tz_name)
    sunset_dates = pd.to_datetime(sunset_times, unit="s", utc=True).tz_convert(tz_name)

    daily_df = pd.DataFrame({"sunrise": sunrise_dates, "sunset": sunset_dates})

    return hourly_df, tz_name, daily_df


# -----------------------------
# Aggregate per hour-of-day
# -----------------------------
def per_hour_means(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean temperature and RH for each hour-of-day across the month."""
    df = hourly_df.copy()
    df["hod"] = df.index.hour
    agg = df.groupby("hod")[["temperature_c", "rh_pct"]].mean()
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




def process_location(location_name: str, lat: float, lon: float, start_date: str, end_date: str, days_in_month: int) -> pd.DataFrame:
    """Process a single location: fetch data, compute averages."""
    print(f"\nProcessing location: {location_name} ({lat}, {lon})")

    # Try to load data from storage first
    cached_data = load_from_storage(location_name, start_date)

    if cached_data:
        print("Using cached data from local storage")
        # Reconstruct data from cached JSON
        hourly_data = pd.DataFrame(
            {
                "temperature_c": cached_data["hourly_data"]["temperature_c"],
                "rh_pct": cached_data["hourly_data"]["rh_pct"],
            },
            index=pd.to_datetime(cached_data["hourly_data"]["time"])
        ).sort_index()

        daily_data = pd.DataFrame(
            {
                "sunrise": pd.to_datetime(cached_data["daily_data"]["sunrise"]),
                "sunset": pd.to_datetime(cached_data["daily_data"]["sunset"]),
            }
        )

        tz_name = cached_data["tz_name"]
    else:
        print("Fetching fresh data from API")
        hourly_df, tz_name, daily_df = fetch_data(lat, lon, start_date, end_date)

        # Save to storage for future use
        cache_data = {
            "hourly_data": {
                "time": [idx.isoformat() for idx in hourly_df.index],
                "temperature_c": hourly_df["temperature_c"].tolist(),
                "rh_pct": hourly_df["rh_pct"].tolist(),
            },
            "daily_data": {
                "sunrise": [ts.isoformat() for ts in daily_df["sunrise"]],
                "sunset": [ts.isoformat() for ts in daily_df["sunset"]],
            },
            "tz_name": tz_name,
            "start_date": start_date,
            "end_date": end_date,
        }
        save_to_storage(cache_data, location_name, start_date)

        # Use fresh data
        hourly_data = hourly_df
        daily_data = daily_df

    agg = per_hour_means(hourly_data)

    return agg


def main():
    """Process all configured locations for the previous full month."""
    # Define months to process (year, month tuples)
    months_to_process = [
        (2025, 6),  # June
        (2025, 7),  # July
        (2025, 8)   # August
    ]

    pdf_filename = "multi_month_heat_index_plots.pdf"
    with PdfPages(pdf_filename) as pdf:
        for year, month in months_to_process:
            _, days_in_month = calendar.monthrange(year, month)
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{days_in_month:02d}"
            month_name = calendar.month_name[month]
            print(f"Processing data for {month_name} {year} ({start_date} to {end_date})")

            fig, ax = plt.subplots(figsize=(12, 6))

            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            color_idx = 0

            # First pass to compute global min and max and store data
            data_by_location = {}
            all_hi = []
            for loc in locations:
                location_name = loc["name"]
                lat = loc["lat"]
                lon = loc["lon"]
                agg = process_location(location_name, lat, lon, start_date, end_date, days_in_month)
                hi_hourly = heat_index_celsius(agg["temperature_c"].values, agg["rh_pct"].values)
                data_by_location[location_name] = {'agg': agg, 'hi': hi_hourly}
                all_hi.append(hi_hourly)

            global_min_hi = min(np.min(hi) for hi in all_hi)
            global_max_hi = max(np.max(hi) for hi in all_hi)

            # Define and add background fill bands first (background)
            print(f"Global min heat index: {global_min_hi:.1f}°C, max: {global_max_hi:.1f}°C")

            added_labels = []

            # Very Cold: below 0°C - blue
            if global_min_hi < 0:
                lower = global_min_hi
                upper = 0
                color = 'blue'
                label = 'Very Cold (<0°C)'
                alpha = 0.4  # Further increased for visibility
                ax.axhspan(lower, upper, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added Very Cold fill: {lower:.1f} to {upper:.1f}")

            # Cold: 0-10°C - light blue
            lower_cold = max(0, global_min_hi)
            upper_cold = min(10, global_max_hi)
            if lower_cold < upper_cold:
                color = 'lightblue'
                label = 'Cold (0-10°C)'
                alpha = 0.4  # Further increased
                ax.axhspan(lower_cold, upper_cold, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added Cold fill: {lower_cold:.1f} to {upper_cold:.1f}")

            # Caution: 27-32°C - intensified yellow
            lower_caution = max(27, global_min_hi)
            upper_caution = min(32, global_max_hi)
            if lower_caution < upper_caution:
                color = 'yellow'
                label = 'Caution (27-32°C)'
                alpha = 0.8  # Increased for more intensity
                ax.axhspan(lower_caution, upper_caution, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added intensified Caution fill: {lower_caution:.1f} to {upper_caution:.1f}")

            # Extreme Caution: 32-41°C - yellow
            lower_ext_caution = max(32, global_min_hi)
            upper_ext_caution = min(41, global_max_hi)
            if lower_ext_caution < upper_ext_caution:
                color = 'orange'
                label = 'Extreme Caution (32-41°C)'
                alpha = 0.6  # Further increased
                ax.axhspan(lower_ext_caution, upper_ext_caution, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added Extreme Caution fill: {lower_ext_caution:.1f} to {upper_ext_caution:.1f}")

            # Danger: 41-54°C - orange
            lower_danger = max(41, global_min_hi)
            upper_danger = min(54, global_max_hi)
            if lower_danger < upper_danger:
                color = 'orangered'
                label = 'Danger (41-54°C)'
                alpha = 0.7  # Further increased
                ax.axhspan(lower_danger, upper_danger, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added Danger fill: {lower_danger:.1f} to {upper_danger:.1f}")

            # Extreme Danger: above 54°C - red
            lower_ext_danger = max(54, global_min_hi)
            upper_ext_danger = global_max_hi
            if lower_ext_danger < upper_ext_danger:
                color = 'red'
                label = 'Extreme Danger (>54°C)'
                alpha = 0.8  # Further increased
                ax.axhspan(lower_ext_danger, upper_ext_danger, xmin=0, xmax=1, color=color, alpha=alpha, label=label)
                added_labels.append(label)
                print(f"Added Extreme Danger fill: {lower_ext_danger:.1f} to {upper_ext_danger:.1f}")
            else:
                print("No Extreme Danger fill (max < 54°C)")

            # Now plot lines on top
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            hours = range(24)

            for i, loc in enumerate(locations):
                location_name = loc["name"]
                hi_hourly = data_by_location[location_name]['hi']
                ax.plot(hours, hi_hourly, label=location_name, color=colors[i % len(colors)], linewidth=2, marker='o')

            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Heat Index (°C)")
            ax.set_title(f"Monthly Average 24h Heat Index - {month_name} {year}")
            ax.set_xticks(range(0, 25, 2))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)

            # Adjust y-limits to include full range
            ax.set_ylim(global_min_hi - 1, global_max_hi + 2)
            print(f"Y-axis limits set to: {global_min_hi - 1:.1f} to {global_max_hi + 2:.1f}")

            plt.tight_layout()

            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Page for {month_name} {year} added to {pdf_filename}")

    print(f"\nMulti-month plot saved to {pdf_filename}")


if __name__ == "__main__":
    main()
