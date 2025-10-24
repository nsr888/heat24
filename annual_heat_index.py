"""
Heat Index Heatmap Generator

This script fetches historical weather data from Open-Meteo API, calculates heat index,
and generates annual heatmaps showing average heat index by hour of day and month.
"""

import os
import json
import math
import datetime
import requests
import pytz
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Location definitions (Paphos, Novi Sad)
locations = [
    {"name": "Paphos", "lat": 34.7768, "lon": 32.4245},
    {"name": "Novi Sad", "lat": 45.2517, "lon": 19.8369},
]


def get_cache_filename(loc_name):
    """Generate cache filename from location name"""
    safe_name = loc_name.lower().replace(" ", "_")
    return f"archive-weather-{safe_name}-2024.json"


def load_from_cache(filename):
    """Load data from JSON cache file if it exists"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None


def save_to_cache(filename, data):
    """Save data to JSON cache file"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def heat_index_c(temp_c, rh):
    """
    Calculate heat index in Celsius based on temperature (°C) and relative humidity (%).
    Implements the same formula as the US National Weather Service.
    """
    if temp_c < 27.0 or rh < 40.0:
        return temp_c

    # Convert temperature to Fahrenheit for calculation
    T = temp_c * 9.0 / 5.0 + 32.0
    R = rh

    # Calculate heat index using Rothfusz regression
    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * T**2
        - 0.05481717 * R**2
        + 0.00122874 * T**2 * R
        + 0.00085282 * T * R**2
        - 0.00000199 * T**2 * R**2
    )

    # Additional adjustments for specific conditions
    if R < 13 and 80 <= T <= 112:
        HI -= ((13 - R) / 4.0) * math.sqrt((17.0 - abs(T - 95.0)) / 17.0)
    if R > 85 and 80 <= T <= 87:
        HI += ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0)

    # Convert back to Celsius
    return (HI - 32.0) * 5.0 / 9.0


def fetch_archive(lat, lon, loc_name):
    """
    Fetch weather data with retry logic, using cache when available.
    Returns data structured similarly to Open-Meteo API response.
    """
    cache_file = get_cache_filename(loc_name)
    cached_data = load_from_cache(cache_file)
    if cached_data:
        print(f"Loaded cached data for {loc_name}")
        return cached_data

    print(f"Fetching fresh data for {loc_name}")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }

    # Setup retry mechanism
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        save_to_cache(cache_file, data)
        print(f"Cached data for {loc_name}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise


def process_data(om):
    """
    Process raw weather data into monthly hourly averages.
    Returns a 24x12 matrix of heat index values (hours x months).
    """
    tz = pytz.timezone(om["timezone"])

    # Initialize sums and counts arrays (24 hours x 12 months)
    hmd_sums = [[0.0] * 12 for _ in range(24)]
    hmd_counts = [[0] * 12 for _ in range(24)]

    times = om["hourly"]["time"]
    temps = om["hourly"]["temperature_2m"]
    rhums = om["hourly"]["relative_humidity_2m"]

    for i in range(len(times)):
        try:
            # Parse and localize the datetime
            naive = datetime.datetime.strptime(times[i], "%Y-%m-%dT%H:%M")
            t = tz.localize(naive)

            month_idx = t.month - 1  # 0-11
            hour_idx = t.hour  # 0-23

            temp = temps[i]
            rh = rhums[i]
            hi = heat_index_c(temp, rh)

            if not math.isnan(hi):
                hmd_sums[hour_idx][month_idx] += hi
                hmd_counts[hour_idx][month_idx] += 1
        except Exception as e:
            print(f"Error processing time {times[i]}: {e}")

    # Calculate averages
    avg_data = [[float("nan")] * 12 for _ in range(24)]
    for hour_idx in range(24):
        for month_idx in range(12):
            if hmd_counts[hour_idx][month_idx] > 0:
                avg_data[hour_idx][month_idx] = (
                    hmd_sums[hour_idx][month_idx] / hmd_counts[hour_idx][month_idx]
                )

    return avg_data


def create_heatmap(data, location_name):
    """
    Generate and save a heatmap visualization from processed data.
    """
    # Define color map based on heat index thresholds
    colors = ["#90EE90", "#FFFF00", "#FFA500", "#FF0000", "#8B0000"]
    bounds = [20, 27, 32, 41, 54, 60]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Create figure and axis
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Plot heatmap
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    # Configure labels and ticks
    ax.set_title(f"Annual Heat Index Heatmap (Hour vs Month) - {location_name}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")

    # Month labels - one per month
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )

    # Hour labels - every 2 hours
    ax.set_yticks(np.arange(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    # Add grid lines for better readability
    ax.set_xticks(np.arange(13) - 0.5, minor=True)
    ax.set_yticks(np.arange(25) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Create custom legend
    legend_elements = [
        Patch(facecolor=colors[0], label="<27°C (Comfortable)"),
        Patch(facecolor=colors[1], label="27-32°C (Caution)"),
        Patch(facecolor=colors[2], label="32-41°C (Extreme Caution)"),
        Patch(facecolor=colors[3], label="41-54°C (Danger)"),
        Patch(facecolor=colors[4], label=">54°C (Extreme Danger)"),
    ]

    # Position the legend below the plot
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=True,
        title="Heat Index Risk Levels",
    )

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    filename = f"heat_index_annual_{location_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def main():
    """Main processing loop for all locations"""
    for loc in locations:
        print(f"\nProcessing location: {loc['name']}")
        try:
            om = fetch_archive(loc["lat"], loc["lon"], loc["name"])
            avg_data = process_data(om)
            create_heatmap(avg_data, loc["name"])
        except Exception as e:
            print(f"Error processing {loc['name']}: {e}")


if __name__ == "__main__":
    main()
