# heat24

A Python application that fetches weather data for multiple locations in Cyprus and generates heat index charts to help determine optimal times for outdoor activities while avoiding the sun.

## Example output

![heat_index_plot_paphos_opt](https://github.com/user-attachments/assets/37ec437b-720e-45f7-80e3-36bfa050ab78)

## Features


- Fetches 7 days of hourly temperature and humidity data for multiple locations in Cyprus
- Calculates heat index using the NOAA formula
- Generates 24-hour heat index charts with color-coded risk levels
- Automatically determines optimal walking times (morning and evening)
- Shows sunrise/sunset times on the charts
- Caches data locally to avoid repeated API calls
- Supports multiple locations including Paphos, Limassol, Nicosia, and more

## Installation

1. Install [uv](https://docs.astral.sh/uv/), a fast Python package installer and resolver:
   ```bash
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows:
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heat24
   ```

3. Install dependencies:
   ```bash
   make install
   # or
   uv pip install -e .
   ```

## Usage

Run the application:
```bash
make run
# or
uv run main.py
```

The application will:
1. Fetch weather data for all configured locations in Cyprus
2. Generate heat index charts for each location
3. Save charts as PNG files in the current directory
4. Display sunrise/sunset information for each location

## Development

### Commands

- Install dependencies: `make install`
- Run the application: `make run`
- Lint code: `make lint`
- Format code: `make format`
- Type check: `make typecheck`
- Run tests: `make test`
- Run all checks: `make check`
- Clean cache files: `make clean`

### Dependencies

- Python >= 3.12
- matplotlib >= 3.10.5
- numpy >= 2.3.2
- openmeteo-requests >= 1.7.1
- pandas >= 2.3.2
- requests-cache >= 1.2.1
- retry-requests >= 2.0.0

### Code Quality

This project uses:
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- [mypy](http://mypy-lang.org/) for type checking
- [pytest](https://docs.pytest.org/) for testing

## How It Works

1. The application fetches 7 days of hourly temperature and relative humidity data from the Open-Meteo API for each configured location
2. It calculates the heat index using the NOAA formula, which combines temperature and humidity to determine how hot it feels
3. For each location, it computes 24-hour averages across the week to smooth out daily variations
4. It generates a chart showing:
   - Heat index levels throughout the day with color-coded risk zones
   - Sunrise and sunset times
   - Recommended walking times (morning and evening)
   - Periods to avoid due to high heat index
5. Data is cached locally for 24 hours to reduce API calls

## Configuration

Locations are configured in `main.py` in the `LOCATIONS` dictionary. You can add or modify locations by changing the name and coordinates (latitude, longitude).

## Output

The application generates PNG charts for each location showing:
- 24-hour heat index curve with color-coded risk levels
- Sunrise/sunset markers
- Recommended activity windows
- Time periods to avoid

Charts are saved as `heat_index_plot_{location_name}.png` in the current directory.
