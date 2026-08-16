"""
SMARD Data Collector for Master Thesis
=======================================
Energy Demand Prediction and Causal Analysis for Germany

Downloads electricity market data from the Bundesnetzagentur SMARD API.
All data is freely available under CC BY 4.0 license.

API Docs: https://github.com/bundesAPI/smard-api
Data Source: https://www.smard.de

Usage:
    pip install requests pandas tqdm
    python smard_data_collector.py

Author: [Your Name]
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory
DATA_DIR = Path("data/smard")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Base URL for the SMARD API
BASE_URL = "https://www.smard.de/app/chart_data"

# Time period: 2015-01-01 to present
# (SMARD has 15-min data from ~2015 onwards)
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2026, 3, 1)  # Adjust to current date

# Resolution options: "quarterhour", "hour", "day", "week", "month"
# Recommendation: start with "hour" for manageable size (~96k rows/year)
# Switch to "quarterhour" later if needed for specific analyses
RESOLUTION = "hour"

# Region: "DE" for all of Germany
# Also available per TSO: "50Hertz", "Amprion", "TenneT", "TransnetBW"
REGIONS = ["DE"]  # Add TSO regions later if doing regional analysis

# ============================================================================
# SMARD API FILTER CODES
# ============================================================================

# --- ELECTRICITY GENERATION (by source) ---
GENERATION_FILTERS = {
    # Renewable sources
    "gen_wind_onshore":     4067,   # Wind Onshore
    "gen_wind_offshore":    1225,   # Wind Offshore
    "gen_solar":            4068,   # Photovoltaik (Solar PV)
    "gen_biomass":          4066,   # Biomasse
    "gen_hydro":            1226,   # Wasserkraft (Hydropower)
    "gen_other_renewable":  1228,   # Sonstige Erneuerbare
    # Conventional sources
    "gen_lignite":          1223,   # Braunkohle (Lignite/Brown Coal)
    "gen_hard_coal":        4069,   # Steinkohle (Hard Coal)
    "gen_natural_gas":      4071,   # Erdgas (Natural Gas)
    "gen_nuclear":          1224,   # Kernenergie (Nuclear) -- drops to 0 after Apr 2023
    "gen_pumped_storage":   4070,   # Pumpspeicher (Pumped Storage)
    "gen_other_conventional": 1227, # Sonstige Konventionelle
}

# --- ELECTRICITY CONSUMPTION ---
CONSUMPTION_FILTERS = {
    "consumption_total":    410,    # Gesamt Netzlast (Total Grid Load) -- KEY TARGET
    "consumption_residual": 4359,   # Residuallast (Residual Load = demand - renewables)
    "consumption_pumped":   4387,   # Pumpspeicher-Verbrauch (Pumped Storage Consumption)
}

# --- MARKET DATA ---
MARKET_FILTERS = {
    "price_day_ahead":      4169,   # Day-Ahead Großhandelspreise (Wholesale Price)
    # Cross-border physical flows (selected neighbors)
    # Note: more filter codes exist for other countries
}

# --- FORECAST DATA (for forecast error analysis) ---
FORECAST_FILTERS = {
    "forecast_wind_onshore":  4801,  # Prognostizierte Wind Onshore
    "forecast_wind_offshore": 4802,  # Prognostizierte Wind Offshore
    "forecast_solar":         4803,  # Prognostizierte Photovoltaik
    "forecast_total_gen":     4804,  # Prognostizierte Gesamterzeugung
    "forecast_consumption":   4805,  # Prognostizierter Stromverbrauch
}

# Combine all filters
ALL_FILTERS = {
    **GENERATION_FILTERS,
    **CONSUMPTION_FILTERS,
    **MARKET_FILTERS,
    **FORECAST_FILTERS,
}

# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_timestamps(filter_id: int, region: str, resolution: str) -> list:
    """Get available timestamps for a given filter/region/resolution combo."""
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("timestamps", [])
    except Exception as e:
        print(f"  Error fetching timestamps for filter {filter_id}: {e}")
        return []


def get_timeseries(filter_id: int, region: str, resolution: str, timestamp: int) -> list:
    """Get timeseries data for a specific timestamp chunk."""
    url = (
        f"{BASE_URL}/{filter_id}/{region}/"
        f"{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("series", [])
    except Exception as e:
        print(f"  Error fetching data for filter {filter_id}, ts {timestamp}: {e}")
        return []


def download_filter(name: str, filter_id: int, region: str, resolution: str) -> pd.DataFrame:
    """Download all data for a single filter and return as DataFrame."""
    print(f"\n  Fetching timestamps for '{name}' (filter={filter_id})...")
    timestamps = get_timestamps(filter_id, region, resolution)

    if not timestamps:
        print(f"  No timestamps found for '{name}'")
        return pd.DataFrame()

    # Filter timestamps to our date range
    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)
    timestamps = [ts for ts in timestamps if start_ms <= ts <= end_ms]
    print(f"  Found {len(timestamps)} timestamp chunks in date range")

    all_rows = []
    for ts in tqdm(timestamps, desc=f"  Downloading {name}", leave=False):
        series = get_timeseries(filter_id, region, resolution, ts)
        for point in series:
            ts_val = point[0]  # Unix timestamp in ms
            value = point[1]   # Value (MWh for generation, EUR/MWh for prices)
            if ts_val is not None:
                all_rows.append({"timestamp_ms": ts_val, name: value})
        time.sleep(0.1)  # Be nice to the API

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df.set_index("datetime").sort_index()
    # Remove duplicates (API chunks can overlap)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ============================================================================
# MAIN DOWNLOAD PIPELINE
# ============================================================================

def download_all(regions=None, filters=None, resolution=None):
    """
    Download all configured data and save as CSV files.

    Parameters
    ----------
    regions : list, optional
        Override REGIONS config
    filters : dict, optional
        Override ALL_FILTERS config. Pass a subset to download only specific data.
    resolution : str, optional
        Override RESOLUTION config
    """
    regions = regions or REGIONS
    filters = filters or ALL_FILTERS
    resolution = resolution or RESOLUTION

    for region in regions:
        print(f"\n{'='*60}")
        print(f"Region: {region}")
        print(f"Resolution: {resolution}")
        print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
        print(f"Filters: {len(filters)}")
        print(f"{'='*60}")

        all_dfs = []

        for name, filter_id in filters.items():
            cache_file = DATA_DIR / f"{name}_{region}_{resolution}.csv"

            # Skip if already downloaded
            if cache_file.exists():
                print(f"\n  '{name}' already exists, loading from cache...")
                df = pd.read_csv(cache_file, index_col="datetime", parse_dates=True)
                all_dfs.append(df)
                continue

            df = download_filter(name, filter_id, region, resolution)

            if not df.empty:
                df.to_csv(cache_file)
                print(f"  Saved {len(df)} rows to {cache_file}")
                all_dfs.append(df)

        # Merge all filters into one combined file
        if all_dfs:
            print(f"\n  Merging {len(all_dfs)} datasets...")
            combined = pd.concat(all_dfs, axis=1)
            combined = combined.sort_index()
            out_file = DATA_DIR / f"combined_{region}_{resolution}.csv"
            combined.to_csv(out_file)
            print(f"  Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} cols")
            print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
            print(f"  Saved to {out_file}")

            # Print summary statistics
            print(f"\n  Missing values per column:")
            for col in combined.columns:
                n_missing = combined[col].isna().sum()
                pct = 100 * n_missing / len(combined)
                if n_missing > 0:
                    print(f"    {col}: {n_missing} ({pct:.1f}%)")


# ============================================================================
# WEATHER DATA HELPER (Open-Meteo — free, no API key)
# ============================================================================

def download_weather(
    lat: float = 51.5,    # Central Germany (approx.)
    lon: float = 10.5,
    start: str = "2015-01-01",
    end: str = "2026-03-01",
):
    """
    Download historical weather data from Open-Meteo (free, no API key).
    Uses the DWD ICON model for Germany-specific accuracy.

    Parameters for energy analysis:
    - temperature_2m: drives heating/cooling demand (→ HDD/CDD)
    - windspeed_10m: correlates with wind generation
    - direct_radiation: correlates with solar PV generation
    - diffuse_radiation: also relevant for solar output
    """
    print("\nDownloading weather data from Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "windspeed_10m",
            "windgusts_10m",
            "direct_radiation",
            "diffuse_radiation",
            "cloudcover",
            "precipitation",
        ]),
        "timezone": "Europe/Berlin",
        "models": "best_match",
    }

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"]).set_index("datetime")

        out_file = DATA_DIR / "weather_germany_hourly.csv"
        df.to_csv(out_file)
        print(f"  Weather data: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Saved to {out_file}")
        return df
    except Exception as e:
        print(f"  Error downloading weather data: {e}")
        print("  TIP: Open-Meteo limits historical requests to ~1 year chunks.")
        print("       You may need to loop over years and concatenate.")
        return pd.DataFrame()


# ============================================================================
# DERIVED FEATURES
# ============================================================================

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional features useful for causal analysis.
    Call after merging SMARD + weather data.
    """
    df = df.copy()

    # --- Time features ---
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # --- Heating Degree Days (base 18°C, hourly proxy) ---
    if "temperature_2m" in df.columns:
        df["hdd"] = (18.0 - df["temperature_2m"]).clip(lower=0)
        df["cdd"] = (df["temperature_2m"] - 24.0).clip(lower=0)  # Cooling Degree
        df["hdd_lag1"] = df["hdd"].shift(1)

    # --- Renewable share ---
    re_cols = [c for c in df.columns if c.startswith("gen_") and
               any(k in c for k in ["wind", "solar", "biomass", "hydro", "other_renewable"])]
    if re_cols:
        df["gen_renewables_total"] = df[re_cols].sum(axis=1)

    conv_cols = [c for c in df.columns if c.startswith("gen_") and
                 any(k in c for k in ["lignite", "hard_coal", "natural_gas", "nuclear", "other_conventional"])]
    if conv_cols:
        df["gen_conventional_total"] = df[conv_cols].sum(axis=1)

    total_gen_cols = re_cols + conv_cols
    if total_gen_cols:
        df["gen_total"] = df[total_gen_cols].sum(axis=1)
        df["renewable_share"] = df["gen_renewables_total"] / df["gen_total"].replace(0, float("nan"))

    # --- Shock dummy variables ---
    # War: Russia-Ukraine (Feb 24, 2022)
    df["war_dummy"] = (df.index >= "2022-02-24").astype(int)

    # Nuclear exit: Last 3 reactors shut down (Apr 15, 2023)
    df["nuclear_exit_dummy"] = (df.index >= "2023-04-15").astype(int)

    # Energy crisis period (roughly Oct 2021 – Dec 2022, gas price spike)
    df["energy_crisis_dummy"] = (
        (df.index >= "2021-10-01") & (df.index <= "2022-12-31")
    ).astype(int)

    return df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SMARD Data Collector — Master Thesis")
    print("Energy Demand Prediction & Causal Analysis for Germany")
    print("=" * 60)

    # Step 1: Download SMARD electricity market data
    # Start with the most important filters for initial exploration
    PRIORITY_FILTERS = {
        # Targets
        "consumption_total":    410,
        "consumption_residual": 4359,
        # Key renewable generation
        "gen_wind_onshore":     4067,
        "gen_wind_offshore":    1225,
        "gen_solar":            4068,
        # Key conventional generation
        "gen_natural_gas":      4071,
        "gen_nuclear":          1224,
        "gen_lignite":          1223,
        "gen_hard_coal":        4069,
        # Price
        "price_day_ahead":      4169,
    }

    # Download priority data first (faster, ~10 filters)
    download_all(filters=PRIORITY_FILTERS)

    # Uncomment to download ALL filters (takes longer):
    # download_all(filters=ALL_FILTERS)

    # Uncomment to also download per-TSO regional data:
    # download_all(regions=["50Hertz", "Amprion", "TenneT", "TransnetBW"],
    #              filters=PRIORITY_FILTERS)

    # Step 2: Download weather data
    # Note: Open-Meteo may require chunking for long periods
    # download_weather()

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {DATA_DIR.resolve()}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run this script to download data")
    print("2. Load combined CSV and weather CSV")
    print("3. Merge on datetime index")
    print("4. Call compute_derived_features()")
    print("5. Start with EDA and mutual information analysis")
