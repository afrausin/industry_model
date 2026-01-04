"""
NY Fed Data Loader for Industry Model

Loads Weekly Economic Index (WEI) and Survey of Consumer Expectations (SCE)
from the raw JSON data files.
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from ..config import ModelConfig


class NYFedLoader:
    """
    Load NY Fed data for signal generation.

    Datasets:
    - Weekly Economic Index (WEI): High-frequency growth proxy
    - Survey of Consumer Expectations (SCE): Consumer inflation expectations
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize NY Fed data loader.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        self.data_dir = self.config.project_root / "data" / "storage" / "raw" / "ny_fed"

    def load_wei(self) -> pd.DataFrame:
        """
        Load Weekly Economic Index (WEI).

        The WEI is a high-frequency measure of real economic activity.
        It scales to 4-week moving average of GDP growth at an annual rate.

        Returns:
            DataFrame indexed by date with WEI values.
        """
        wei_file = self.data_dir / "weekly_economic_index.json"

        if not wei_file.exists():
            print(f"WEI data not found at {wei_file}")
            return pd.DataFrame()

        try:
            with open(wei_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading WEI data: {e}")
            return pd.DataFrame()

        # Extract time series data
        records = []

        # Handle different possible data structures
        if isinstance(data, dict):
            if "data" in data:
                raw_data = data["data"]
            elif "values" in data:
                raw_data = data["values"]
            elif "wei" in data:
                raw_data = data["wei"]
            else:
                # Try to extract from nested structure
                raw_data = data
        else:
            raw_data = data

        # Parse records
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    date_val = item.get("date") or item.get("Date")
                    wei_val = item.get("wei") or item.get("WEI") or item.get("value")
                    if date_val and wei_val is not None:
                        try:
                            records.append({
                                "date": pd.to_datetime(date_val),
                                "wei": float(wei_val)
                            })
                        except (ValueError, TypeError):
                            continue

        if not records:
            print("No WEI data records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("date").set_index("date")

        return df

    def load_sce_overview(self) -> pd.DataFrame:
        """
        Load Survey of Consumer Expectations overview.

        Returns:
            DataFrame with SCE summary data.
        """
        sce_file = self.data_dir / "sce_overview.json"

        if not sce_file.exists():
            print(f"SCE overview not found at {sce_file}")
            return pd.DataFrame()

        try:
            with open(sce_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading SCE overview: {e}")
            return pd.DataFrame()

        # Convert to DataFrame
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    def load_sce_inflation(self) -> pd.DataFrame:
        """
        Load SCE inflation expectations.

        Includes 1-year and 3-year ahead inflation expectations.

        Returns:
            DataFrame with inflation expectation data.
        """
        sce_file = self.data_dir / "sce_inflation.json"

        if not sce_file.exists():
            print(f"SCE inflation data not found at {sce_file}")
            return pd.DataFrame()

        try:
            with open(sce_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading SCE inflation: {e}")
            return pd.DataFrame()

        # Parse the data
        records = []

        if isinstance(data, dict):
            if "data" in data:
                raw_data = data["data"]
            else:
                raw_data = [data]
        else:
            raw_data = data

        for item in raw_data:
            if isinstance(item, dict):
                record = {}
                for key, value in item.items():
                    if "date" in key.lower():
                        try:
                            record["date"] = pd.to_datetime(value)
                        except:
                            pass
                    elif "1y" in key.lower() or "1_year" in key.lower():
                        record["inflation_1y"] = value
                    elif "3y" in key.lower() or "3_year" in key.lower():
                        record["inflation_3y"] = value
                    else:
                        record[key] = value

                if record:
                    records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "date" in df.columns:
            df = df.sort_values("date").set_index("date")

        return df

    def get_wei_weekly(self) -> pd.Series:
        """
        Get WEI as a weekly time series.

        Returns:
            Series of WEI values indexed by week.
        """
        df = self.load_wei()

        if df.empty or "wei" not in df.columns:
            return pd.Series(dtype=float, name="wei")

        # Resample to weekly (already weekly, but align to standard dates)
        weekly = df["wei"].resample("W-FRI").last()
        weekly.name = "wei"

        return weekly

    def get_wei_change(self, periods: int = 4) -> pd.Series:
        """
        Get WEI change over specified periods.

        Args:
            periods: Number of periods for change calculation.

        Returns:
            Series of WEI changes.
        """
        wei = self.get_wei_weekly()

        if wei.empty:
            return pd.Series(dtype=float, name="wei_change")

        change = wei.diff(periods)
        change.name = "wei_change"

        return change

    def get_latest_wei(self) -> Optional[float]:
        """
        Get the most recent WEI value.

        Returns:
            Latest WEI value or None.
        """
        wei = self.get_wei_weekly()

        if wei.empty:
            return None

        return wei.iloc[-1]

    def is_wei_improving(self, periods: int = 4) -> Optional[bool]:
        """
        Check if WEI is improving (rising).

        Args:
            periods: Lookback periods for comparison.

        Returns:
            True if improving, False if deteriorating, None if unknown.
        """
        change = self.get_wei_change(periods)

        if change.empty:
            return None

        return change.iloc[-1] > 0
