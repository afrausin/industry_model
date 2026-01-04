"""
Atlanta Fed Data Loader for Industry Model

Loads GDPNow real-time GDP nowcast data from raw JSON files.
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from ..config import ModelConfig


class AtlantaFedLoader:
    """
    Load Atlanta Fed GDPNow data for signal generation.

    GDPNow is a nowcasting model that estimates current-quarter GDP growth
    in real-time, updating with each new economic data release.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Atlanta Fed data loader.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        self.data_dir = self.config.project_root / "data" / "storage" / "raw" / "atlanta_fed"

    def load_gdpnow_current(self) -> Dict[str, Any]:
        """
        Load current GDPNow estimate.

        Returns:
            Dict with current estimate details.
        """
        gdpnow_file = self.data_dir / "gdpnow_current.json"

        if not gdpnow_file.exists():
            print(f"GDPNow current data not found at {gdpnow_file}")
            return {}

        try:
            with open(gdpnow_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading GDPNow current: {e}")
            return {}

        return data

    def load_gdpnow_history(self) -> pd.DataFrame:
        """
        Load historical GDPNow forecasts.

        Returns:
            DataFrame with historical forecast evolution.
        """
        history_file = self.data_dir / "gdpnow_history.json"

        if not history_file.exists():
            print(f"GDPNow history not found at {history_file}")
            return pd.DataFrame()

        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading GDPNow history: {e}")
            return pd.DataFrame()

        # Parse records
        records = []

        if isinstance(data, dict):
            if "history" in data:
                raw_data = data["history"]
            elif "data" in data:
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
                    elif "estimate" in key.lower() or "forecast" in key.lower() or "gdp" in key.lower():
                        try:
                            record["gdpnow_estimate"] = float(value)
                        except:
                            pass
                    elif "quarter" in key.lower():
                        record["quarter"] = value

                if record and "gdpnow_estimate" in record:
                    records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "date" in df.columns:
            df = df.sort_values("date").set_index("date")

        return df

    def load_gdpnow_components(self) -> Dict[str, Any]:
        """
        Load GDPNow component contributions.

        Returns:
            Dict with component-level forecast details.
        """
        components_file = self.data_dir / "gdpnow_components.json"

        if not components_file.exists():
            print(f"GDPNow components not found at {components_file}")
            return {}

        try:
            with open(components_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading GDPNow components: {e}")
            return {}

        return data

    def get_current_estimate(self) -> Optional[float]:
        """
        Get the current GDPNow estimate.

        Returns:
            Current GDP growth estimate (annualized %) or None.
        """
        current = self.load_gdpnow_current()

        if not current:
            return None

        # Try different possible keys
        for key in ["estimate", "gdp_estimate", "forecast", "value", "gdpnow"]:
            if key in current:
                try:
                    return float(current[key])
                except (ValueError, TypeError):
                    continue

        # Try nested structure
        if "data" in current and isinstance(current["data"], dict):
            for key in ["estimate", "gdp_estimate", "forecast", "value"]:
                if key in current["data"]:
                    try:
                        return float(current["data"][key])
                    except (ValueError, TypeError):
                        continue

        return None

    def get_estimate_change(self, periods: int = 1) -> Optional[float]:
        """
        Get the change in GDPNow estimate from previous update.

        Args:
            periods: Number of updates to look back.

        Returns:
            Change in estimate or None.
        """
        history = self.load_gdpnow_history()

        if history.empty or "gdpnow_estimate" not in history.columns:
            return None

        if len(history) < periods + 1:
            return None

        current = history["gdpnow_estimate"].iloc[-1]
        previous = history["gdpnow_estimate"].iloc[-(periods + 1)]

        return current - previous

    def is_gdpnow_improving(self) -> Optional[bool]:
        """
        Check if GDPNow estimate is being revised higher.

        Returns:
            True if improving, False if deteriorating, None if unknown.
        """
        change = self.get_estimate_change()

        if change is None:
            return None

        return change > 0

    def get_gdpnow_series(self) -> pd.Series:
        """
        Get GDPNow estimates as a time series.

        Returns:
            Series of GDPNow estimates indexed by date.
        """
        history = self.load_gdpnow_history()

        if history.empty or "gdpnow_estimate" not in history.columns:
            return pd.Series(dtype=float, name="gdpnow")

        series = history["gdpnow_estimate"]
        series.name = "gdpnow"

        return series

    def get_gdpnow_weekly(self) -> pd.Series:
        """
        Get GDPNow estimates aligned to weekly frequency.

        Returns:
            Weekly series with forward-filled GDPNow values.
        """
        series = self.get_gdpnow_series()

        if series.empty:
            return pd.Series(dtype=float, name="gdpnow")

        # Resample to weekly with forward fill
        weekly = series.resample("W-FRI").last().ffill()
        weekly.name = "gdpnow"

        return weekly
