"""
High-Frequency Activity Signal - WEI, Claims, and GDPNow

Signal Group 4: Uses high-frequency indicators for real-time activity.
WEI rising / Claims falling → cyclicals
WEI falling / Claims rising → defensives
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, CYCLICAL_SECTORS, DEFENSIVE_SECTORS, ModelConfig
from ..data.ny_fed_loader import NYFedLoader
from ..data.atlanta_fed_loader import AtlantaFedLoader
from ..data.macro_loader import MacroFeatureLoader


class HighFreqSignal(BaseSignal):
    """
    High-frequency activity signal using WEI, Claims, and GDPNow.

    Logic:
    - WEI rising → cyclicals (XLI, XLB, XLY)
    - WEI falling → defensives (XLU, XLP, XLV)
    - Initial Claims rising → defensives
    - GDPNow upgrade → cyclicals, GDPNow downgrade → defensives
    """

    name: str = "high_freq"
    description: str = "High-frequency activity signal (WEI, Claims, GDPNow)"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._ny_fed_loader = NYFedLoader(config)
        self._atlanta_fed_loader = AtlantaFedLoader(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on high-frequency indicators."""
        signal_strength = 0.0
        n_signals = 0

        # WEI signal
        wei_signal = self._get_wei_signal()
        if wei_signal is not None:
            signal_strength += wei_signal
            n_signals += 1

        # Initial Claims signal (inverse - rising claims is negative)
        claims_signal = self._get_claims_signal(as_of_date)
        if claims_signal is not None:
            signal_strength += claims_signal
            n_signals += 1

        # GDPNow signal
        gdpnow_signal = self._get_gdpnow_signal()
        if gdpnow_signal is not None:
            signal_strength += gdpnow_signal
            n_signals += 1

        # Average signals
        if n_signals > 0:
            signal_strength /= n_signals
        else:
            return self._empty_scores()

        # Apply to sectors
        scores = self._empty_scores()

        for sector in CYCLICAL_SECTORS:
            if sector in scores.index:
                scores[sector] = signal_strength

        for sector in DEFENSIVE_SECTORS:
            if sector in scores.index:
                scores[sector] = -signal_strength

        scores.name = self.name
        return self._normalize_scores(scores)

    def _get_wei_signal(self) -> Optional[float]:
        """Get WEI-based signal."""
        wei_improving = self._ny_fed_loader.is_wei_improving(periods=4)

        if wei_improving is None:
            return None

        # Get magnitude from change
        wei_change = self._ny_fed_loader.get_wei_change(periods=4)
        if wei_change.empty:
            return 0.5 if wei_improving else -0.5

        # WEI is scaled like GDP growth, so 2% change is significant
        latest_change = wei_change.iloc[-1]
        if pd.isna(latest_change):
            return 0.5 if wei_improving else -0.5

        signal = latest_change / 2.0  # Normalize
        return max(min(signal, 1.0), -1.0)

    def _get_claims_signal(self, as_of_date: date) -> Optional[float]:
        """Get Initial Claims signal (rising claims is negative)."""
        claims = self._macro_loader.get_initial_claims(as_of_date)

        if claims.empty or len(claims) < 5:
            return None

        # Calculate 4-week change
        claims_change = claims.diff(4).iloc[-1]

        if pd.isna(claims_change):
            return None

        # Normalize: 50k change in claims is significant
        # Negative because rising claims is bad
        signal = -claims_change / 50000.0
        return max(min(signal, 1.0), -1.0)

    def _get_gdpnow_signal(self) -> Optional[float]:
        """Get GDPNow-based signal."""
        gdpnow_improving = self._atlanta_fed_loader.is_gdpnow_improving()

        if gdpnow_improving is None:
            return None

        # Get magnitude
        change = self._atlanta_fed_loader.get_estimate_change()
        if change is None:
            return 0.3 if gdpnow_improving else -0.3

        # 1% GDP revision is significant
        signal = change / 1.0
        return max(min(signal, 1.0), -1.0)

    def get_high_freq_info(self) -> dict:
        """Get high-frequency indicator information."""
        return {
            "wei_latest": self._ny_fed_loader.get_latest_wei(),
            "wei_improving": self._ny_fed_loader.is_wei_improving(),
            "gdpnow_estimate": self._atlanta_fed_loader.get_current_estimate(),
            "gdpnow_improving": self._atlanta_fed_loader.is_gdpnow_improving(),
        }
