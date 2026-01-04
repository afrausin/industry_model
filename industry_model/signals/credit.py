"""
Credit Signal - Based on HY Spreads and Financial Conditions

Signal Group 3: Monitors credit conditions for risk-on/risk-off.
Tightening spreads = risk-on → XLY, XLK, XLI
Widening spreads = risk-off → XLU, XLP, XLV
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, CREDIT_SPREAD_SENSITIVITY, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class CreditSignal(BaseSignal):
    """
    Credit conditions signal based on HY spreads and NFCI.

    Logic:
    - Tightening spreads (falling BAMLH0A0HYM2) = risk-on
    - Widening spreads = risk-off
    - NFCI < 0 (loose conditions) → risk assets
    - NFCI > 0 (tight conditions) → defensive assets
    """

    name: str = "credit"
    description: str = "Credit spread and financial conditions signal"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on credit conditions."""
        # Get high-yield credit spread
        hy_spread = self._macro_loader.get_credit_spread(as_of_date)

        # Get financial conditions index
        nfci = self._macro_loader.get_financial_conditions(as_of_date)

        if hy_spread.empty and nfci.empty:
            return self._empty_scores()

        signal_strength = 0.0
        n_signals = 0

        # Credit spread signal (falling = positive/risk-on)
        if not hy_spread.empty and len(hy_spread) > self.config.credit_spread_ma_periods:
            spread_change = hy_spread.diff(self.config.credit_spread_ma_periods).iloc[-1]

            if not pd.isna(spread_change):
                # Normalize: 100bp change is significant
                spread_signal = -spread_change / 1.0  # Negative because falling is good
                spread_signal = max(min(spread_signal, 1.0), -1.0)
                signal_strength += spread_signal
                n_signals += 1

        # NFCI signal (negative = loose conditions = positive/risk-on)
        if not nfci.empty:
            current_nfci = nfci.iloc[-1]
            if not pd.isna(current_nfci):
                # NFCI is already scaled around 0, std dev ~1
                nfci_signal = -current_nfci  # Negative NFCI is positive for risk
                nfci_signal = max(min(nfci_signal, 1.0), -1.0)
                signal_strength += nfci_signal
                n_signals += 1

        # Average the signals
        if n_signals > 0:
            signal_strength /= n_signals
        else:
            return self._empty_scores()

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = CREDIT_SPREAD_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_credit_info(self, as_of_date: date) -> dict:
        """Get credit conditions information."""
        hy_spread = self._macro_loader.get_credit_spread(as_of_date)
        nfci = self._macro_loader.get_financial_conditions(as_of_date)

        info = {
            "hy_spread": None,
            "hy_spread_change": None,
            "nfci": None,
            "conditions": "unknown",
        }

        if not hy_spread.empty:
            info["hy_spread"] = hy_spread.iloc[-1]
            if len(hy_spread) > self.config.credit_spread_ma_periods:
                info["hy_spread_change"] = hy_spread.diff(self.config.credit_spread_ma_periods).iloc[-1]

        if not nfci.empty:
            info["nfci"] = nfci.iloc[-1]
            if info["nfci"] is not None:
                info["conditions"] = "tight" if info["nfci"] > 0 else "loose"

        return info
