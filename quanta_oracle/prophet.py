"""
Prophet-style additive time series forecasting model.

    y(t) = trend(t) + seasonality(t) + residual(t)

- Trend: piecewise linear with automatic changepoints.
- Seasonality: Fourier series (configurable harmonics per period).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_array(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("Input must be 1-D")
    return a


# ---------------------------------------------------------------------------
# Fourier feature generation
# ---------------------------------------------------------------------------

def _make_fourier_features(
    t: np.ndarray,
    period: float,
    order: int,
) -> np.ndarray:
    """Generate sin/cos Fourier features for a given period.

    Returns an (n, 2*order) matrix where columns are:
        sin(2*pi*1*t/period), cos(2*pi*1*t/period),
        sin(2*pi*2*t/period), cos(2*pi*2*t/period), ...
    """
    n = len(t)
    features = np.zeros((n, 2 * order))
    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * t / period
        features[:, 2 * (k - 1)] = np.sin(angle)
        features[:, 2 * (k - 1) + 1] = np.cos(angle)
    return features


# ---------------------------------------------------------------------------
# Piecewise linear trend
# ---------------------------------------------------------------------------

def _piecewise_linear(
    t: np.ndarray,
    changepoints: np.ndarray,
    k: float,
    m: float,
    deltas: np.ndarray,
) -> np.ndarray:
    """Evaluate piecewise linear trend at times *t*.

    Parameters
    ----------
    t : (n,) array of time indices.
    changepoints : (S,) array of changepoint time locations.
    k : base slope.
    m : base intercept.
    deltas : (S,) array of slope adjustments at each changepoint.

    The trend at time t is:
        trend(t) = (k + sum(delta_j for s_j <= t)) * t
                   + (m + sum(-s_j * delta_j for s_j <= t))
    """
    n = len(t)
    trend = np.zeros(n)
    for i in range(n):
        slope = k
        offset = m
        for j, cp in enumerate(changepoints):
            if t[i] >= cp:
                slope += deltas[j]
                offset -= changepoints[j] * deltas[j]
        trend[i] = slope * t[i] + offset
    return trend


# ---------------------------------------------------------------------------
# Prophet model
# ---------------------------------------------------------------------------

class Prophet:
    """Additive time series model with piecewise-linear trend and Fourier
    seasonality.

    Parameters
    ----------
    yearly_seasonality : bool
        Include a yearly seasonal component (period = 365.25 for daily data).
    weekly_seasonality : bool
        Include a weekly seasonal component (period = 7 for daily data).
    n_changepoints : int
        Number of potential changepoints placed in the first 80% of the series.
    fourier_order : int
        Number of Fourier harmonics for each seasonal component.
    yearly_period : float
        Override the yearly period (default 365.25).
    weekly_period : float
        Override the weekly period (default 7.0).
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        n_changepoints: int = 25,
        fourier_order: int = 10,
        yearly_period: float = 365.25,
        weekly_period: float = 7.0,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.n_changepoints = max(n_changepoints, 0)
        self.fourier_order = max(fourier_order, 1)
        self.yearly_period = yearly_period
        self.weekly_period = weekly_period

        # Fitted state
        self._fitted = False
        self._k: float = 0.0                    # base slope
        self._m: float = 0.0                    # base intercept
        self._changepoints: np.ndarray = np.array([])
        self._deltas: np.ndarray = np.array([])
        self._seasonal_coeffs: Optional[np.ndarray] = None
        self._residual_std: float = 0.0
        self._t_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
    ) -> "Prophet":
        """Fit the Prophet model.

        Parameters
        ----------
        timestamps : array-like of float
            Numeric time indices (e.g. days since epoch, or simply 0..n-1).
        values : array-like of float
            Observed time series values.

        Returns
        -------
        self
        """
        t = _to_array(timestamps)
        y = _to_array(values)
        if len(t) != len(y):
            raise ValueError("timestamps and values must have equal length")
        n = len(t)
        if n < 4:
            raise ValueError("Need at least 4 data points")

        self._t_train = t.copy()
        self._y_train = y.copy()

        # --- Step 1: Place changepoints in the first 80% of the series ---
        cp_range = int(0.8 * n)
        n_cp = min(self.n_changepoints, cp_range - 1)
        if n_cp > 0:
            cp_indices = np.linspace(1, cp_range - 1, n_cp, dtype=int)
            self._changepoints = t[cp_indices]
        else:
            self._changepoints = np.array([])

        # --- Step 2: Fit piecewise-linear trend via least squares ---------
        # Design matrix columns: [1, t, A_1, A_2, ..., A_S]
        # where A_j(t) = max(0, t - s_j)  (hinge basis)
        n_cp_actual = len(self._changepoints)
        n_trend_cols = 2 + n_cp_actual
        A_trend = np.ones((n, n_trend_cols))
        A_trend[:, 1] = t
        for j, cp in enumerate(self._changepoints):
            A_trend[:, 2 + j] = np.maximum(0.0, t - cp)

        # Solve via least squares
        trend_coeffs, _, _, _ = np.linalg.lstsq(A_trend, y, rcond=None)

        self._m = float(trend_coeffs[0])
        self._k = float(trend_coeffs[1])
        self._deltas = trend_coeffs[2:].copy() if n_cp_actual > 0 else np.array([])

        trend_fitted = A_trend @ trend_coeffs

        # --- Step 3: Fit Fourier seasonality on detrended data ------------
        detrended = y - trend_fitted

        seasonal_features = []
        if self.yearly_seasonality:
            seasonal_features.append(
                _make_fourier_features(t, self.yearly_period, self.fourier_order)
            )
        if self.weekly_seasonality:
            seasonal_features.append(
                _make_fourier_features(t, self.weekly_period, self.fourier_order)
            )

        if seasonal_features:
            X_seasonal = np.hstack(seasonal_features)
            self._seasonal_coeffs, _, _, _ = np.linalg.lstsq(
                X_seasonal, detrended, rcond=None
            )
            seasonal_fitted = X_seasonal @ self._seasonal_coeffs
        else:
            self._seasonal_coeffs = None
            seasonal_fitted = np.zeros(n)

        # Residual standard deviation (for uncertainty intervals)
        residual = y - trend_fitted - seasonal_fitted
        self._residual_std = float(np.std(residual))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, future_timestamps: ArrayLike) -> dict:
        """Generate forecasts for *future_timestamps*.

        Returns
        -------
        dict with keys:
            ``yhat``       — point forecast
            ``trend``      — trend component
            ``seasonal``   — seasonal component
            ``yhat_lower`` — lower 95% interval
            ``yhat_upper`` — upper 95% interval
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet")

        t = _to_array(future_timestamps)
        n = len(t)

        # Trend
        n_cp = len(self._changepoints)
        A_trend = np.ones((n, 2 + n_cp))
        A_trend[:, 1] = t
        for j, cp in enumerate(self._changepoints):
            A_trend[:, 2 + j] = np.maximum(0.0, t - cp)

        trend_coeffs = np.concatenate([[self._m, self._k], self._deltas])
        trend = A_trend @ trend_coeffs

        # Seasonality
        seasonal_features = []
        if self.yearly_seasonality:
            seasonal_features.append(
                _make_fourier_features(t, self.yearly_period, self.fourier_order)
            )
        if self.weekly_seasonality:
            seasonal_features.append(
                _make_fourier_features(t, self.weekly_period, self.fourier_order)
            )

        if seasonal_features and self._seasonal_coeffs is not None:
            X_seasonal = np.hstack(seasonal_features)
            seasonal = X_seasonal @ self._seasonal_coeffs
        else:
            seasonal = np.zeros(n)

        yhat = trend + seasonal
        margin = 1.96 * self._residual_std

        return {
            "yhat": yhat,
            "trend": trend,
            "seasonal": seasonal,
            "yhat_lower": yhat - margin,
            "yhat_upper": yhat + margin,
        }

    def _make_fourier_features(
        self,
        t: np.ndarray,
        period: float,
        order: int,
    ) -> np.ndarray:
        """Instance-method wrapper around module-level Fourier features."""
        return _make_fourier_features(t, period, order)

    def __repr__(self) -> str:
        return (
            f"Prophet(yearly={self.yearly_seasonality}, "
            f"weekly={self.weekly_seasonality}, "
            f"n_changepoints={self.n_changepoints}, "
            f"fourier_order={self.fourier_order})"
        )
