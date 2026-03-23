"""
Simple neural network layers built on NumPy (no PyTorch dependency).

Provides Linear, LayerNorm, activation functions, an LSTM cell,
and a lightweight MLP forecaster.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _kaiming_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming (He) uniform initialisation for ReLU networks."""
    limit = math.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def _xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier (Glorot) uniform initialisation."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


# ---------------------------------------------------------------------------
# Linear (Dense) layer
# ---------------------------------------------------------------------------

class Linear:
    """Fully-connected (dense) layer: ``z = x @ W + b``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.W = _kaiming_uniform(in_features, out_features)
        self.b = np.zeros(out_features)
        # Cached for backward pass
        self._input: Optional[np.ndarray] = None
        self._dW: Optional[np.ndarray] = None
        self._db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the linear transformation.

        Parameters
        ----------
        x : (..., in_features) array

        Returns
        -------
        (..., out_features) array
        """
        self._input = x
        return x @ self.W + self.b

    def _backward(self, d_output: np.ndarray) -> np.ndarray:
        """Compute gradients and return upstream gradient.

        Parameters
        ----------
        d_output : (..., out_features) array
            Gradient of loss w.r.t. this layer's output.

        Returns
        -------
        (..., in_features) array — gradient w.r.t. this layer's input.
        """
        self._dW = self._input.T @ d_output
        self._db = d_output.sum(axis=0)
        return d_output @ self.W.T

    def parameters(self) -> list[np.ndarray]:
        """Return a list of trainable parameter arrays [W, b]."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features})"


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

class LayerNorm:
    """Layer normalization over the last dimension.

    ``y = (x - mean) / sqrt(var + eps) * gamma + beta``

    Parameters
    ----------
    features : int
        Size of the last dimension.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, features: int, eps: float = 1e-5):
        self.features = features
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def parameters(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"LayerNorm({self.features})"


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self):
        self._input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return np.maximum(0.0, x)

    def _backward(self, d_output: np.ndarray) -> np.ndarray:
        """Mask gradient where input was <= 0."""
        return d_output * (self._input > 0)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid:
    """Logistic sigmoid activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        pos = x >= 0
        z = np.zeros_like(x)
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        z[~pos] = exp_x / (1.0 + exp_x)
        return z

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh:
    """Hyperbolic tangent activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LSTMCell:
    """Single Long Short-Term Memory cell.

    Gate equations (concatenated input ``[x, h_prev]``):

    - Forget gate:  f = sigmoid(W_f @ [x, h] + b_f)
    - Input gate:   i = sigmoid(W_i @ [x, h] + b_i)
    - Candidate:    g = tanh(W_g @ [x, h] + b_g)
    - Output gate:  o = sigmoid(W_o @ [x, h] + b_o)
    - Cell state:   c = f * c_prev + i * g
    - Hidden state: h = o * tanh(c)

    Parameters
    ----------
    input_size : int
        Dimensionality of the input vector ``x``.
    hidden_size : int
        Dimensionality of the hidden / cell state.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined = input_size + hidden_size

        # Weights for all four gates, stacked: [forget, input, candidate, output]
        self.W = _xavier_uniform(combined, 4 * hidden_size)
        self.b = np.zeros(4 * hidden_size)

        self._sigmoid = Sigmoid()

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one LSTM step.

        Parameters
        ----------
        x : (input_size,) array — current input.
        h_prev : (hidden_size,) array — previous hidden state.
        c_prev : (hidden_size,) array — previous cell state.

        Returns
        -------
        (h_next, c_next) — updated hidden and cell states.
        """
        combined = np.concatenate([x, h_prev])
        gates = combined @ self.W + self.b

        hs = self.hidden_size
        f_gate = self._sigmoid.forward(gates[0 * hs: 1 * hs])
        i_gate = self._sigmoid.forward(gates[1 * hs: 2 * hs])
        g_gate = np.tanh(gates[2 * hs: 3 * hs])
        o_gate = self._sigmoid.forward(gates[3 * hs: 4 * hs])

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * np.tanh(c_next)

        return h_next, c_next

    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"LSTMCell({self.input_size}, {self.hidden_size})"


# ---------------------------------------------------------------------------
# Simple MLP Forecaster
# ---------------------------------------------------------------------------

class SimpleForecaster:
    """Two-layer MLP for univariate time series forecasting.

    Takes a lookback window of past values and produces *horizon*
    future predictions.

    Architecture::

        Input(lookback) -> Linear -> ReLU -> Linear -> Output(horizon)

    Parameters
    ----------
    lookback : int
        Number of past observations used as input features.
    horizon : int
        Number of future steps to forecast.
    hidden : int
        Width of the hidden layer.
    """

    def __init__(self, lookback: int = 20, horizon: int = 5, hidden: int = 64):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden = hidden

        self.layer1 = Linear(lookback, hidden)
        self.relu = ReLU()
        self.layer2 = Linear(hidden, horizon)

        # Normalization stats (set during training)
        self._train_mean: float = 0.0
        self._train_std: float = 1.0
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass through the network.

        Parameters
        ----------
        x : (batch, lookback) array

        Returns
        -------
        (batch, horizon) array
        """
        z = self.layer1.forward(x)
        z = self.relu.forward(z)
        z = self.layer2.forward(z)
        return z

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        series: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
    ) -> list[float]:
        """Train the forecaster on a univariate time series.

        Creates sliding windows of length ``lookback + horizon`` from
        *series*, normalizes them to zero-mean / unit-variance, and runs
        mini-batch SGD (full-batch per epoch) with back-propagation.

        Parameters
        ----------
        series : 1-D array
            Training time series (length must be >= lookback + horizon + 1).
        epochs : int
            Number of training iterations over the full dataset.
        lr : float
            Learning rate for SGD weight updates.

        Returns
        -------
        list[float]
            MSE loss recorded at every epoch.
        """
        series = np.asarray(series, dtype=np.float64).ravel()

        min_len = self.lookback + self.horizon + 1
        if len(series) < min_len:
            raise ValueError(
                f"Series length ({len(series)}) must be >= "
                f"lookback + horizon + 1 ({min_len})"
            )

        # Compute and store normalization statistics
        self._train_mean = float(np.mean(series))
        self._train_std = float(np.std(series))
        if self._train_std == 0:
            self._train_std = 1.0

        normed = (series - self._train_mean) / self._train_std

        # Build sliding windows ------------------------------------------------
        n_windows = len(normed) - self.lookback - self.horizon + 1
        X = np.zeros((n_windows, self.lookback))
        Y = np.zeros((n_windows, self.horizon))
        for i in range(n_windows):
            X[i] = normed[i : i + self.lookback]
            Y[i] = normed[i + self.lookback : i + self.lookback + self.horizon]

        # Training loop --------------------------------------------------------
        loss_history: list[float] = []
        rng = np.random.default_rng()

        for epoch in range(epochs):
            # Shuffle windows
            idx = rng.permutation(n_windows)
            X_shuf = X[idx]
            Y_shuf = Y[idx]

            # Forward pass
            output = self._forward(X_shuf)

            # MSE loss
            diff = output - Y_shuf
            loss = float(np.mean(diff ** 2))
            loss_history.append(loss)

            # Backward pass ----------------------------------------------------
            # d_loss / d_output  =  2 * (output - target) / N
            d_loss = 2.0 * diff / diff.size

            d_z = self.layer2._backward(d_loss)
            d_z = self.relu._backward(d_z)
            self.layer1._backward(d_z)

            # SGD update -------------------------------------------------------
            self.layer2.W -= lr * self.layer2._dW
            self.layer2.b -= lr * self.layer2._db
            self.layer1.W -= lr * self.layer1._dW
            self.layer1.b -= lr * self.layer1._db

            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}/{epochs}  loss={loss:.6f}")

        self._trained = True
        return loss_history

    def fit(
        self,
        series: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
    ) -> list[float]:
        """Alias for :meth:`train` (matches ARIMA / Prophet API)."""
        return self.train(series, epochs=epochs, lr=lr)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, series: np.ndarray) -> np.ndarray:
        """Generate a forecast from the last ``lookback`` values of *series*.

        If the model has been trained, the input is normalized using the
        training statistics and the output is denormalized back to the
        original scale.

        Parameters
        ----------
        series : (>= lookback,) 1-D array

        Returns
        -------
        (horizon,) array of predictions.
        """
        series = np.asarray(series, dtype=np.float64).ravel()
        if len(series) < self.lookback:
            raise ValueError(
                f"Series length ({len(series)}) must be >= lookback ({self.lookback})"
            )
        window = series[-self.lookback:]

        if self._trained:
            window = (window - self._train_mean) / self._train_std

        # Reshape to (1, lookback) for the forward pass then squeeze back
        z = self._forward(window.reshape(1, -1)).ravel()

        if self._trained:
            z = z * self._train_std + self._train_mean

        return z

    def parameters(self) -> list[np.ndarray]:
        return self.layer1.parameters() + self.layer2.parameters()

    def __repr__(self) -> str:
        return (
            f"SimpleForecaster(lookback={self.lookback}, "
            f"horizon={self.horizon}, hidden={self.hidden})"
        )
