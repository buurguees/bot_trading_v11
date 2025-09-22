"""
core/environments/trading_env.py
---------------------------------
Entorno Gym mínimo para ejecución PPO basada en barras/feature windows.

Nota: Este es un entorno base para arrancar; se puede enriquecer con
gestión de posición, riesgos y más señales/observaciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, lookback: int = 60, fee: float = 0.0001, slippage: float = 0.00005,
                 features: list[str] | None = None, normalization: str = "zscore", risk_penalty: float = 0.0):
        assert {"close"}.issubset(df.columns), "df must have close"
        self.df = df.reset_index(drop=True)
        self.lookback = int(lookback)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.risk_penalty = float(risk_penalty)

        # Observación: ventana [lookback, features]
        self.features = features if features else ["close"]
        self.features = [f for f in self.features if f in self.df.columns]
        if not self.features:
            self.features = ["close"]
        self.normalization = normalization
        obs_shape = (self.lookback, len(self.features))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Acciones: 0=hold, 1=open_long, 2=open_short, 3=close
        self.action_space = spaces.Discrete(4)

        # Estado interno
        self.idx = self.lookback
        self.position = 0  # -1,0,1
        self.entry_price = None
        self.equity = 1.0

    def _get_obs(self):
        window = self.df.loc[self.idx - self.lookback:self.idx - 1, self.features].values.astype(np.float32)
        if window.shape[0] >= 2:
            if self.normalization == "zscore":
                mu = window.mean(axis=0, keepdims=True)
                sd = window.std(axis=0, keepdims=True) + 1e-6
                window = (window - mu) / sd
            elif self.normalization == "minmax":
                mn = window.min(axis=0, keepdims=True)
                mx = window.max(axis=0, keepdims=True)
                window = (window - mn) / (mx - mn + 1e-6)
        return window

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.idx = self.lookback
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        return self._get_obs(), {}

    def step(self, action: int):
        done = False
        info = {}

        px = float(self.df.loc[self.idx, "close"])
        reward = 0.0

        # slippage/fee cost baseline
        def cost():
            return px * (self.fee + self.slippage)

        if action == 1:  # open_long
            if self.position == 0:
                self.position = 1
                self.entry_price = px
                reward -= cost()
        elif action == 2:  # open_short
            if self.position == 0:
                self.position = -1
                self.entry_price = px
                reward -= cost()
        elif action == 3:  # close
            if self.position != 0 and self.entry_price is not None:
                pnl = (px - self.entry_price) if self.position == 1 else (self.entry_price - px)
                pnl /= max(1e-8, self.entry_price)
                reward += float(pnl)
                reward -= cost()
                self.position = 0
                self.entry_price = None

        # paso de tiempo
        self.idx += 1
        # penalización por mantener posición (riesgo)
        if self.position != 0:
            reward -= self.risk_penalty
        if self.idx >= len(self.df):
            done = True

        obs = self._get_obs() if not done else np.zeros((self.lookback, len(self.features)), dtype=np.float32)
        return obs, float(reward), done, False, info

    def render(self):
        pass


