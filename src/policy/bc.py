# src/policy/bc.py
"""
Behavior Cloning (BC) — simple MLP baseline.

Maps a normalised observation (32-dim) → action (16-dim) with a small
feed-forward network.  Fast to train, easy to debug, and surprisingly
effective as a first policy.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    Feed-forward Behavior Cloning policy.

    Parameters
    ----------
    obs_dim : observation dimensionality (default 32 = 16 qpos + 16 qvel)
    act_dim : action dimensionality    (default 16 = finger ctrl)
    hidden  : list of hidden-layer widths
    dropout : dropout probability between layers (0 = off)
    """

    def __init__(
        self,
        obs_dim: int = 32,
        act_dim: int = 16,
        hidden: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden or [256, 256]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → action: (B, act_dim)"""
        return self.net(obs)

    # ── Convenience ────────────────────────────────────────────────────────

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and target actions."""
        return nn.functional.mse_loss(pred, target)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor,
                norm: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Normalise obs, forward, then un-normalise action.

        *norm* should contain ``obs_mean, obs_std, act_mean, act_std``.
        """
        self.eval()
        if norm is not None:
            dev = obs.device
            obs = (obs - norm["obs_mean"].to(dev)) / norm["obs_std"].to(dev)
        act = self.forward(obs)
        if norm is not None:
            act = act * norm["act_std"].to(dev) + norm["act_mean"].to(dev)
        return act
