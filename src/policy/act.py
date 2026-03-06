# src/policy/act.py
"""
Action Chunking with Transformers (ACT).

Simplified re-implementation inspired by Tony Zhao et al., 2023:
    "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"

Key ideas
---------
* **CVAE encoder** (training only): takes the *full* future action chunk +
  observation and produces a latent ``z ~ N(mu, sigma)``.
* **Transformer decoder**: takes ``[z; obs]`` as prefix tokens, uses learned
  action queries to decode ``chunk_size`` future actions in parallel.
* **KL loss** regularises the latent to N(0, I); MSE loss on predicted actions.

At inference the encoder is bypassed and ``z`` is sampled from the prior.

This file is self-contained: no external ACT library needed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (length, d_model)."""
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# CVAE Encoder (training only)
# ─────────────────────────────────────────────────────────────────────────────

class _CVAEEncoder(nn.Module):
    """
    Encodes (obs, action_chunk) → (mu, logvar) for the latent z.
    Uses a small Transformer encoder over the action sequence, conditioned
    on the observation via a CLS-style token.
    """

    def __init__(self, obs_dim: int, act_dim: int, chunk_size: int,
                 d_model: int, n_heads: int, n_layers: int, latent_dim: int):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(_sinusoidal_pe(chunk_size + 2, d_model),
                                      requires_grad=False)  # +2 for cls & obs

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        obs:     (B, obs_dim)
        actions: (B, chunk, act_dim)
        → mu, logvar: each (B, latent_dim)
        """
        B = obs.shape[0]
        cls = self.cls_token.expand(B, -1, -1)                    # (B,1,d)
        obs_tok = self.obs_proj(obs).unsqueeze(1)                 # (B,1,d)
        act_tok = self.act_proj(actions)                          # (B,chunk,d)
        seq = torch.cat([cls, obs_tok, act_tok], dim=1)           # (B, 2+chunk, d)
        seq = seq + self.pos_embed[: seq.size(1)]

        out = self.encoder(seq)                                    # (B, 2+chunk, d)
        cls_out = out[:, 0]                                        # (B, d)
        return self.mu_head(cls_out), self.logvar_head(cls_out)


# ─────────────────────────────────────────────────────────────────────────────
# ACT Policy
# ─────────────────────────────────────────────────────────────────────────────

class ACTPolicy(nn.Module):
    """
    Action-Chunking Transformer policy with CVAE latent.

    Parameters
    ----------
    obs_dim     : observation dim (default 32)
    act_dim     : action dim (default 16)
    chunk_size  : number of future actions to predict (default 10)
    d_model     : internal transformer width (default 256)
    n_heads     : attention heads (default 4)
    enc_layers  : CVAE encoder layers (default 2)
    dec_layers  : decoder layers (default 4)
    latent_dim  : CVAE latent dimension (default 32)
    kl_weight   : weight on KL loss term (default 10.0)
    """

    def __init__(
        self,
        obs_dim: int = 32,
        act_dim: int = 16,
        chunk_size: int = 10,
        d_model: int = 256,
        n_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 4,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # ── CVAE encoder (training) ─────────────────────────────────────────
        self.cvae_encoder = _CVAEEncoder(
            obs_dim, act_dim, chunk_size,
            d_model, n_heads, enc_layers, latent_dim,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.z_proj   = nn.Linear(latent_dim, d_model)

        # Learned action queries — one per future timestep
        self.action_queries = nn.Parameter(
            torch.randn(1, chunk_size, d_model) * 0.02
        )
        self.query_pe = nn.Parameter(
            _sinusoidal_pe(chunk_size, d_model), requires_grad=False
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.action_head = nn.Linear(d_model, act_dim)

    # ── Forward (training) ──────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Training forward pass.

        obs:     (B, obs_dim)
        actions: (B, chunk_size, act_dim)  — ground-truth future chunk
        Returns: pred_actions (B, chunk, act_dim), kl_loss (scalar)
        """
        B = obs.shape[0]

        # 1. Encode → latent
        mu, logvar = self.cvae_encoder(obs, actions)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)          # reparameterisation

        # 2. Decode
        pred = self._decode(obs, z)                    # (B, chunk, act_dim)

        # 3. KL divergence  D_KL(q(z|x,a) || N(0,I))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return pred, kl

    def _decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Transformer decode: [z; obs] as memory, action_queries as target."""
        B = obs.shape[0]
        z_tok   = self.z_proj(z).unsqueeze(1)            # (B,1,d)
        obs_tok = self.obs_proj(obs).unsqueeze(1)         # (B,1,d)
        memory  = torch.cat([z_tok, obs_tok], dim=1)      # (B,2,d)

        queries = self.action_queries.expand(B, -1, -1) + self.query_pe   # (B,chunk,d)

        out = self.decoder(queries, memory)                               # (B,chunk,d)
        return self.action_head(out)                                      # (B,chunk,act)

    # ── Loss ────────────────────────────────────────────────────────────────

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor,
                kl: torch.Tensor) -> torch.Tensor:
        """Reconstruction (MSE) + weighted KL."""
        recon = F.mse_loss(pred, target)
        return recon + self.kl_weight * kl

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, obs: torch.Tensor,
                norm: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Sample from prior z ~ N(0,I), decode → action chunk.

        Returns the **first action** of the chunk: (B, act_dim).
        (Temporal ensembling across overlapping chunks is handled externally.)
        """
        self.eval()
        dev = obs.device
        if norm is not None:
            obs = (obs - norm["obs_mean"].to(dev)) / norm["obs_std"].to(dev)

        B = obs.shape[0]
        z = torch.randn(B, self.latent_dim, device=dev)
        chunk = self._decode(obs, z)                     # (B, chunk, act)

        if norm is not None:
            chunk = chunk * norm["act_std"].to(dev) + norm["act_mean"].to(dev)
        return chunk[:, 0]                                # first action only
