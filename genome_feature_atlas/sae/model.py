"""TopK Sparse Autoencoder matching the ESMC SAE architecture.

Architecture: x → center → ReLU-encoder → TopK → linear-decoder → x_hat
Parameters: d_model=1536, n_features=16384, k=64 (matching ESMC-6B-sae-k64-codebook16384).

Loss: MSE reconstruction + AuxK auxiliary loss to prevent dead features.

Adapted from the Biohub ESM repository:
  https://github.com/Biohub/esm/blob/main/cookbook/snippets/sae.py
The TopKSAE class and SAEConfig are derived nearly verbatim from that source.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int = 1536
    n_features: int = 16384
    k: int = 64
    k_aux: int = 512
    aux_loss_coeff: float = 1 / 32
    dead_threshold: float = 1e-4
    ema_decay: float = 0.999


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Matches the ESMC SAE architecture:
    - Encoder: W_enc (d_model → n_features), bias b_enc
    - Activation: ReLU then TopK (keep top-k activations per token)
    - Decoder: W_dec (n_features → d_model), shared bias b_pre
    - Decoder columns maintained at unit norm

    b_pre is subtracted before encoding and added after decoding (standard
    "pre-encoder bias" / "decoder bias" shared trick from Anthropic / ESMC SAEs).
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_model, cfg.n_features

        self.b_pre = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.empty(d, n))
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.W_dec = nn.Parameter(torch.empty(n, d))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self._normalize_decoder_inplace()

        # Dead-feature tracking (not a Parameter — not saved by default,
        # but included in state_dict via register_buffer so it survives checkpoints)
        self.register_buffer("feature_ema", torch.zeros(n))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_sparse, pre_acts).

        z_sparse: (B, n_features) with exactly k non-zeros per row.
        pre_acts: (B, n_features) ReLU activations before TopK masking.
        """
        x_centered = x - self.b_pre
        pre_acts = F.relu(x_centered @ self.W_enc + self.b_enc)
        topk_vals, topk_idx = pre_acts.topk(self.cfg.k, dim=-1)
        z = torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)
        return z, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, z, pre_acts)."""
        z, pre_acts = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, pre_acts

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        pre_acts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute reconstruction loss + AuxK loss.

        AuxK loss: make top-k_aux *dead* features reconstruct the residual,
        preventing feature collapse without L1 penalty.
        """
        recon = F.mse_loss(x_hat, x)

        dead_mask = self.feature_ema < self.cfg.dead_threshold
        n_dead = int(dead_mask.sum().item())

        if n_dead > 0:
            k_aux = min(self.cfg.k_aux, n_dead)
            dead_pre = pre_acts * dead_mask.float()
            topk_aux_vals, topk_aux_idx = dead_pre.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(pre_acts).scatter_(-1, topk_aux_idx, topk_aux_vals)
            x_hat_aux = self.decode(z_aux)
            # AuxK target: reproduce x (same as recon loss, but only through dead features)
            # Detach x_hat so gradients flow only through dead-feature path
            aux = F.mse_loss(x_hat_aux, x_hat.detach() + (x - x_hat).detach())
        else:
            aux = x.new_tensor(0.0)

        total = recon + self.cfg.aux_loss_coeff * aux
        metrics = {
            "loss": total.item(),
            "recon": recon.item(),
            "aux": aux.item(),
            "dead_frac": dead_mask.float().mean().item(),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self):
        """Clamp decoder columns to unit norm. Call after every optimizer step."""
        self._normalize_decoder_inplace()

    @torch.no_grad()
    def _normalize_decoder_inplace(self):
        norms = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data.div_(norms.clamp(min=1.0))

    @torch.no_grad()
    def update_feature_ema(self, z: torch.Tensor):
        """Update EMA of per-feature activation rates from a batch."""
        activation_rate = (z > 0).float().mean(dim=0)
        self.feature_ema.mul_(self.cfg.ema_decay).add_(
            activation_rate * (1 - self.cfg.ema_decay)
        )

    @torch.no_grad()
    def init_from_data(self, sample: torch.Tensor):
        """Set b_pre to the mean of a sample of training embeddings.

        Call once before training begins. sample: (N, d_model) float32.
        """
        self.b_pre.data.copy_(sample.mean(dim=0))
        # Re-sync W_dec to normalized W_enc.T after b_pre shift
        self.W_dec.data = self.W_enc.data.T.clone()
        self._normalize_decoder_inplace()

    # ------------------------------------------------------------------
    # Sparsity diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def l0(self, z: torch.Tensor) -> float:
        """Mean number of active features per token."""
        return (z > 0).float().sum(dim=-1).mean().item()
