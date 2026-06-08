"""Orchestrates loading of per-feature top-activation loci from the feature scan HDF5.

Usage:
    from genome_feature_atlas.feature_loci import FeatureLociLoader

    loader = FeatureLociLoader("results/feature_top200.h5")
    loci = loader.get_loci(feature_id=42, n_loci=10)
    # loci: [(chrom, start, end, activation), ...]
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class FeatureLociLoader:
    """Wraps the feature scan HDF5 produced by scan_features.py.

    Arrays are loaded eagerly at construction (≈50 MB for 16384×200). Coordinates
    follow the same 0-based half-open convention used throughout the project:
    each bin covers [start, start + resolution_bp).

    Args:
        h5_path: path to feature_top200.h5 (or similar) from scan_features.py
    """

    def __init__(self, h5_path: Path | str) -> None:
        h5_path = Path(h5_path)
        with h5py.File(h5_path, "r") as f:
            self._activations: np.ndarray         = f["activations"][:]           # (n_features, n_top) float32
            self._starts: np.ndarray              = f["starts"][:]                # (n_features, n_top) int32
            self._chrom_idx: np.ndarray           = f["chrom_idx"][:]             # (n_features, n_top) uint8
            self._chrom_names: list[str]          = list(f["chrom_names"].asstr()[:])
            self._total_active_counts: np.ndarray = f["total_active_counts"][:]   # (n_features,) int32
            self._resolution_bp: int              = int(f.attrs["resolution_bp"])
            self._window_stride_bp: int           = int(f.attrs["window_stride_bp"])

    # ── public API ────────────────────────────────────────────────────────────

    def get_loci(
        self,
        feature_id: int,
        n_loci: int = 10,
    ) -> list[tuple[str, int, int, float]]:
        """Return the top-N active loci for a feature.

        Args:
            feature_id: SAE feature index (0-based)
            n_loci:     maximum number of loci to return

        Returns:
            List of (chrom, start, end, activation) tuples sorted descending by
            activation; only loci with activation > 0 are included.
        """
        vals   = self._activations[feature_id]   # (n_top,) sorted desc
        starts = self._starts[feature_id]
        cidxs  = self._chrom_idx[feature_id]

        result: list[tuple[str, int, int, float]] = []
        for i in range(len(vals)):
            act = float(vals[i])
            if act <= 0:
                break
            if len(result) >= n_loci:
                break
            chrom = self._chrom_names[int(cidxs[i])]
            start = int(starts[i])
            end   = start + self._resolution_bp
            result.append((chrom, start, end, act))
        return result

    def activation_stats(self, feature_id: int, n_loci: int) -> dict:
        """Return activation distribution stats for the top-N loci of a feature.

        Returns:
            max_val:              activation of the top locus
            min_shown_val:        activation of the nth locus (weakest shown)
            flatness:             min_shown_val / max_val  (1.0 = flat, near 0 = sharp cliff)
            total_active_windows: genome-wide count of non-overlapping windows with activation > 0
            window_stride_bp:     width of each window in bp
        """
        acts = self._activations[feature_id, :n_loci]
        acts = acts[acts > 0]
        if len(acts) == 0:
            return {
                "max_val": 0.0, "min_shown_val": 0.0, "flatness": 0.0,
                "total_active_windows": 0, "window_stride_bp": self._window_stride_bp,
            }
        max_val   = float(acts[0])
        min_shown = float(acts[-1])
        return {
            "max_val":              round(max_val, 4),
            "min_shown_val":        round(min_shown, 4),
            "flatness":             round(min_shown / max_val, 3),
            "total_active_windows": int(self._total_active_counts[feature_id]),
            "window_stride_bp":     self._window_stride_bp,
        }

    def global_activation_background(self, n_loci: int) -> dict:
        """Percentile stats across ALL active features — instant numpy, no tabix.

        Provides a population reference for comparing individual feature stats.
        `flatness` and `windows` are derived the same way as `activation_stats()`.
        """
        active = np.array(self.active_feature_ids)
        max_vals = self._activations[active, 0]

        tail_idx  = min(n_loci - 1, self._activations.shape[1] - 1)
        tail_vals = self._activations[active, tail_idx].astype(float)
        tail_vals[tail_vals <= 0] = np.nan  # dead/short features → NaN
        flatness  = np.where(max_vals > 0, tail_vals / max_vals, np.nan)
        windows   = self._total_active_counts[active].astype(float)

        def _p(arr: np.ndarray, *pcts: int) -> list[float]:
            return [round(float(np.nanpercentile(arr, p)), 4) for p in pcts]

        return {
            "n_active_features": int(len(active)),
            "max_val":  dict(zip(("p25", "p50", "p75"), _p(max_vals, 25, 50, 75))),
            "flatness": dict(zip(("p25", "p50", "p75"), _p(flatness, 25, 50, 75))),
            "windows":  dict(zip(("p25", "p50", "p75"), _p(windows,  25, 50, 75))),
        }

    @property
    def n_features(self) -> int:
        return int(self._activations.shape[0])

    @property
    def resolution_bp(self) -> int:
        return self._resolution_bp

    @property
    def window_stride_bp(self) -> int:
        return self._window_stride_bp

    @property
    def active_feature_ids(self) -> list[int]:
        """Feature indices that have at least one positive activation."""
        has_active = (self._activations[:, 0] > 0)
        return list(np.where(has_active)[0])
