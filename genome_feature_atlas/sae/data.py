"""HDF5 dataset for AlphaGenome trunk embeddings.

Reads float16 embeddings from /{chrom}/embeddings (shape: n_bins × 1536)
and returns float32 chunks. Designed for multi-worker DataLoader: the HDF5
file handle is opened lazily in each worker process to avoid fork-safety issues.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingH5Dataset(Dataset):
    """Random-access dataset over chunked HDF5 embeddings.

    Each item is a contiguous chunk of `chunk_size` consecutive genomic bins
    from a single chromosome, returned as a float32 tensor of shape
    (chunk_size, embedding_dim).  The last chunk of a chromosome may be
    smaller than chunk_size.

    Args:
        h5_path: Path to HDF5 file produced by extract_alphagenome_embeddings.py.
        chromosomes: If given, only use these chromosome keys (e.g. ["chr21"]).
            Useful for train/val splits.
        chunk_size: Number of consecutive bins per dataset item.  Larger values
            reduce HDF5 seek overhead; smaller values give finer-grained shuffling.
    """

    def __init__(
        self,
        h5_path: str | Path,
        chromosomes: list[str] | None = None,
        chunk_size: int = 4096,
    ):
        self.h5_path = str(h5_path)
        self.chunk_size = chunk_size
        self._file: h5py.File | None = None

        # Build index before forking so all workers share the same view
        with h5py.File(self.h5_path, "r") as f:
            available = sorted(f.keys())
            if chromosomes is not None:
                missing = [c for c in chromosomes if c not in f]
                if missing:
                    raise ValueError(f"Chromosomes not in HDF5: {missing}")
                available = [c for c in chromosomes if c in f]

            self._index: list[tuple[str, int, int]] = []
            self.embedding_dim: int = int(f.attrs.get("embedding_dim", 1536))

            for chrom in available:
                n_bins = f[chrom]["embeddings"].shape[0]
                for start in range(0, n_bins, chunk_size):
                    end = min(start + chunk_size, n_bins)
                    self._index.append((chrom, start, end))

        self._chroms_used = available

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        chrom, start, end = self._index[idx]
        emb = self._file[chrom]["embeddings"][start:end]  # float16 numpy
        return torch.from_numpy(emb.astype(np.float32))

    # ------------------------------------------------------------------
    # File handle lifecycle (fork-safe lazy open)
    # ------------------------------------------------------------------

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __getstate__(self):
        # Close before pickling so each worker opens its own handle
        state = self.__dict__.copy()
        if state["_file"] is not None:
            state["_file"].close()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_bins(self) -> int:
        """Total number of genomic bins across all included chromosomes."""
        return sum(end - start for _, start, end in self._index)

    @property
    def chromosomes(self) -> list[str]:
        return list(self._chroms_used)

    def sample_for_init(self, n: int = 16384, device: str = "cpu") -> torch.Tensor:
        """Draw n random bins for SAE init_from_data (b_pre initialization).

        Returns a (n, embedding_dim) float32 tensor.
        """
        total = self.n_bins
        chosen = np.random.choice(total, size=min(n, total), replace=False)
        chosen_sorted = np.sort(chosen)

        # Build a flat bin-to-(chrom, local_idx) mapping efficiently
        # by iterating through index chunks
        samples = []
        bin_offset = 0
        ptr = 0
        for chrom, start, end in self._index:
            chunk_len = end - start
            # Find which chosen indices fall in this chunk
            while ptr < len(chosen_sorted) and chosen_sorted[ptr] < bin_offset + chunk_len:
                local_idx = int(chosen_sorted[ptr] - bin_offset)
                self._ensure_open()
                emb = self._file[chrom]["embeddings"][start + local_idx]
                samples.append(emb.astype(np.float32))
                ptr += 1
            bin_offset += chunk_len
            if ptr >= len(chosen_sorted):
                break

        arr = np.stack(samples, axis=0)
        return torch.from_numpy(arr).to(device)
