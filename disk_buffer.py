"""Disk-backed activation buffer for SAE training on pre-gathered activations.

Loads sharded .pt files written by gather_video_activations.py and provides
the same iterator interface as ActivationBuffer: yields [batch_size, d_model]
tensors indefinitely, cycling through shards with shuffling.
"""

import json
import logging
import os
import random

import torch

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"


class DiskActivationBuffer:
    """Streams pre-gathered activation shards from disk with in-memory shuffling.

    Maintains a rolling in-memory buffer, loads new shards when the buffer
    runs low, and yields random batches of the requested size. Cycles through
    all shards indefinitely, reshuffling shard order each epoch.

    This class is a drop-in replacement for ActivationBuffer — the trainSAE()
    function only needs an iterator yielding [batch_size, d_model] tensors.
    """

    def __init__(
        self,
        activation_dir: str,
        out_batch_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        shards_in_memory: int = 4,
        shuffle_shards: bool = True,
    ):
        self.activation_dir = activation_dir
        self.out_batch_size = out_batch_size
        self.device = device
        self.dtype = dtype
        self.shards_in_memory = shards_in_memory
        self.shuffle_shards = shuffle_shards

        self.metadata = self._load_metadata()
        self.d_submodule = self.metadata["d_model"]
        self.shard_paths = self._discover_shards()

        if not self.shard_paths:
            raise RuntimeError(f"No shard files found in {activation_dir}")

        logger.info(
            f"DiskActivationBuffer: {len(self.shard_paths)} shards, "
            f"d_model={self.d_submodule}, batch_size={out_batch_size}"
        )

        self._shard_order = list(range(len(self.shard_paths)))
        self._shard_cursor = 0
        self._init_buffer()

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        unread_indices = self._unread_indices()
        if len(unread_indices) < self._half_capacity:
            self._refill()
            unread_indices = self._unread_indices()

        batch_size = min(self.out_batch_size, len(unread_indices))
        selected = unread_indices[torch.randperm(len(unread_indices))[:batch_size]]
        self._read_mask[selected] = True
        return self._buffer[selected].to(dtype=self.dtype)

    def _unread_indices(self) -> torch.Tensor:
        """Return 1-D tensor of indices for unread buffer positions."""
        indices = (~self._read_mask).nonzero().squeeze()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        return indices

    @property
    def config(self) -> dict:
        return {
            "activation_dir": self.activation_dir,
            "d_submodule": self.d_submodule,
            "out_batch_size": self.out_batch_size,
            "num_shards": len(self.shard_paths),
            "metadata": self.metadata,
        }

    def _init_buffer(self):
        """Load initial shards into memory."""
        initial_data = self._load_next_shards(self.shards_in_memory)
        self._buffer = initial_data.to(self.device)
        self._read_mask = torch.zeros(len(self._buffer), dtype=torch.bool, device=self.device)
        self._half_capacity = len(self._buffer) // 2

    def _refill(self):
        """Replace read activations with fresh data from the next shard(s)."""
        unread = self._buffer[~self._read_mask]
        new_data = self._load_next_shards(1)
        self._buffer = torch.cat([unread, new_data.to(self.device)], dim=0)
        self._read_mask = torch.zeros(len(self._buffer), dtype=torch.bool, device=self.device)
        self._half_capacity = len(self._buffer) // 2

    def _load_next_shards(self, count: int) -> torch.Tensor:
        """Load `count` shards from disk, cycling and reshuffling as needed."""
        chunks = []
        for _ in range(count):
            if self._shard_cursor >= len(self._shard_order):
                self._start_new_epoch()
            shard_idx = self._shard_order[self._shard_cursor]
            shard_data = torch.load(self.shard_paths[shard_idx], weights_only=True)
            chunks.append(shard_data)
            self._shard_cursor += 1
        return torch.cat(chunks, dim=0)

    def _start_new_epoch(self):
        """Reshuffle shard order and reset cursor."""
        if self.shuffle_shards:
            random.shuffle(self._shard_order)
        self._shard_cursor = 0
        logger.info("DiskActivationBuffer: starting new epoch over shards")

    def _load_metadata(self) -> dict:
        metadata_path = os.path.join(self.activation_dir, METADATA_FILENAME)
        with open(metadata_path, "r") as f:
            return json.load(f)

    def _discover_shards(self) -> list[str]:
        shard_files = sorted(
            f for f in os.listdir(self.activation_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )
        return [os.path.join(self.activation_dir, f) for f in shard_files]
