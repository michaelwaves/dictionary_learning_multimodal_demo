"""Shared utilities for activation gathering across video models."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SHARD_FILENAME_TEMPLATE = "shard_{:04d}.pt"
METADATA_FILENAME = "metadata.json"
PROGRESS_FILENAME = "progress.json"
RUN_CONFIG_FILENAME = "run_config.json"
FRAME_LOAD_TIMEOUT_SECONDS = 60

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

TEMPORAL_STRIDE = {"ltx": 8, "wan": 4}
SPATIAL_DIVISOR = {"ltx": 32, "wan": 16}
DEFAULT_MODEL_NAMES = {
    "ltx": "Lightricks/LTX-Video-0.9.5",
    "wan": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}
MIN_VAE_CHUNK_FRAMES = 9


def make_run_dir(base: str = "activations/runs") -> str:
    run_dir = os.path.join(base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir: str, config_dict: dict):
    path = os.path.join(run_dir, RUN_CONFIG_FILENAME)
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def remove_norm_outliers(
    activations: torch.Tensor, threshold_multiple: int,
) -> torch.Tensor:
    if activations.numel() == 0:
        return activations
    norms = activations.norm(dim=-1)
    median_norm = norms.median()
    return activations[norms <= median_norm * threshold_multiple]


def _resolve_module(root: torch.nn.Module, module_path: str) -> torch.nn.Module:
    current = root
    for part in module_path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


@contextmanager
def multi_module_hooks(root: torch.nn.Module, module_paths: list[str]):
    captured = {}
    handles = []
    for path in module_paths:
        target = _resolve_module(root, path)

        def _make_hook(p):
            def hook_fn(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[p] = tensor.detach()
            return hook_fn

        handles.append(target.register_forward_hook(_make_hook(path)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def preprocess_frames(
    frames: list[np.ndarray], height: int, width: int, device: str,
) -> torch.Tensor:
    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    t = t.to(device=device, dtype=torch.float32).div_(255.0)
    t = F.interpolate(
        t, size=(height, width),
        mode="bicubic", align_corners=False, antialias=True,
    ).clamp_(0.0, 1.0)
    t = t * 2.0 - 1.0
    return t.permute(1, 0, 2, 3).unsqueeze(0)


def reshape_vae_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 5:
        B, C, T, H, W = activation.shape
        return activation.permute(0, 2, 3, 4, 1).reshape(-1, C)
    elif activation.ndim == 4:
        B, C, H, W = activation.shape
        return activation.permute(0, 2, 3, 1).reshape(-1, C)
    elif activation.ndim == 3:
        return activation.squeeze(0)
    return activation.reshape(-1, activation.shape[-1])


def reshape_transformer_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 3:
        return activation.squeeze(0)
    elif activation.ndim > 3:
        return activation.reshape(-1, activation.shape[-1])
    return activation


class ShardWriter:
    def __init__(self, module_dir: str):
        self.module_dir = module_dir
        os.makedirs(module_dir, exist_ok=True)
        self.shard_index = self._count_existing_shards()
        self.buffer: list[torch.Tensor] = []
        self.total_tokens = 0

    def append(self, activations: torch.Tensor):
        self.buffer.append(activations.cpu().to(torch.float32))

    def flush(self):
        if not self.buffer:
            return
        shard_data = torch.cat(self.buffer, dim=0)
        shard_path = os.path.join(
            self.module_dir, SHARD_FILENAME_TEMPLATE.format(self.shard_index),
        )
        torch.save(shard_data, shard_path)
        self.total_tokens += shard_data.shape[0]
        logger.info(f"{self.module_dir}: shard {self.shard_index} — {shard_data.shape[0]} tokens")
        self.shard_index += 1
        self.buffer = []

    def _count_existing_shards(self) -> int:
        if not os.path.exists(self.module_dir):
            return 0
        return sum(1 for f in os.listdir(self.module_dir)
                   if f.startswith("shard_") and f.endswith(".pt"))


class ProgressTracker:
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, PROGRESS_FILENAME)
        self.processed: set[str] = self._load()

    def mark_done(self, video_path: str):
        self.processed.add(video_path)

    def is_done(self, video_path: str) -> bool:
        return video_path in self.processed

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"processed_videos": sorted(self.processed)}, f)

    def _load(self) -> set[str]:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return set(json.load(f).get("processed_videos", []))
        return set()


class FramePrefetcher:
    def __init__(self, video_paths, num_frames, max_workers, frame_reader=None):
        from video_utils import read_video_pyav
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._video_paths = video_paths
        self._num_frames = num_frames
        self._frame_reader = frame_reader or read_video_pyav
        self._futures = {}
        self._next_submit = 0
        for _ in range(min(max_workers, len(video_paths))):
            self._submit_next()

    def get_frames(self, video_index: int) -> list[np.ndarray] | None:
        if video_index in self._futures:
            future = self._futures.pop(video_index)
            self._submit_next()
            try:
                return future.result(timeout=FRAME_LOAD_TIMEOUT_SECONDS)
            except Exception as e:
                logger.debug(f"Prefetch failed for {self._video_paths[video_index]}: {e}")
                return None
        try:
            return self._frame_reader(self._video_paths[video_index], self._num_frames)
        except Exception as e:
            logger.debug(f"Sync load failed for {self._video_paths[video_index]}: {e}")
            return None

    def shutdown(self):
        self._pool.shutdown(wait=False)

    def _submit_next(self):
        if self._next_submit < len(self._video_paths):
            idx = self._next_submit
            self._futures[idx] = self._pool.submit(
                self._frame_reader, self._video_paths[idx], self._num_frames,
            )
            self._next_submit += 1


def load_model_vae(model_name: str, vae_type: str, device: str):
    if vae_type == "wan":
        from wan_model import load_wan_vae
        return load_wan_vae(model_name, device)
    from ltx_model import load_ltx_vae
    return load_ltx_vae(model_name, device, enable_tiling=False)


def align_to_vae_frame_count(n: int, temporal_stride: int = 8) -> int:
    if n <= 1:
        return 1
    return ((n - 2) // temporal_stride + 1) * temporal_stride + 1


def chunk_frames_for_vae(frames: list, chunk_size: int, temporal_stride: int = 8) -> list[list]:
    chunks = []
    for start in range(0, len(frames), chunk_size):
        chunk = frames[start:start + chunk_size]
        if len(chunk) < MIN_VAE_CHUNK_FRAMES:
            break
        aligned_size = align_to_vae_frame_count(len(chunk), temporal_stride)
        while len(chunk) < aligned_size:
            chunk.append(chunk[-1])
        chunks.append(chunk)
    return chunks
