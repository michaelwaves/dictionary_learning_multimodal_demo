"""
Stage 1: Gather video activations from LTX-Video and save to disk.

Processes videos through LTX-Video (Lightricks/LTX-Video, 2B), captures
activations at specified hook modules via forward hooks, and saves sharded
.pt files for SAE training.

Supports two hook targets:
  hook_target="transformer": VAE encode -> noise -> transformer forward -> hook
  hook_target="vae_encoder": VAE encode -> hook (no transformer/T5 loaded)

Usage:
    python gather_ltx_activations.py \
        --video_dir /path/to/videos \
        --output_dir ./activations \
        --hook_target transformer \
        --hook_modules transformer_blocks.14 transformer_blocks.20 \
        --num_frames 17 \
        --device cuda:0

Hookable module paths -- transformer (relative to transformer):
  transformer_blocks.{0-27}        -> d=2048  (block output)
  transformer_blocks.{i}.attn1     -> self-attention output
  transformer_blocks.{i}.attn2     -> cross-attention output
  transformer_blocks.{i}.ff        -> feed-forward output
  proj_in                          -> d=2048  (input projection)

Hookable module paths -- vae_encoder (relative to vae):
  encoder.conv_in                          -> d=128
  encoder.down_blocks.0.resnets.{0-3}     -> d=128
  encoder.down_blocks.0                    -> d=256
  encoder.down_blocks.1.resnets.{0-2}     -> d=256
  encoder.down_blocks.1                    -> d=512
  encoder.down_blocks.2.resnets.{0-2}     -> d=512
  encoder.down_blocks.3.resnets.{0-2}     -> d=512
  encoder.mid_block                        -> d=512
  encoder.mid_block.resnets.{0-3}         -> d=512
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from video_utils import read_video_pyav, scan_video_directory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARD_FILENAME_TEMPLATE = "shard_{:04d}.pt"
METADATA_FILENAME = "metadata.json"
PROGRESS_FILENAME = "progress.json"
FRAME_LOAD_TIMEOUT_SECONDS = 60

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GatherLTXConfig:
    model_name: str = "Lightricks/LTX-Video"
    hook_target: str = "transformer"       # "transformer" or "vae_encoder"
    # e.g. ["transformer_blocks.14", "transformer_blocks.20"]
    hook_modules: list[str] = None
    video_dir: str = ""
    output_dir: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    # Video preprocessing
    num_frames: int = 17       # must satisfy (n-1)%8==0
    height: int = 256          # must be divisible by 32
    width: int = 256           # must be divisible by 32

    # Diffusion (only used for hook_target="transformer")
    text_prompt: str = ""      # empty = unconditional
    max_text_seq_len: int = 128
    num_train_timesteps: int = 1000
    shift: float = 1.0        # flow-match sigma shift

    # Saving
    shard_size: int = 50       # videos per shard flush
    max_norm_multiple: int = 10
    prefetch_workers: int = 4

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)

    def __post_init__(self):
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError(
                f"num_frames must satisfy (n-1)%8==0 for VAE temporal compression, "
                f"got {self.num_frames}. Valid values: 9, 17, 25, 33, ...")
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(
                f"height and width must be divisible by 32 for VAE spatial compression, "
                f"got {self.height}x{self.width}")


# ---------------------------------------------------------------------------
# Norm outlier filtering (same as Qwen gather script)
# ---------------------------------------------------------------------------


def remove_norm_outliers(
    activations: torch.Tensor,
    threshold_multiple: int,
) -> torch.Tensor:
    """Remove activations whose norm exceeds median_norm * threshold_multiple."""
    if activations.numel() == 0:
        return activations
    norms = activations.norm(dim=-1)
    median_norm = norms.median()
    within_threshold = norms <= median_norm * threshold_multiple
    return activations[within_threshold]


# ---------------------------------------------------------------------------
# Multi-module hook context manager
# ---------------------------------------------------------------------------


def _resolve_module(root: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """Resolve a dotted path like 'transformer_blocks.14' relative to root."""
    current = root
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


@contextmanager
def multi_module_hooks(
    root: torch.nn.Module,
    module_paths: list[str],
):
    """Context manager that registers forward hooks on multiple modules.

    Yields a dict populated with {module_path: activation_tensor} after forward.
    """
    captured = {}
    handles = []

    for path in module_paths:
        target = _resolve_module(root, path)

        def _make_hook(p):
            def hook_fn(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[p] = tensor.detach()
            return hook_fn

        handle = target.register_forward_hook(_make_hook(path))
        handles.append(handle)

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# ShardWriter & ProgressTracker (same as Qwen gather script)
# ---------------------------------------------------------------------------


class ShardWriter:
    """Accumulates activation tensors and writes them to numbered shard files."""

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
            self.module_dir, SHARD_FILENAME_TEMPLATE.format(self.shard_index)
        )
        torch.save(shard_data, shard_path)
        self.total_tokens += shard_data.shape[0]
        logger.info(
            f"{self.module_dir}: shard {self.shard_index} — {shard_data.shape[0]} tokens"
        )
        self.shard_index += 1
        self.buffer = []

    def _count_existing_shards(self) -> int:
        return sum(
            1 for f in os.listdir(self.module_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ) if os.path.exists(self.module_dir) else 0


class ProgressTracker:
    """Tracks which videos have been processed for crash-resume support."""

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


# ---------------------------------------------------------------------------
# Frame prefetcher
# ---------------------------------------------------------------------------


class FramePrefetcher:
    """Prefetches video frames in background threads while the GPU is busy."""

    def __init__(self, video_paths: list[str], num_frames: int, max_workers: int):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._video_paths = video_paths
        self._num_frames = num_frames
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
                logger.debug(
                    f"Prefetch failed for {self._video_paths[video_index]}: {e}")
                return None

        try:
            return read_video_pyav(self._video_paths[video_index], self._num_frames)
        except Exception as e:
            logger.debug(
                f"Sync load failed for {self._video_paths[video_index]}: {e}")
            return None

    def shutdown(self):
        self._pool.shutdown(wait=False)

    def _submit_next(self):
        if self._next_submit < len(self._video_paths):
            idx = self._next_submit
            self._futures[idx] = self._pool.submit(
                read_video_pyav, self._video_paths[idx], self._num_frames
            )
            self._next_submit += 1


# ---------------------------------------------------------------------------
# Video preprocessing
# ---------------------------------------------------------------------------


def preprocess_frames(
    frames: list[np.ndarray],
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    """Convert list of (H, W, 3) uint8 frames to (1, 3, F, H, W) float tensor in [-1, 1]."""
    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    t = t.to(device=device, dtype=torch.float32).div_(255.0)
    t = F.interpolate(
        t, size=(height, width),
        mode="bicubic", align_corners=False, antialias=True,
    ).clamp_(0.0, 1.0)
    t = t * 2.0 - 1.0
    # (F, 3, H, W) -> (1, 3, F, H, W)
    t = t.permute(1, 0, 2, 3).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_vae(model_name: str, device: str, enable_tiling: bool):
    from diffusers import AutoencoderKLLTXVideo

    logger.info(f"Loading VAE from {model_name} (float32 for stability)...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    if enable_tiling:
        vae.enable_tiling()
    return vae


def load_transformer(model_name: str, device: str, dtype: torch.dtype):
    from diffusers import LTXVideoTransformer3DModel

    logger.info(f"Loading transformer from {model_name} ({dtype})...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    transformer.eval()
    return transformer


def load_scheduler(model_name: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler",
    )


def compute_text_embeddings(
    model_name: str,
    text_prompt: str,
    max_seq_len: int,
    device: str,
    cast_dtype: torch.dtype,
    free_after: bool = True,
):
    """Compute text embeddings once. Optionally frees T5 afterwards."""
    from transformers import AutoTokenizer, T5EncoderModel

    logger.info("Loading T5 text encoder for embedding cache...")
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=cast_dtype,
    ).to(device)
    text_encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer",
    )

    text_inputs = tokenizer(
        text_prompt,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=cast_dtype)

    logger.info(
        f"Cached text embeddings: prompt='{text_prompt[:50]}', shape={prompt_embeds.shape}")

    if free_after:
        del text_encoder, tokenizer
        torch.cuda.empty_cache()
        logger.info("Freed T5 text encoder (~10GB VRAM saved)")

    return prompt_embeds, attention_mask


# ---------------------------------------------------------------------------
# Activation reshaping helpers
# ---------------------------------------------------------------------------


def reshape_vae_activation(activation: torch.Tensor) -> torch.Tensor:
    """Reshape VAE encoder activation to (N, C) — one row per spatiotemporal position."""
    if activation.ndim == 5:
        B, C, T, H, W = activation.shape
        return activation.permute(0, 2, 3, 4, 1).reshape(-1, C)
    elif activation.ndim == 4:
        B, C, H, W = activation.shape
        return activation.permute(0, 2, 3, 1).reshape(-1, C)
    elif activation.ndim == 3:
        return activation.squeeze(0)
    else:
        return activation.reshape(-1, activation.shape[-1])


def reshape_transformer_activation(activation: torch.Tensor) -> torch.Tensor:
    """Reshape transformer activation to (N, d) — one row per token."""
    if activation.ndim == 3:
        return activation.squeeze(0)   # (1, seq, d) -> (seq, d)
    elif activation.ndim > 3:
        return activation.reshape(-1, activation.shape[-1])
    return activation


# ---------------------------------------------------------------------------
# Single-video processing
# ---------------------------------------------------------------------------


@torch.no_grad()
def process_video_vae(
    video_tensor: torch.Tensor,
    vae: torch.nn.Module,
    hook_modules: list[str],
) -> dict[str, torch.Tensor]:
    """Run VAE encode with hooks and return {module_path: (N, d)} activations."""
    with multi_module_hooks(vae, hook_modules) as captured:
        vae.encode(video_tensor)

    results = {}
    for path, act in captured.items():
        results[path] = reshape_vae_activation(act).float()
    return results


@torch.no_grad()
def process_video_transformer(
    video_tensor: torch.Tensor,
    vae: torch.nn.Module,
    transformer: torch.nn.Module,
    hook_modules: list[str],
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_train_timesteps: int,
    shift: float,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> dict[str, torch.Tensor]:
    """VAE encode -> noise -> transformer forward -> hook capture.

    Returns {module_path: (N, d)} activations.
    """
    device = video_tensor.device
    transformer_dtype = next(transformer.parameters()).dtype

    # VAE encode
    latent_dist = vae.encode(video_tensor).latent_dist
    latents = latent_dist.sample()   # (1, 128, F_lat, H_lat, W_lat)

    # Normalize latents
    scaling_factor = vae.config.scaling_factor
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(
            vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(
            vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * scaling_factor

    # Pack to sequence: (1, 128, F, H, W) -> (1, F*H*W, 128)
    B, C, Fl, Hl, Wl = latents.shape
    latents = latents.permute(0, 2, 3, 4, 1).reshape(B, Fl * Hl * Wl, C)

    # Sample random timestep (flow matching)
    sigma = torch.rand(1, device=device, dtype=latents.dtype)
    if shift != 1.0:
        sigma = shift * sigma / (1 + (shift - 1) * sigma)

    noise = torch.randn_like(latents)
    noised_latents = (1.0 - sigma) * latents + sigma * noise
    noised_latents = noised_latents.to(transformer_dtype)

    timestep = (sigma * num_train_timesteps).to(transformer_dtype)

    with multi_module_hooks(transformer, hook_modules) as captured:
        transformer(
            hidden_states=noised_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            encoder_attention_mask=prompt_mask,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            return_dict=False,
        )

    results = {}
    for path, act in captured.items():
        results[path] = reshape_transformer_activation(act).float()
    return results


# ---------------------------------------------------------------------------
# Probing d_in for each hook module
# ---------------------------------------------------------------------------


@torch.no_grad()
def probe_d_in(
    config: GatherLTXConfig,
    vae: torch.nn.Module,
    transformer: torch.nn.Module | None,
    prompt_embeds: torch.Tensor | None,
    prompt_mask: torch.Tensor | None,
) -> dict[str, int]:
    """Run a minimal forward pass to detect d_in for each hook module."""
    # Build a small dummy video: 9 frames, 128x128 (minimum that works)
    dummy = torch.randn(
        1, 3, 9, 128, 128, device=config.device, dtype=torch.float32,
    ).clamp_(-1.0, 1.0)

    if config.hook_target == "vae_encoder":
        results = process_video_vae(dummy, vae, config.hook_modules)
    else:
        latent_num_frames = (9 - 1) // 8 + 1  # = 2
        latent_height = 128 // 32              # = 4
        latent_width = 128 // 32               # = 4
        results = process_video_transformer(
            dummy, vae, transformer, config.hook_modules,
            prompt_embeds, prompt_mask,
            config.num_train_timesteps, config.shift,
            latent_num_frames, latent_height, latent_width,
        )

    dims = {}
    for path, act in results.items():
        dims[path] = act.shape[-1]
        logger.info(f"Probed d_in={dims[path]} for hook '{path}'")
    return dims


# ---------------------------------------------------------------------------
# Core gathering loop
# ---------------------------------------------------------------------------


def gather_activations(config: GatherLTXConfig):
    """Process all videos and save per-module activations to disk."""
    video_paths = scan_video_directory(config.video_dir)
    if not video_paths:
        raise RuntimeError(f"No video files found in {config.video_dir}")
    logger.info(f"Found {len(video_paths)} videos")

    os.makedirs(config.output_dir, exist_ok=True)

    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in video_paths if not progress.is_done(p)]
    logger.info(
        f"Already processed: {len(video_paths) - len(remaining)}, remaining: {len(remaining)}")
    if not remaining:
        logger.info("All videos already processed.")
        return

    # Load models
    # For VAE encoder hooks, don't tile (tiling splits forward, hook only captures last tile)
    enable_tiling = config.hook_target != "vae_encoder"
    vae = load_vae(config.model_name, config.device, enable_tiling)

    transformer = None
    prompt_embeds = None
    prompt_mask = None

    if config.hook_target == "transformer":
        transformer = load_transformer(
            config.model_name, config.device, config.torch_dtype)
        transformer_dtype = next(transformer.parameters()).dtype
        prompt_embeds, prompt_mask = compute_text_embeddings(
            config.model_name, config.text_prompt, config.max_text_seq_len,
            config.device, transformer_dtype, free_after=True,
        )

        # Precompute latent dimensions
        latent_num_frames = (config.num_frames - 1) // 8 + 1
        latent_height = config.height // 32
        latent_width = config.width // 32
        logger.info(
            f"Latent dims: {latent_num_frames}x{latent_height}x{latent_width} "
            f"= {latent_num_frames * latent_height * latent_width} tokens/video")

    # Probe d_in per module
    d_in_map = probe_d_in(config, vae, transformer, prompt_embeds, prompt_mask)

    # Sanitize module paths for filesystem (replace dots with underscores)
    def module_dir_name(module_path: str) -> str:
        return module_path.replace(".", "_")

    shard_writers = {
        path: ShardWriter(os.path.join(
            config.output_dir, module_dir_name(path)))
        for path in config.hook_modules
    }
    prefetcher = FramePrefetcher(
        remaining, config.num_frames, config.prefetch_workers)
    videos_since_last_flush = 0

    for video_index, video_path in enumerate(tqdm(remaining, desc="Gathering LTX activations")):
        frames = prefetcher.get_frames(video_index)
        if frames is None:
            progress.mark_done(video_path)
            continue

        try:
            video_tensor = preprocess_frames(
                frames, config.height, config.width, config.device)
        except Exception as e:
            logger.debug(f"Failed to preprocess {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        # Extract activations
        try:
            if config.hook_target == "vae_encoder":
                activations = process_video_vae(
                    video_tensor, vae, config.hook_modules)
            else:
                activations = process_video_transformer(
                    video_tensor, vae, transformer, config.hook_modules,
                    prompt_embeds, prompt_mask,
                    config.num_train_timesteps, config.shift,
                    latent_num_frames, latent_height, latent_width,
                )
        except Exception as e:
            logger.warning(f"Forward pass failed for {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        for module_path in config.hook_modules:
            if module_path not in activations:
                continue
            act = activations[module_path]
            if config.max_norm_multiple > 0 and act.numel() > 0:
                act = remove_norm_outliers(act, config.max_norm_multiple)
            if act.numel() > 0:
                shard_writers[module_path].append(act)

        del activations, video_tensor
        torch.cuda.empty_cache()

        progress.mark_done(video_path)
        videos_since_last_flush += 1

        if videos_since_last_flush >= config.shard_size:
            for writer in shard_writers.values():
                writer.flush()
            progress.save()
            videos_since_last_flush = 0

    # Flush remaining
    for writer in shard_writers.values():
        writer.flush()
    progress.save()

    _write_module_metadata(config, shard_writers, d_in_map, len(video_paths))
    prefetcher.shutdown()

    logger.info(
        f"Done. Processed {len(remaining)} videos this run ({len(video_paths)} total).")


def _write_module_metadata(
    config: GatherLTXConfig,
    shard_writers: dict[str, ShardWriter],
    d_in_map: dict[str, int],
    num_videos: int,
):
    """Write metadata.json for each module directory."""
    for module_path, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name,
            "hook_target": config.hook_target,
            "hook_module": module_path,
            "d_model": d_in_map.get(module_path, -1),
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "text_prompt": config.text_prompt,
            "shift": config.shift,
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens,
            "num_shards": writer.shard_index,
            "num_videos": num_videos,
            "save_dtype": "float32",
        }
        metadata_path = os.path.join(writer.module_dir, METADATA_FILENAME)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"{module_path}: {writer.total_tokens} tokens in {writer.shard_index} shards "
            f"(d_in={d_in_map.get(module_path, '?')})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> GatherLTXConfig:
    parser = argparse.ArgumentParser(
        description="Gather video activations from LTX-Video")
    parser.add_argument("--model_name", type=str,
                        default="Lightricks/LTX-Video")
    parser.add_argument("--hook_target", type=str, default="transformer",
                        choices=["transformer", "vae_encoder"])
    parser.add_argument("--hook_modules", type=str, nargs="+", required=True,
                        help="Module paths to hook, e.g. transformer_blocks.14")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--text_prompt", type=str, default="")
    parser.add_argument("--max_text_seq_len", type=int, default=128)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--max_norm_multiple", type=int, default=10)
    parser.add_argument("--prefetch_workers", type=int, default=4)
    args = parser.parse_args()
    return GatherLTXConfig(**vars(args))


if __name__ == "__main__":
    gather_activations(parse_args())
