"""Gather VAE latent means from LTX-Video for SAE training.

Encodes videos through the LTX VAE and saves latent_dist.mean
reshaped to (N, C) — one row per spatiotemporal position.
"""

import json
import logging
import os
import random
from dataclasses import dataclass

import click
import torch
from tqdm import tqdm

from gather_ltx_activations import (
    METADATA_FILENAME,
    FramePrefetcher,
    ProgressTracker,
    ShardWriter,
    load_vae,
    preprocess_frames,
    remove_norm_outliers,
    reshape_vae_activation,
)
from video_utils import read_all_frames, read_consecutive_frames, scan_video_directory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VaeLatentConfig:
    video_dir: str
    output_dir: str
    model_name: str = "Lightricks/LTX-Video-0.9.5"
    device: str = "cuda:0"
    num_frames: int = 17
    height: int = 256
    width: int = 256
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4
    full_video: bool = False
    max_videos: int = 0


def read_consecutive_frames_random_start(
    video_path: str, num_frames: int,
) -> list:
    start_fraction = random.random() * 0.5
    return read_consecutive_frames(video_path, num_frames, start_fraction)


def align_to_vae_frame_count(n: int) -> int:
    """Round up to nearest valid VAE frame count where (result - 1) % 8 == 0."""
    if n <= 1:
        return 1
    return ((n - 2) // 8 + 1) * 8 + 1


MIN_VAE_CHUNK_FRAMES = 9


def chunk_frames_for_vae(
    frames: list, chunk_size: int,
) -> list[list]:
    """Split a full video's frames into VAE-aligned non-overlapping chunks."""
    chunks = []
    for start in range(0, len(frames), chunk_size):
        chunk = frames[start:start + chunk_size]
        if len(chunk) < MIN_VAE_CHUNK_FRAMES:
            break
        aligned_size = align_to_vae_frame_count(len(chunk))
        while len(chunk) < aligned_size:
            chunk.append(chunk[-1])
        chunks.append(chunk)
    return chunks


@torch.no_grad()
def encode_latent_mean(video_tensor: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    latent_mean = vae.encode(video_tensor, return_dict=True).latent_dist.mean
    return reshape_vae_activation(latent_mean).float()


@torch.no_grad()
def encode_full_video_in_chunks(
    frames: list,
    vae: torch.nn.Module,
    chunk_size: int,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    """Encode all frames by chunking into VAE-aligned segments."""
    chunks = chunk_frames_for_vae(frames, chunk_size)
    activations = []
    for chunk in chunks:
        video_tensor = preprocess_frames(chunk, height, width, device)
        activations.append(encode_latent_mean(video_tensor, vae))
        del video_tensor
    return torch.cat(activations, dim=0)


def gather_vae_latents(config: VaeLatentConfig):
    video_paths = scan_video_directory(config.video_dir)
    if not video_paths:
        raise RuntimeError(f"No videos found in {config.video_dir}")
    logger.info(f"Found {len(video_paths)} videos")

    os.makedirs(config.output_dir, exist_ok=True)
    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in video_paths if not progress.is_done(p)]
    if config.max_videos > 0:
        remaining = remaining[:config.max_videos]
    logger.info(f"Remaining: {len(remaining)} / {len(video_paths)}")
    if not remaining:
        return

    vae = load_vae(config.model_name, config.device, enable_tiling=False)

    dummy = torch.randn(1, 3, 9, 128, 128, device=config.device).clamp_(-1, 1)
    d_in = encode_latent_mean(dummy, vae).shape[-1]
    logger.info(f"Latent d_in={d_in}")
    del dummy

    writer = ShardWriter(config.output_dir)

    if config.full_video:
        frame_reader = lambda path, _num_frames: read_all_frames(path)
    else:
        frame_reader = read_consecutive_frames_random_start

    prefetcher = FramePrefetcher(
        remaining, config.num_frames, config.prefetch_workers,
        frame_reader=frame_reader)
    videos_since_flush = 0

    desc = "Encoding full videos" if config.full_video else "Encoding VAE latents"
    for idx, video_path in enumerate(tqdm(remaining, desc=desc)):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(video_path)
            continue

        try:
            if config.full_video:
                activations = encode_full_video_in_chunks(
                    frames, vae, config.num_frames,
                    config.height, config.width, config.device,
                )
            else:
                video_tensor = preprocess_frames(
                    frames, config.height, config.width, config.device)
                activations = encode_latent_mean(video_tensor, vae)
                del video_tensor
        except Exception as e:
            logger.warning(f"Failed {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        if config.max_norm_multiple > 0:
            activations = remove_norm_outliers(
                activations, config.max_norm_multiple)
        if activations.numel() > 0:
            writer.append(activations)

        del activations
        torch.cuda.empty_cache()
        progress.mark_done(video_path)
        videos_since_flush += 1

        if videos_since_flush >= config.shard_size:
            writer.flush()
            progress.save()
            videos_since_flush = 0

    writer.flush()
    progress.save()
    prefetcher.shutdown()

    metadata = {
        "model_name": config.model_name,
        "hook_target": "vae_latent_mean",
        "d_model": d_in,
        "num_frames": config.num_frames,
        "height": config.height,
        "width": config.width,
        "max_norm_multiple": config.max_norm_multiple,
        "total_tokens": writer.total_tokens,
        "num_shards": writer.shard_index,
        "num_videos": len(video_paths),
        "save_dtype": "float32",
    }
    with open(os.path.join(config.output_dir, METADATA_FILENAME), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Done. {writer.total_tokens} tokens, {writer.shard_index} shards (d_in={d_in})")


@click.command()
@click.option("--video-dir", required=True)
@click.option("--output-dir", required=True)
@click.option("--model-name", default="Lightricks/LTX-Video-0.9.5")
@click.option("--device", default="cuda:0")
@click.option("--num-frames", default=17, type=int)
@click.option("--height", default=256, type=int)
@click.option("--width", default=256, type=int)
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
@click.option("--full-video", is_flag=True, help="Encode entire videos in chunks instead of sampling frames")
@click.option("--max-videos", default=0, type=int, help="Max videos to process (0 = all)")
def main(**kwargs):
    gather_vae_latents(VaeLatentConfig(**kwargs))


if __name__ == "__main__":
    main()
