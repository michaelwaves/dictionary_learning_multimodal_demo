"""Gather VAE encoder latent means for SAE training.

Encodes videos through a VAE (LTX or Wan) and saves latent_dist.mean
reshaped to (N, C) -- one row per spatiotemporal position.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, asdict

import click
import torch
from tqdm import tqdm

from gather_utils import (
    METADATA_FILENAME,
    FramePrefetcher,
    ProgressTracker,
    ShardWriter,
    make_run_dir,
    preprocess_frames,
    remove_norm_outliers,
    reshape_vae_activation,
    save_run_config,
)
from video_utils import (
    read_all_frames,
    read_consecutive_frames,
    read_image_as_frame,
    scan_image_directory,
    scan_video_directory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


TEMPORAL_STRIDE = {"ltx": 8, "wan": 4}


def load_model_vae(model_name: str, vae_type: str, device: str):
    if vae_type == "wan":
        from wan_model import load_wan_vae
        return load_wan_vae(model_name, device)
    from gather_ltx_activations import load_vae
    return load_vae(model_name, device, enable_tiling=False)


@dataclass
class VaeLatentConfig:
    video_dir: str = ""
    image_dir: str = ""
    output_dir: str = ""
    model_name: str = "Lightricks/LTX-Video-0.9.5"
    vae_type: str = "ltx"
    device: str = "cuda:0"
    num_frames: int = 17
    height: int = 256
    width: int = 256
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4
    full_video: bool = False
    max_videos: int = 0

    @property
    def is_image_mode(self) -> bool:
        return bool(self.image_dir)

    @property
    def temporal_stride(self) -> int:
        return TEMPORAL_STRIDE[self.vae_type]


def read_consecutive_frames_random_start(video_path: str, num_frames: int) -> list:
    return read_consecutive_frames(video_path, num_frames, random.random() * 0.5)


def align_to_vae_frame_count(n: int, temporal_stride: int = 8) -> int:
    if n <= 1:
        return 1
    return ((n - 2) // temporal_stride + 1) * temporal_stride + 1


MIN_VAE_CHUNK_FRAMES = 9


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


@torch.no_grad()
def encode_latent_mean(video_tensor: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    latent_mean = vae.encode(video_tensor, return_dict=True).latent_dist.mean
    return reshape_vae_activation(latent_mean).float()


@torch.no_grad()
def encode_full_video_in_chunks(frames, vae, chunk_size, height, width, device, temporal_stride=8):
    chunks = chunk_frames_for_vae(frames, chunk_size, temporal_stride)
    activations = []
    for chunk in chunks:
        video_tensor = preprocess_frames(chunk, height, width, device)
        activations.append(encode_latent_mean(video_tensor, vae))
        del video_tensor
    return torch.cat(activations, dim=0)


def gather_vae_latents(config: VaeLatentConfig):
    if config.is_image_mode:
        input_paths = scan_image_directory(config.image_dir)
        if not input_paths:
            raise RuntimeError(f"No images found in {config.image_dir}")
    else:
        input_paths = scan_video_directory(config.video_dir)
        if not input_paths:
            raise RuntimeError(f"No videos found in {config.video_dir}")
    logger.info(f"Found {len(input_paths)} inputs")

    if not config.output_dir:
        config.output_dir = make_run_dir()
    os.makedirs(config.output_dir, exist_ok=True)
    save_run_config(config.output_dir, asdict(config))

    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in input_paths if not progress.is_done(p)]
    if config.max_videos > 0:
        remaining = remaining[:config.max_videos]
    logger.info(f"Remaining: {len(remaining)} / {len(input_paths)}")
    if not remaining:
        return

    vae = load_model_vae(config.model_name, config.vae_type, config.device)
    dummy_frames = 1 if config.is_image_mode else 9
    dummy = torch.randn(1, 3, dummy_frames, 128, 128, device=config.device).clamp_(-1, 1)
    d_in = encode_latent_mean(dummy, vae).shape[-1]
    del dummy

    writer = ShardWriter(config.output_dir)
    if config.is_image_mode:
        frame_reader = read_image_as_frame
    elif config.full_video:
        frame_reader = lambda path, _: read_all_frames(path)
    else:
        frame_reader = read_consecutive_frames_random_start

    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers, frame_reader)
    videos_since_flush = 0
    desc = "Encoding images" if config.is_image_mode else "Encoding VAE latents"

    for idx, path in enumerate(tqdm(remaining, desc=desc)):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(path)
            continue
        try:
            if config.full_video:
                activations = encode_full_video_in_chunks(
                    frames, vae, config.num_frames, config.height, config.width,
                    config.device, config.temporal_stride,
                )
            else:
                video_tensor = preprocess_frames(frames, config.height, config.width, config.device)
                activations = encode_latent_mean(video_tensor, vae)
                del video_tensor
        except Exception as e:
            logger.warning(f"Failed {path}: {e}")
            progress.mark_done(path)
            continue

        if config.max_norm_multiple > 0:
            activations = remove_norm_outliers(activations, config.max_norm_multiple)
        if activations.numel() > 0:
            writer.append(activations)
        del activations
        torch.cuda.empty_cache()
        progress.mark_done(path)
        videos_since_flush += 1
        if videos_since_flush >= config.shard_size:
            writer.flush()
            progress.save()
            videos_since_flush = 0

    writer.flush()
    progress.save()
    prefetcher.shutdown()

    metadata = {
        "model_name": config.model_name, "vae_type": config.vae_type,
        "hook_target": "vae_latent_mean",
        "d_model": d_in, "num_frames": config.num_frames,
        "height": config.height, "width": config.width,
        "max_norm_multiple": config.max_norm_multiple,
        "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
        "num_videos": len(input_paths), "save_dtype": "float32",
    }
    with open(os.path.join(config.output_dir, METADATA_FILENAME), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Done. {writer.total_tokens} tokens, {writer.shard_index} shards. Output: {config.output_dir}")


@click.command()
@click.option("--video-dir", default="")
@click.option("--image-dir", default="")
@click.option("--output-dir", default="")
@click.option("--model-name", default="Lightricks/LTX-Video-0.9.5")
@click.option("--vae-type", default="ltx", type=click.Choice(["ltx", "wan"]))
@click.option("--device", default="cuda:0")
@click.option("--num-frames", default=17, type=int)
@click.option("--height", default=256, type=int)
@click.option("--width", default=256, type=int)
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
@click.option("--full-video", is_flag=True)
@click.option("--max-videos", default=0, type=int)
def main(**kwargs):
    if not kwargs["video_dir"] and not kwargs["image_dir"]:
        raise click.UsageError("Provide either --video-dir or --image-dir")
    gather_vae_latents(VaeLatentConfig(**kwargs))


if __name__ == "__main__":
    main()
