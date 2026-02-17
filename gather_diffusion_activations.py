"""Gather activations from diffusion video models (LTX, Wan).

Usage:
    python gather_diffusion_activations.py \
        --model-type wan --hook-target vae_latent_mean \
        --video-dir /path/to/videos

    python gather_diffusion_activations.py \
        --model-type ltx --hook-target transformer \
        --hook-modules transformer_blocks.14 --video-dir /path/to/videos
"""

import json
import logging
import os
import random

import click
import torch
from tqdm import tqdm

from diffusion_config import (
    DiffusionGatherConfig, load_diffusion_models, probe_d_in, run_forward_pass,
)
from gather_utils import (
    METADATA_FILENAME, FramePrefetcher, ProgressTracker, ShardWriter,
    chunk_frames_for_vae, make_run_dir, preprocess_frames,
    remove_norm_outliers, reshape_vae_activation, save_run_config,
)
from video_utils import (
    read_all_frames, read_consecutive_frames, read_image_as_frame,
    scan_image_directory, scan_video_directory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _module_dir_name(module_path: str) -> str:
    return module_path.replace(".", "_")


def _read_consecutive_random_start(video_path: str, num_frames: int) -> list:
    return read_consecutive_frames(video_path, num_frames, random.random() * 0.5)


@torch.no_grad()
def _encode_full_video_in_chunks(frames, vae, chunk_size, height, width, device, temporal_stride):
    chunks = chunk_frames_for_vae(frames, chunk_size, temporal_stride)
    activations = []
    for chunk in chunks:
        video_tensor = preprocess_frames(chunk, height, width, device)
        latent_mean = vae.encode(video_tensor, return_dict=True).latent_dist.mean
        activations.append(reshape_vae_activation(latent_mean).float())
        del video_tensor
    return torch.cat(activations, dim=0)


def gather_diffusion_activations(config: DiffusionGatherConfig):
    is_image_mode = bool(config.image_dir)
    input_paths = scan_image_directory(config.image_dir) if is_image_mode else scan_video_directory(config.video_dir)
    if not input_paths:
        raise RuntimeError("No input files found")
    logger.info(f"Found {len(input_paths)} inputs")

    if not config.output_dir:
        config.output_dir = make_run_dir()
    os.makedirs(config.output_dir, exist_ok=True)
    save_run_config(config.output_dir, vars(config))

    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in input_paths if not progress.is_done(p)]
    if config.max_videos > 0:
        remaining = remaining[:config.max_videos]
    logger.info(f"Remaining: {len(remaining)} / {len(input_paths)}")
    if not remaining:
        return

    vae, transformer, prompt_embeds, prompt_mask = load_diffusion_models(config)
    d_in_map = probe_d_in(config, vae, transformer, prompt_embeds, prompt_mask)

    targets = config.hook_modules or ["vae_latent_mean"]
    shard_writers = {t: ShardWriter(os.path.join(config.output_dir, _module_dir_name(t))) for t in targets}

    if is_image_mode:
        frame_reader = read_image_as_frame
    elif config.full_video:
        frame_reader = lambda path, _: read_all_frames(path)
    else:
        frame_reader = _read_consecutive_random_start

    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers, frame_reader)
    videos_since_flush = 0

    for idx, path in enumerate(tqdm(remaining, desc=f"Gathering {config.hook_target}")):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(path)
            continue
        try:
            if config.hook_target == "vae_latent_mean" and config.full_video:
                activations = {"vae_latent_mean": _encode_full_video_in_chunks(
                    frames, vae, config.num_frames, config.height, config.width,
                    config.device, config.temporal_stride,
                )}
            else:
                video_tensor = preprocess_frames(frames, config.height, config.width, config.device)
                activations = run_forward_pass(config, video_tensor, vae, transformer, prompt_embeds, prompt_mask)
                del video_tensor
        except Exception as e:
            logger.warning(f"Failed {path}: {e}")
            progress.mark_done(path)
            continue

        for target in targets:
            act = activations.get(target)
            if act is not None and act.numel() > 0:
                if config.max_norm_multiple > 0:
                    act = remove_norm_outliers(act, config.max_norm_multiple)
                if act.numel() > 0:
                    shard_writers[target].append(act)

        del activations
        torch.cuda.empty_cache()
        progress.mark_done(path)
        videos_since_flush += 1
        if videos_since_flush >= config.shard_size:
            for w in shard_writers.values():
                w.flush()
            progress.save()
            videos_since_flush = 0

    for w in shard_writers.values():
        w.flush()
    progress.save()
    prefetcher.shutdown()

    for target, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name, "model_type": config.model_type,
            "hook_target": config.hook_target, "hook_module": target,
            "d_model": d_in_map.get(target, -1),
            "num_frames": config.num_frames, "height": config.height, "width": config.width,
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
            "num_videos": len(input_paths), "save_dtype": "float32",
        }
        with open(os.path.join(writer.module_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"{target}: {writer.total_tokens} tokens, {writer.shard_index} shards")
    logger.info(f"Done. Output: {config.output_dir}")


@click.command()
@click.option("--model-type", required=True, type=click.Choice(["ltx", "wan"]))
@click.option("--hook-target", required=True, type=click.Choice(["transformer", "vae_encoder", "vae_latent_mean"]))
@click.option("--hook-modules", multiple=True)
@click.option("--model-name", default="")
@click.option("--video-dir", default="")
@click.option("--image-dir", default="")
@click.option("--output-dir", default="")
@click.option("--device", default="cuda:0")
@click.option("--dtype", default="bfloat16", type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option("--num-frames", default=17, type=int)
@click.option("--height", default=256, type=int)
@click.option("--width", default=256, type=int)
@click.option("--text-prompt", default="")
@click.option("--max-text-seq-len", default=128, type=int)
@click.option("--num-train-timesteps", default=1000, type=int)
@click.option("--shift", default=1.0, type=float)
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
@click.option("--full-video", is_flag=True)
@click.option("--max-videos", default=0, type=int)
def main(**kwargs):
    if not kwargs["video_dir"] and not kwargs["image_dir"]:
        raise click.UsageError("Provide either --video-dir or --image-dir")
    kwargs["hook_modules"] = list(kwargs["hook_modules"]) or None
    gather_diffusion_activations(DiffusionGatherConfig(**kwargs))


if __name__ == "__main__":
    main()
