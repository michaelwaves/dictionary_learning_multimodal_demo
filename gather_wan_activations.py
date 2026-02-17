"""Stage 1: Gather video activations from Wan2.2 and save to disk.

Usage:
    python gather_wan_activations.py \
        --video_dir /path/to/videos \
        --hook_modules blocks.15 blocks.20 \
        --num_frames 17 --device cuda:0

Hookable transformer paths: blocks.{0-29} (d=3072),
  blocks.{i}.attn1/attn2/ffn
"""

import json
import logging
import os
from dataclasses import dataclass, asdict

import click
import torch
from tqdm import tqdm

from gather_utils import (
    METADATA_FILENAME,
    FramePrefetcher,
    ProgressTracker,
    ShardWriter,
    _DTYPE_MAP,
    make_run_dir,
    preprocess_frames,
    remove_norm_outliers,
    save_run_config,
)
from video_utils import scan_video_directory
from wan_model import (
    compute_wan_text_embeddings,
    load_wan_transformer,
    load_wan_vae,
    probe_wan_d_in,
    process_wan_transformer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GatherWanConfig:
    model_name: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    hook_modules: list[str] = None
    video_dir: str = ""
    output_dir: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    num_frames: int = 17
    height: int = 256
    width: int = 256
    text_prompt: str = ""
    max_text_seq_len: int = 256
    num_train_timesteps: int = 1000
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)

    def __post_init__(self):
        if (self.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (n-1)%4==0 for Wan VAE, got {self.num_frames}. "
                f"Valid: 5, 9, 13, 17, 21, 25...")
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(f"height/width must be divisible by 16, got {self.height}x{self.width}")


def _module_dir_name(module_path: str) -> str:
    return module_path.replace(".", "_")


def gather_wan_activations(config: GatherWanConfig):
    video_paths = scan_video_directory(config.video_dir)
    if not video_paths:
        raise RuntimeError(f"No video files found in {config.video_dir}")
    logger.info(f"Found {len(video_paths)} videos")

    if not config.output_dir:
        config.output_dir = make_run_dir()
    os.makedirs(config.output_dir, exist_ok=True)
    save_run_config(config.output_dir, asdict(config))

    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in video_paths if not progress.is_done(p)]
    logger.info(f"Already processed: {len(video_paths) - len(remaining)}, remaining: {len(remaining)}")
    if not remaining:
        return

    vae = load_wan_vae(config.model_name, config.device)
    transformer = load_wan_transformer(config.model_name, config.device, config.torch_dtype)
    prompt_embeds, prompt_mask = compute_wan_text_embeddings(
        config.model_name, config.text_prompt, config.max_text_seq_len,
        config.device, next(transformer.parameters()).dtype, free_after=True,
    )

    d_in_map = probe_wan_d_in(
        config.device, vae, transformer, config.hook_modules,
        prompt_embeds, prompt_mask, config.num_train_timesteps,
    )

    shard_writers = {
        p: ShardWriter(os.path.join(config.output_dir, _module_dir_name(p)))
        for p in config.hook_modules
    }
    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers)
    videos_since_flush = 0

    for idx, video_path in enumerate(tqdm(remaining, desc="Gathering Wan activations")):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(video_path)
            continue
        try:
            video_tensor = preprocess_frames(frames, config.height, config.width, config.device)
        except Exception:
            progress.mark_done(video_path)
            continue
        try:
            activations = process_wan_transformer(
                video_tensor, vae, transformer, config.hook_modules,
                prompt_embeds, prompt_mask, config.num_train_timesteps,
            )
        except Exception as e:
            logger.warning(f"Forward pass failed for {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        for module_path in config.hook_modules:
            act = activations.get(module_path)
            if act is not None and act.numel() > 0:
                if config.max_norm_multiple > 0:
                    act = remove_norm_outliers(act, config.max_norm_multiple)
                if act.numel() > 0:
                    shard_writers[module_path].append(act)

        del activations, video_tensor
        torch.cuda.empty_cache()
        progress.mark_done(video_path)
        videos_since_flush += 1
        if videos_since_flush >= config.shard_size:
            for w in shard_writers.values():
                w.flush()
            progress.save()
            videos_since_flush = 0

    for w in shard_writers.values():
        w.flush()
    progress.save()

    for module_path, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name, "hook_target": "transformer",
            "hook_module": module_path, "d_model": d_in_map.get(module_path, -1),
            "num_frames": config.num_frames, "height": config.height, "width": config.width,
            "text_prompt": config.text_prompt,
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
            "num_videos": len(video_paths), "save_dtype": "float32",
        }
        with open(os.path.join(writer.module_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"{module_path}: {writer.total_tokens} tokens, {writer.shard_index} shards")

    prefetcher.shutdown()
    logger.info(f"Done. Output: {config.output_dir}")


@click.command()
@click.option("--model-name", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
@click.option("--hook-modules", multiple=True, required=True)
@click.option("--video-dir", required=True)
@click.option("--output-dir", default="")
@click.option("--device", default="cuda:0")
@click.option("--dtype", default="bfloat16", type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option("--num-frames", default=17, type=int)
@click.option("--height", default=256, type=int)
@click.option("--width", default=256, type=int)
@click.option("--text-prompt", default="")
@click.option("--max-text-seq-len", default=256, type=int)
@click.option("--num-train-timesteps", default=1000, type=int)
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
def main(**kwargs):
    kwargs["hook_modules"] = list(kwargs["hook_modules"])
    gather_wan_activations(GatherWanConfig(**kwargs))


if __name__ == "__main__":
    main()
