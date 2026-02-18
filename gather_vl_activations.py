"""Gather video activations from vision models (Qwen3-VL, VJEPA2).

Usage:
    python gather_vl_activations.py \
        --model-type qwen --video-dir /path/to/videos --layers 12 18 24

    python gather_vl_activations.py \
        --model-type vjepa --video-dir /path/to/videos --layers 8 16 23
"""

import json
import logging
import os
from dataclasses import asdict, dataclass

import click
import torch
from tqdm import tqdm

from gather_utils import (
    METADATA_FILENAME, FramePrefetcher, ProgressTracker, ShardWriter,
    _DTYPE_MAP, make_run_dir, remove_norm_outliers, save_run_config,
)
from video_utils import scan_video_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "vjepa": "facebook/vjepa2-vitl-fpc64-256",
}
DEFAULT_NUM_FRAMES = {"qwen": 16, "vjepa": 64}


@dataclass
class VisionGatherConfig:
    model_type: str = "qwen"
    model_name: str = ""
    video_dir: str = ""
    output_dir: str = ""
    layers: list[int] = None
    num_frames: int = 0
    prompt: str = "Describe what happens in this video."
    device: str = "cuda:0"
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4
    dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.model_name:
            self.model_name = DEFAULT_MODEL_NAMES[self.model_type]
        if self.num_frames == 0:
            self.num_frames = DEFAULT_NUM_FRAMES[self.model_type]

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)


def _load_model_and_get_d_model(config: VisionGatherConfig):
    if config.model_type == "vjepa":
        from vjepa_model import load_vjepa_model_and_processor
        model, processor = load_vjepa_model_and_processor(
            config.model_name, config.torch_dtype, config.device,
        )
        return model, processor, model.config.hidden_size
    from qwen_model import load_qwen_model_and_processor
    model, processor = load_qwen_model_and_processor(
        config.model_name, config.torch_dtype, config.device, config.num_frames,
    )
    d_model = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    return model, processor, d_model


def _forward_pass(config: VisionGatherConfig, model, processor, frames):
    if config.model_type == "vjepa":
        from vjepa_model import run_vjepa_forward
        return run_vjepa_forward(model, processor, frames, config.layers, config.device)
    from qwen_model import run_qwen_forward
    return run_qwen_forward(model, processor, frames, config.prompt, config.layers)


def gather_vl_activations(config: VisionGatherConfig):
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
    logger.info(f"Remaining: {len(remaining)} / {len(video_paths)}")
    if not remaining:
        return

    model, processor, d_model = _load_model_and_get_d_model(config)
    shard_writers = {
        layer: ShardWriter(os.path.join(config.output_dir, f"layer_{layer}"))
        for layer in config.layers
    }
    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers)
    videos_since_flush = 0

    for idx, video_path in enumerate(tqdm(remaining, desc=f"Gathering {config.model_type}")):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(video_path)
            continue
        try:
            activations = _forward_pass(config, model, processor, frames)
        except Exception as e:
            logger.warning(f"Failed {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        for layer_idx in config.layers:
            act = activations.get(layer_idx)
            if act is not None and act.numel() > 0:
                if config.max_norm_multiple > 0:
                    act = remove_norm_outliers(act, config.max_norm_multiple)
                if act.numel() > 0:
                    shard_writers[layer_idx].append(act)

        del activations
        torch.cuda.empty_cache()
        progress.mark_done(video_path)
        videos_since_flush += 1
        if videos_since_flush >= config.shard_size:
            for writer in shard_writers.values():
                writer.flush()
            progress.save()
            videos_since_flush = 0

    for writer in shard_writers.values():
        writer.flush()
    progress.save()
    prefetcher.shutdown()

    for layer_idx, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name, "model_type": config.model_type,
            "layer": layer_idx, "d_model": d_model, "num_frames": config.num_frames,
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
            "num_videos": len(video_paths), "save_dtype": "float32",
        }
        with open(os.path.join(writer.module_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Layer {layer_idx}: {writer.total_tokens} tokens, {writer.shard_index} shards")
    logger.info(f"Done. Output: {config.output_dir}")


@click.command()
@click.option("--model-type", default="qwen", type=click.Choice(["qwen", "vjepa"]))
@click.option("--model-name", default="")
@click.option("--video-dir", required=True)
@click.option("--output-dir", default="")
@click.option("--layers", required=True, type=str, help="Comma-separated layer indices, e.g. 8,16,23")
@click.option("--num-frames", default=0, type=int)
@click.option("--prompt", default="Describe what happens in this video.")
@click.option("--device", default="cuda:0")
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
@click.option("--dtype", default="bfloat16", type=click.Choice(["float16", "bfloat16", "float32"]))
def main(**kwargs):
    kwargs["layers"] = [int(x) for x in kwargs["layers"].split(",")]
    gather_vl_activations(VisionGatherConfig(**kwargs))


if __name__ == "__main__":
    main()
