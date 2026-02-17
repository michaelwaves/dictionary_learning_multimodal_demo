"""Gather video activations from vision-language models (Qwen3-VL).

Hooks on language model residual stream layers, filters to visual-only
token positions, and saves sharded activations for SAE training.

Usage:
    python gather_vl_activations.py \
        --video-dir /path/to/videos \
        --layers 12 18 24 \
        --device cuda:0
"""

import json
import logging
import os
from dataclasses import dataclass, asdict

import click
import torch
from tqdm import tqdm

from gather_utils import (
    METADATA_FILENAME, FramePrefetcher, ProgressTracker, ShardWriter,
    _DTYPE_MAP, make_run_dir, remove_norm_outliers, save_run_config,
)
from qwen_model import (
    build_chat_text, load_qwen_model_and_processor, move_inputs_to_device,
    multi_layer_hooks, prepare_video_inputs, select_visual_tokens,
)
from video_utils import scan_video_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VLGatherConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    video_dir: str = ""
    output_dir: str = ""
    layers: list[int] = None
    num_frames: int = 16
    prompt: str = "Describe what happens in this video."
    device: str = "cuda:0"
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4
    dtype: str = "bfloat16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)


def gather_vl_activations(config: VLGatherConfig):
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

    model, processor = load_qwen_model_and_processor(
        config.model_name, config.torch_dtype, config.device, config.num_frames,
    )
    model_device = next(model.parameters()).device
    video_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    chat_text = build_chat_text(processor, config.prompt)
    d_model = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size

    shard_writers = {
        layer: ShardWriter(os.path.join(config.output_dir, f"layer_{layer}"))
        for layer in config.layers
    }
    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers)
    videos_since_flush = 0

    for idx, video_path in enumerate(tqdm(remaining, desc="Gathering VL activations")):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(video_path)
            continue
        inputs = prepare_video_inputs(processor, frames, chat_text)
        if inputs is None:
            progress.mark_done(video_path)
            continue
        inputs = move_inputs_to_device(inputs, model_device)
        input_ids = inputs["input_ids"]

        with multi_layer_hooks(model, config.layers) as captured:
            try:
                with torch.no_grad():
                    model(**inputs)
            except Exception as e:
                logger.warning(f"Forward pass failed for {video_path}: {e}")
                progress.mark_done(video_path)
                continue

        for layer_idx in config.layers:
            if layer_idx not in captured:
                continue
            visual_acts = select_visual_tokens(captured[layer_idx], input_ids, video_pad_token_id)
            if config.max_norm_multiple > 0 and visual_acts.numel() > 0:
                visual_acts = remove_norm_outliers(visual_acts, config.max_norm_multiple)
            if visual_acts.numel() > 0:
                shard_writers[layer_idx].append(visual_acts)

        del captured, inputs
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
            "model_name": config.model_name, "layer": layer_idx,
            "d_model": d_model, "num_frames": config.num_frames,
            "prompt": config.prompt, "filter_mode": "visual_only",
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
            "num_videos": len(video_paths), "save_dtype": "float32",
        }
        with open(os.path.join(writer.module_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Layer {layer_idx}: {writer.total_tokens} tokens, {writer.shard_index} shards")
    logger.info(f"Done. Output: {config.output_dir}")


@click.command()
@click.option("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
@click.option("--video-dir", required=True)
@click.option("--output-dir", default="")
@click.option("--layers", required=True, type=int, multiple=True)
@click.option("--num-frames", default=16, type=int)
@click.option("--prompt", default="Describe what happens in this video.")
@click.option("--device", default="cuda:0")
@click.option("--shard-size", default=50, type=int)
@click.option("--max-norm-multiple", default=10, type=int)
@click.option("--prefetch-workers", default=4, type=int)
@click.option("--dtype", default="bfloat16", type=click.Choice(["float16", "bfloat16", "float32"]))
def main(**kwargs):
    kwargs["layers"] = list(kwargs["layers"])
    gather_vl_activations(VLGatherConfig(**kwargs))


if __name__ == "__main__":
    main()
