"""
Stage 1: Gather video activations from Qwen3-VL and save to disk.

Processes videos through Qwen3-VL-8B-Instruct, captures residual stream
activations at multiple LM layers via forward hooks, filters to visual-only
token positions, and saves sharded .pt files for SAE training.

Usage:
    python gather_video_activations.py \
        --model_name "Qwen/Qwen3-VL-8B-Instruct" \
        --video_dir /path/to/videos \
        --output_dir ./activations \
        --layers 12 18 24 \
        --num_frames 16 \
        --device cuda:0
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
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from video_utils import read_video_pyav, scan_video_directory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARD_FILENAME_TEMPLATE = "shard_{:04d}.pt"
METADATA_FILENAME = "metadata.json"
PROGRESS_FILENAME = "progress.json"
FRAME_LOAD_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GatherConfig:
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
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]


# ---------------------------------------------------------------------------
# Pure activation filtering
# ---------------------------------------------------------------------------


def select_visual_tokens(
    activations: torch.Tensor,
    input_ids: torch.Tensor,
    video_pad_token_id: int,
) -> torch.Tensor:
    """Select only the visual token positions from a sequence of activations.

    Args:
        activations: [batch, seq_len, d_model]
        input_ids: [batch, seq_len]
        video_pad_token_id: token ID marking visual positions

    Returns:
        [N, d_model] where N is the number of visual tokens.
    """
    visual_mask = input_ids == video_pad_token_id
    return activations[visual_mask]


def remove_norm_outliers(
    activations: torch.Tensor,
    threshold_multiple: int,
) -> torch.Tensor:
    """Remove activations whose norm exceeds median_norm * threshold_multiple.

    Qwen models produce sporadic high-norm activation sinks that hurt SAE training.
    """
    if activations.numel() == 0:
        return activations
    norms = activations.norm(dim=-1)
    median_norm = norms.median()
    within_threshold = norms <= median_norm * threshold_multiple
    return activations[within_threshold]


# ---------------------------------------------------------------------------
# Multi-layer hook context manager
# ---------------------------------------------------------------------------


@contextmanager
def multi_layer_hooks(model: torch.nn.Module, layer_indices: list[int]):
    """Context manager that registers forward hooks on multiple LM layers.

    Yields a dict that will be populated with {layer_index: activation_tensor}
    after a forward pass through the model.

    Usage:
        with multi_layer_hooks(model, [12, 18, 24]) as captured:
            model(**inputs)
        # captured[12] is now the layer-12 activation tensor
    """
    captured = {}
    handles = []

    for layer_idx in layer_indices:
        submodule = model.model.language_model.layers[layer_idx]

        def _make_hook(idx):
            def hook_fn(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[idx] = tensor.detach()
            return hook_fn

        handle = submodule.register_forward_hook(_make_hook(layer_idx))
        handles.append(handle)

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Accumulates activation tensors and writes them to numbered shard files."""

    def __init__(self, layer_dir: str):
        self.layer_dir = layer_dir
        os.makedirs(layer_dir, exist_ok=True)
        self.shard_index = self._count_existing_shards()
        self.buffer: list[torch.Tensor] = []
        self.total_tokens = 0

    def append(self, activations: torch.Tensor):
        """Add a chunk of activations to the current shard buffer."""
        self.buffer.append(activations.cpu().to(torch.float32))

    def flush(self):
        """Write buffered activations to a numbered shard file."""
        if not self.buffer:
            return
        shard_data = torch.cat(self.buffer, dim=0)
        shard_path = os.path.join(
            self.layer_dir, SHARD_FILENAME_TEMPLATE.format(self.shard_index)
        )
        torch.save(shard_data, shard_path)
        self.total_tokens += shard_data.shape[0]
        logger.info(
            f"{self.layer_dir}: shard {self.shard_index} — {shard_data.shape[0]} tokens"
        )
        self.shard_index += 1
        self.buffer = []

    def _count_existing_shards(self) -> int:
        return sum(
            1 for f in os.listdir(self.layer_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ) if os.path.exists(self.layer_dir) else 0


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


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

        # Pre-submit initial batch
        for _ in range(min(max_workers, len(video_paths))):
            self._submit_next()

    def get_frames(self, video_index: int) -> list[np.ndarray] | None:
        """Retrieve prefetched frames for the given video index."""
        if video_index in self._futures:
            future = self._futures.pop(video_index)
            self._submit_next()
            try:
                return future.result(timeout=FRAME_LOAD_TIMEOUT_SECONDS)
            except Exception as e:
                logger.debug(
                    f"Prefetch failed for {self._video_paths[video_index]}: {e}")
                return None

        # Fallback: load synchronously
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
# Model + processor loading
# ---------------------------------------------------------------------------


def load_model_and_processor(config: GatherConfig):
    """Load Qwen3-VL model and processor with fixed frame count."""
    logger.info(f"Loading {config.model_name}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=config.torch_dtype,
        low_cpu_mem_usage=True,
        device_map=config.device,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(config.model_name)
    processor.video_processor.min_frames = config.num_frames
    processor.video_processor.max_frames = config.num_frames

    return model, processor


def build_chat_text(processor: AutoProcessor, prompt: str) -> str:
    """Build the chat-templated prompt with a video placeholder."""
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def prepare_video_inputs(
    processor: AutoProcessor,
    frames: list[np.ndarray],
    chat_text: str,
) -> dict | None:
    """Run the VL processor on video frames. Returns model-ready input dict."""
    try:
        return processor(
            text=[chat_text], videos=[frames], return_tensors="pt", padding=True
        )
    except Exception as e:
        logger.debug(f"Processor failed: {e}")
        return None


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


# ---------------------------------------------------------------------------
# Core gathering loop
# ---------------------------------------------------------------------------


def gather_activations(config: GatherConfig):
    """Process all videos and save per-layer activations to disk."""
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

    model, processor = load_model_and_processor(config)
    model_device = next(model.parameters()).device
    video_pad_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|video_pad|>")
    chat_text = build_chat_text(processor, config.prompt)
    d_model = getattr(model.config, "hidden_size",
                      None) or model.config.text_config.hidden_size

    shard_writers = {
        layer: ShardWriter(os.path.join(config.output_dir, f"layer_{layer}"))
        for layer in config.layers
    }
    prefetcher = FramePrefetcher(
        remaining, config.num_frames, config.prefetch_workers)
    videos_since_last_flush = 0

    for video_index, video_path in enumerate(tqdm(remaining, desc="Gathering activations")):
        frames = prefetcher.get_frames(video_index)
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

            visual_activations = select_visual_tokens(
                captured[layer_idx], input_ids, video_pad_token_id
            )
            if config.max_norm_multiple > 0 and visual_activations.numel() > 0:
                visual_activations = remove_norm_outliers(
                    visual_activations, config.max_norm_multiple
                )
            if visual_activations.numel() > 0:
                shard_writers[layer_idx].append(visual_activations)

        del captured, inputs
        torch.cuda.empty_cache()

        progress.mark_done(video_path)
        videos_since_last_flush += 1

        if videos_since_last_flush >= config.shard_size:
            for writer in shard_writers.values():
                writer.flush()
            progress.save()
            videos_since_last_flush = 0

    # Flush remaining buffered activations
    for writer in shard_writers.values():
        writer.flush()
    progress.save()

    _write_layer_metadata(config, shard_writers, d_model, len(video_paths))
    prefetcher.shutdown()

    logger.info(
        f"Done. Processed {len(remaining)} videos this run ({len(video_paths)} total).")


def _write_layer_metadata(
    config: GatherConfig,
    shard_writers: dict[int, ShardWriter],
    d_model: int,
    num_videos: int,
):
    """Write metadata.json for each layer directory."""
    for layer_idx, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name,
            "layer": layer_idx,
            "d_model": d_model,
            "num_frames": config.num_frames,
            "prompt": config.prompt,
            "filter_mode": "visual_only",
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens,
            "num_shards": writer.shard_index,
            "num_videos": num_videos,
            "save_dtype": "float32",
        }
        metadata_path = os.path.join(writer.layer_dir, METADATA_FILENAME)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Layer {layer_idx}: {writer.total_tokens} tokens in {writer.shard_index} shards")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> GatherConfig:
    parser = argparse.ArgumentParser(
        description="Gather video activations from Qwen3-VL")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--prompt", type=str,
                        default="Describe what happens in this video.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--max_norm_multiple", type=int, default=10)
    parser.add_argument("--prefetch_workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    return GatherConfig(**vars(args))


if __name__ == "__main__":
    gather_activations(parse_args())
