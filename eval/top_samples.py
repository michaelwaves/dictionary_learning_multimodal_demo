"""Visualize top-activating video chunks per SAE feature."""

from video_utils import read_all_frames, scan_video_directory
from gather_utils import (
    TEMPORAL_STRIDE, chunk_frames_for_vae, load_model_vae,
    multi_module_hooks, preprocess_frames,
)
from eval.top_k_tracker import TopKTracker
from eval.heatmap import render_feature_heatmap
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary_learning.dictionary import AutoEncoder
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


ENCODE_SIZE = 256
THUMB_HEIGHT = 96
DISPLAY_FRAMES = 8
MAX_VIDEOS = 1


@dataclass
class Encoder:
    vae: torch.nn.Module
    sae: torch.nn.Module
    hook_module: str | None
    device: str


@click.command()
@click.option("--sae-path", required=True, type=click.Path(exists=True))
@click.option("--video-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default="top_samples")
@click.option("--hook-module", default=None)
@click.option("--vae-model", default="Lightricks/LTX-Video-0.9.5")
@click.option("--vae-type", default="ltx", type=click.Choice(["ltx", "wan"]))
@click.option("--num-frames", default=33, type=int)
@click.option("--samples-per-feature", default=10, type=int)
@click.option("--num-features", default=20, type=int)
@click.option("--feature-indices", default=None, help="Comma-separated")
@click.option("--device", default="cuda")
def main(
    sae_path, video_dir, output_dir, hook_module,
    vae_model, vae_type, num_frames,
    samples_per_feature, num_features, feature_indices, device,
):
    run_dir = os.path.join(
        output_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    encoder = Encoder(
        vae=load_model_vae(vae_model, vae_type, device),
        sae=MatryoshkaBatchTopKSAE.from_pretrained(sae_path, device=device),
        hook_module=hook_module, device=device,
    )
    encoder.sae.eval()
    stride = TEMPORAL_STRIDE[vae_type]

    paths = scan_video_directory(video_dir)
    paths = paths[:MAX_VIDEOS]
    click.echo(f"Scanning {len(paths)} videos...")
    tracker = scan_corpus(paths, encoder, num_frames,
                          stride, samples_per_feature)

    indices = (
        [int(x) for x in feature_indices.split(",")]
        if feature_indices else tracker.top_features(num_features)
    )
    click.echo(f"Rendering {len(indices)} features...")
    for feat in tqdm(indices, desc="Rendering"):
        chunks = tracker.top_chunks(feat)
        if not chunks:
            continue
        render_feature_grid(chunks, feat, num_frames, encoder).save(
            os.path.join(run_dir, f"feature_{feat:04d}.png"),
        )
    click.echo(f"Saved to {run_dir}")


def scan_corpus(video_paths, encoder, chunk_size, temporal_stride, k):
    tracker = TopKTracker(encoder.sae.dict_size, k)
    for path in tqdm(video_paths, desc="Scanning"):
        frames = read_all_frames(path)
        chunks = chunk_frames_for_vae(frames, chunk_size, temporal_stride)
        for i, chunk in enumerate(chunks):
            offset = i * chunk_size
            features, _ = encode_chunk(chunk, encoder)
            # is this chunk full or the last one(truncated)
            raw_count = min(chunk_size, len(frames) - offset)
            tracker.update(features.mean(dim=0).cpu(), path, offset, raw_count)
    return tracker


def render_feature_grid(chunks, feature_idx, chunk_size, encoder):
    rows = []
    for ref in chunks:
        frames = read_all_frames(ref.video_path)[
            ref.start:ref.start + ref.count]
        while len(frames) < chunk_size:
            frames.append(frames[-1])
        features, (t, h, w) = encode_chunk(frames[:chunk_size], encoder)
        act_map = features[:, feature_idx].reshape(t, h, w)
        sample_idx = np.linspace(0, len(frames) - 1, DISPLAY_FRAMES, dtype=int)
        thumbs = [_resize_thumb(frames[i]) for i in sample_idx]
        lat_idx = [min(i * t // len(frames), t - 1) for i in sample_idx]
        rows.append(np.array(render_feature_heatmap(act_map, thumbs, lat_idx)))
    max_w = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
              for r in rows]
    return Image.fromarray(np.concatenate(padded, axis=0))


@torch.no_grad()
def encode_chunk(frames, encoder):
    tensor = preprocess_frames(
        frames, ENCODE_SIZE, ENCODE_SIZE, encoder.device)
    if encoder.hook_module:
        with multi_module_hooks(encoder.vae, [encoder.hook_module]) as captured:
            encoder.vae.encode(tensor)
        act = captured[encoder.hook_module]
    else:
        act = encoder.vae.encode(tensor, return_dict=True).latent_dist.mean

    if act.ndim == 5:
        b, c, t, h, w = act.shape
        flat, dims = act.permute(0, 2, 3, 4, 1).reshape(-1, c), (t, h, w)
    elif act.ndim == 4:
        b, c, h, w = act.shape
        flat, dims = act.permute(0, 2, 3, 1).reshape(-1, c), (1, h, w)
    else:
        flat = act.reshape(-1, act.shape[-1])
        flat, dims = flat, (flat.shape[0], 1, 1)
    return encoder.sae.encode(flat.to(encoder.sae.W_enc.device)), dims


def _resize_thumb(frame):
    h, w = frame.shape[:2]
    return np.array(Image.fromarray(frame).resize(
        (int(w * THUMB_HEIGHT / h), THUMB_HEIGHT), Image.LANCZOS,
    ))


if __name__ == "__main__":
    main()
