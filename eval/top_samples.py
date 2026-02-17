"""Find top-activating video chunks per SAE feature across a corpus."""

import os
import sys
from dataclasses import dataclass, field

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from eval.heatmap import render_feature_heatmap
from gather_ltx_activations import load_vae, multi_module_hooks, preprocess_frames
from gather_ltx_encoder_activations import chunk_frames_for_vae
from video_utils import read_all_frames, scan_video_directory


@dataclass(order=True)
class ChunkReference:
    score: float
    video_path: str = field(compare=False)
    chunk_start: int = field(compare=False)
    chunk_frame_count: int = field(compare=False)


class TopChunkTracker:
    """Track top-k scoring chunks per SAE feature via a score tensor."""

    def __init__(self, num_features: int, k: int):
        self.scores = torch.full((num_features, k), -float("inf"))
        self.refs: list[list[ChunkReference | None]] = [
            [None] * k for _ in range(num_features)
        ]
        self.fire_counts = torch.zeros(num_features, dtype=torch.long)

    def update(
        self, mean_features: torch.Tensor,
        video_path: str, chunk_start: int, chunk_frame_count: int,
    ):
        mean_features = mean_features.cpu()
        self.fire_counts += (mean_features > 0).long()
        min_scores, min_slots = self.scores.min(dim=1)
        for idx in (mean_features > min_scores).nonzero(as_tuple=True)[0].tolist():
            slot = min_slots[idx].item()
            score = mean_features[idx].item()
            self.scores[idx, slot] = score
            self.refs[idx][slot] = ChunkReference(
                score, video_path, chunk_start, chunk_frame_count,
            )

    def get_top_chunks(self, feature_idx: int) -> list[ChunkReference]:
        return sorted([r for r in self.refs[feature_idx] if r], reverse=True)

    def most_sparse_features(self, n: int, min_fire_count: int = 2) -> list[int]:
        """Features with fewest chunk activations, excluding dead/too-rare."""
        counts = self.fire_counts.float().clone()
        counts[counts < min_fire_count] = float("inf")
        k = min(n, (counts < float("inf")).sum().item())
        if k == 0:
            return []
        return counts.topk(k, largest=False).indices.tolist()


@torch.no_grad()
def compute_chunk_feature_means(frames, vae, sae, hook_module, height, width, device):
    video_tensor = preprocess_frames(frames, height, width, device)
    if hook_module:
        with multi_module_hooks(vae, [hook_module]) as captured:
            vae.encode(video_tensor)
        activation = captured[hook_module]
    else:
        activation = vae.encode(video_tensor, return_dict=True).latent_dist.mean
    if activation.ndim == 5:
        flat = activation.permute(0, 2, 3, 4, 1).reshape(-1, activation.shape[1])
    elif activation.ndim == 4:
        flat = activation.permute(0, 2, 3, 1).reshape(-1, activation.shape[1])
    else:
        flat = activation.reshape(-1, activation.shape[-1])
    features = sae.encode(flat.to(sae.W_enc.device))
    del video_tensor, activation
    return features.mean(dim=0)


def scan_corpus(
    video_paths, vae, sae, hook_module,
    chunk_size, height, width, device, samples_per_feature,
):
    tracker = TopChunkTracker(sae.dict_size, samples_per_feature)
    for video_path in tqdm(video_paths, desc="Scanning corpus"):
        try:
            all_frames = read_all_frames(video_path)
        except Exception:
            continue
        chunks = chunk_frames_for_vae(all_frames, chunk_size)
        offset = 0
        for chunk in chunks:
            raw_count = min(chunk_size, len(all_frames) - offset)
            try:
                means = compute_chunk_feature_means(
                    chunk, vae, sae, hook_module, height, width, device,
                )
            except Exception:
                offset += chunk_size
                continue
            tracker.update(means, video_path, offset, raw_count)
            offset += chunk_size
            del means
            torch.cuda.empty_cache()
        del all_frames
    return tracker


def render_feature_grid(chunks, chunk_size, thumb_height):
    strips = []
    for ref in chunks:
        frames = read_all_frames(ref.video_path)
        frames = frames[ref.chunk_start:ref.chunk_start + ref.chunk_frame_count]
        while len(frames) < chunk_size:
            frames.append(frames[-1])
        panels = []
        for frame in frames[:chunk_size]:
            h, w = frame.shape[:2]
            thumb_w = int(w * thumb_height / h)
            panels.append(np.array(
                Image.fromarray(frame).resize((thumb_w, thumb_height), Image.LANCZOS),
            ))
        strips.append(np.concatenate(panels, axis=1))
    max_w = max(s.shape[1] for s in strips)
    rows = []
    for strip in strips:
        if strip.shape[1] < max_w:
            pad = np.zeros(
                (thumb_height, max_w - strip.shape[1], 3), dtype=np.uint8,
            )
            strip = np.concatenate([strip, pad], axis=1)
        rows.append(strip)
    return Image.fromarray(np.concatenate(rows, axis=0))


@click.command()
@click.option("--sae-path", required=True, type=click.Path(exists=True))
@click.option("--video-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--hook-module", default=None)
@click.option("--vae-model", default="Lightricks/LTX-Video-0.9.5")
@click.option("--num-frames", default=33, type=int, help="Chunk size matching gather")
@click.option("--samples-per-feature", default=10, type=int)
@click.option("--topk-features", default=20, type=int)
@click.option("--feature-indices", default=None, help="Comma-separated feature indices")
@click.option("--min-fire-count", default=2, type=int, help="Exclude features firing on fewer chunks")
@click.option("--thumb-height", default=96, type=int)
@click.option("--max-videos", default=0, type=int, help="0=all")
@click.option("--device", default="cuda")
def main(
    sae_path, video_dir, output_dir, hook_module, vae_model, num_frames,
    samples_per_feature, topk_features, feature_indices, min_fire_count,
    thumb_height, max_videos, device,
):
    os.makedirs(output_dir, exist_ok=True)
    vae = load_vae(vae_model, device, enable_tiling=False)
    sae = MatryoshkaBatchTopKSAE.from_pretrained(sae_path, device=device)
    sae.eval()
    video_paths = scan_video_directory(video_dir)
    if max_videos > 0:
        video_paths = video_paths[:max_videos]
    click.echo(f"Scanning {len(video_paths)} videos...")
    tracker = scan_corpus(
        video_paths, vae, sae, hook_module,
        num_frames, 256, 256, device, samples_per_feature,
    )
    if feature_indices:
        indices = [int(x) for x in feature_indices.split(",")]
    else:
        indices = tracker.most_sparse_features(topk_features, min_fire_count)
    click.echo(f"Rendering {len(indices)} features...")
    for feat_idx in tqdm(indices, desc="Rendering"):
        chunks = tracker.get_top_chunks(feat_idx)
        if not chunks:
            continue
        image = render_feature_grid(chunks, num_frames, thumb_height)
        fires = tracker.fire_counts[feat_idx].item()
        image.save(os.path.join(output_dir, f"feature_{feat_idx:04d}_fires{fires}.png"))
    click.echo(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
