"""Find top-activating video chunks per SAE feature across a corpus."""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import click
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from eval.heatmap import render_feature_heatmap
from gather_utils import (
    TEMPORAL_STRIDE, chunk_frames_for_vae, load_model_vae,
    multi_module_hooks, preprocess_frames,
)
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
def encode_chunk_features(frames, vae, sae, hook_module, height, width, device):
    """Encode frames through VAE + SAE, return (N, dict_size) features and spatial dims."""
    video_tensor = preprocess_frames(frames, height, width, device)
    if hook_module:
        with multi_module_hooks(vae, [hook_module]) as captured:
            vae.encode(video_tensor)
        activation = captured[hook_module]
    else:
        activation = vae.encode(video_tensor, return_dict=True).latent_dist.mean
    if activation.ndim == 5:
        _, c, t, h, w = activation.shape
        flat = activation.permute(0, 2, 3, 4, 1).reshape(-1, c)
        spatial_dims = (t, h, w)
    elif activation.ndim == 4:
        _, c, h, w = activation.shape
        flat = activation.permute(0, 2, 3, 1).reshape(-1, c)
        spatial_dims = (1, h, w)
    else:
        flat = activation.reshape(-1, activation.shape[-1])
        spatial_dims = (flat.shape[0], 1, 1)
    features = sae.encode(flat.to(sae.W_enc.device))
    del video_tensor, activation
    return features, spatial_dims


def scan_corpus(
    video_paths, vae, sae, hook_module,
    chunk_size, height, width, device, samples_per_feature, temporal_stride=8,
):
    tracker = TopChunkTracker(sae.dict_size, samples_per_feature)
    for video_path in tqdm(video_paths, desc="Scanning corpus"):
        try:
            all_frames = read_all_frames(video_path)
        except Exception:
            continue
        chunks = chunk_frames_for_vae(all_frames, chunk_size, temporal_stride)
        offset = 0
        for chunk in chunks:
            raw_count = min(chunk_size, len(all_frames) - offset)
            try:
                features, _ = encode_chunk_features(
                    chunk, vae, sae, hook_module, height, width, device,
                )
            except Exception:
                offset += chunk_size
                continue
            tracker.update(features.mean(dim=0), video_path, offset, raw_count)
            offset += chunk_size
            del features
            torch.cuda.empty_cache()
        del all_frames
    return tracker


LABEL_WIDTH = 220
LABEL_BG = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)

try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except (OSError, IOError):
    _FONT = ImageFont.load_default()


def render_row_label(lines: list[str], height: int) -> np.ndarray:
    label = Image.new("RGB", (LABEL_WIDTH, height), LABEL_BG)
    draw = ImageDraw.Draw(label)
    y = 4
    for line in lines:
        draw.text((6, y), line, fill=TEXT_COLOR, font=_FONT)
        y += 15
    return np.array(label)


@torch.no_grad()
def render_feature_grid(
    chunks, feature_idx, chunk_size, thumb_height, fire_count,
    vae, sae, hook_module, height, width, device,
):
    strips = []
    for rank, ref in enumerate(chunks, 1):
        frames = read_all_frames(ref.video_path)
        frames = frames[ref.chunk_start:ref.chunk_start + ref.chunk_frame_count]
        while len(frames) < chunk_size:
            frames.append(frames[-1])
        frames = frames[:chunk_size]

        features, (t_lat, h_lat, w_lat) = encode_chunk_features(
            frames, vae, sae, hook_module, height, width, device,
        )
        activation_map = features[:, feature_idx].reshape(t_lat, h_lat, w_lat)
        del features
        torch.cuda.empty_cache()

        thumbs = []
        for frame in frames:
            h, w = frame.shape[:2]
            thumb_w = int(w * thumb_height / h)
            thumbs.append(np.array(
                Image.fromarray(frame).resize((thumb_w, thumb_height), Image.LANCZOS),
            ))
        latent_indices = [
            min(i * t_lat // len(frames), t_lat - 1) for i in range(len(frames))
        ]
        heatmap_strip = np.array(
            render_feature_heatmap(activation_map, thumbs, latent_indices),
        )

        video_name = os.path.basename(ref.video_path)
        if len(video_name) > 30:
            video_name = video_name[:27] + "..."
        end_frame = ref.chunk_start + ref.chunk_frame_count
        label = render_row_label([
            f"#{rank}  score: {ref.score:.4f}",
            video_name,
            f"frames {ref.chunk_start}-{end_frame}",
        ], thumb_height)
        strips.append(np.concatenate([label, heatmap_strip], axis=1))

    max_w = max(s.shape[1] for s in strips)

    # Title row
    title_label = render_row_label([
        f"Feature {feature_idx}  |  {fire_count} fires",
    ], 20)
    title_pad = np.full(
        (20, max_w - LABEL_WIDTH, 3), LABEL_BG[0], dtype=np.uint8,
    )
    title_row = np.concatenate([title_label, title_pad], axis=1)
    rows = [title_row]
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
@click.option("--output-dir", default="top_samples", type=click.Path())
@click.option("--hook-module", default=None)
@click.option("--vae-model", default="Lightricks/LTX-Video-0.9.5")
@click.option("--vae-type", default="ltx", type=click.Choice(["ltx", "wan"]))
@click.option("--num-frames", default=33, type=int, help="Chunk size matching gather")
@click.option("--samples-per-feature", default=10, type=int)
@click.option("--topk-features", default=20, type=int)
@click.option("--feature-indices", default=None, help="Comma-separated feature indices")
@click.option("--min-fire-count", default=2, type=int, help="Exclude features firing on fewer chunks")
@click.option("--thumb-height", default=96, type=int)
@click.option("--max-videos", default=0, type=int, help="0=all")
@click.option("--device", default="cuda")
def main(
    sae_path, video_dir, output_dir, hook_module, vae_model, vae_type, num_frames,
    samples_per_feature, topk_features, feature_indices, min_fire_count,
    thumb_height, max_videos, device,
):
    run_dir = os.path.join(
        output_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    temporal_stride = TEMPORAL_STRIDE[vae_type]

    vae = load_model_vae(vae_model, vae_type, device)
    sae = MatryoshkaBatchTopKSAE.from_pretrained(sae_path, device=device)
    sae.eval()
    video_paths = scan_video_directory(video_dir)
    if max_videos > 0:
        video_paths = video_paths[:max_videos]
    click.echo(f"Scanning {len(video_paths)} videos...")
    tracker = scan_corpus(
        video_paths, vae, sae, hook_module,
        num_frames, 256, 256, device, samples_per_feature, temporal_stride,
    )
    if feature_indices:
        indices = [int(x) for x in feature_indices.split(",")]
    else:
        indices = tracker.most_sparse_features(topk_features, min_fire_count)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "sae_path": os.path.abspath(sae_path),
        "video_dir": os.path.abspath(video_dir),
        "vae_model": vae_model,
        "vae_type": vae_type,
        "hook_module": hook_module,
        "num_frames": num_frames,
        "num_videos_scanned": len(video_paths),
        "samples_per_feature": samples_per_feature,
        "topk_features": topk_features,
        "feature_indices": indices,
        "min_fire_count": min_fire_count,
        "thumb_height": thumb_height,
        "dict_size": sae.dict_size,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"Rendering {len(indices)} features...")
    for feat_idx in tqdm(indices, desc="Rendering"):
        chunks = tracker.get_top_chunks(feat_idx)
        if not chunks:
            continue
        fires = tracker.fire_counts[feat_idx].item()
        image = render_feature_grid(
            chunks, feat_idx, num_frames, thumb_height, fires,
            vae, sae, hook_module, 256, 256, device,
        )
        image.save(os.path.join(run_dir, f"feature_{feat_idx:04d}_fires{fires}.png"))
    click.echo(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
