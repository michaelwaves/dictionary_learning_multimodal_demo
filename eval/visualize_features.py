"""Visualize SAE feature activations as heatmaps over video frames."""

import os
import sys

import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from eval.heatmap import render_feature_heatmap, sample_frame_indices
from gather_ltx_activations import load_vae, multi_module_hooks, preprocess_frames
from video_utils import read_consecutive_frames, read_video_pyav


@torch.no_grad()
def extract_feature_activations(
    video_tensor: torch.Tensor,
    vae: torch.nn.Module,
    sae: MatryoshkaBatchTopKSAE,
    hook_module: str | None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Run VAE encode + SAE encode, return (N, dict_size) features and spatial dims."""
    if hook_module:
        with multi_module_hooks(vae, [hook_module]) as captured:
            vae.encode(video_tensor)
        activation = captured[hook_module]
    else:
        activation = vae.encode(video_tensor, return_dict=True).latent_dist.mean
    if activation.ndim == 5:
        _, channels, temporal, height, width = activation.shape
        spatial_dims = (temporal, height, width)
        flat = activation.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    elif activation.ndim == 4:
        _, channels, height, width = activation.shape
        spatial_dims = (1, height, width)
        flat = activation.permute(0, 2, 3, 1).reshape(-1, channels)
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")

    feature_activations = sae.encode(flat.to(sae.W_enc.device))
    return feature_activations, spatial_dims


def select_top_features(
    feature_activations: torch.Tensor,
    min_count: int,
    max_count: int,
    topk: int,
) -> torch.Tensor:
    """Filter by fire count range, rank by total magnitude, return top-k indices."""
    fire_counts = (feature_activations > 0).sum(dim=0)
    total_magnitude = feature_activations.sum(dim=0)

    in_range = (fire_counts >= min_count) & (fire_counts <= max_count)
    masked_magnitude = torch.where(in_range, total_magnitude, torch.tensor(-1.0))

    num_valid = in_range.sum().item()
    k = min(topk, num_valid)
    if k == 0:
        return torch.tensor([], dtype=torch.long)

    return masked_magnitude.topk(k).indices


def load_display_frames(
    video_path: str,
    num_display: int,
    sampling: str,
    num_vae_frames: int,
    start_fraction: float,
) -> tuple[list, list[int]]:
    """Load display frames and compute their latent temporal indices.

    Returns:
        display_frames: List of (H, W, 3) uint8 arrays.
        latent_indices: Latent temporal index for each display frame.
    """
    num_latent = (num_vae_frames - 1) // 8 + 1

    if sampling == "consecutive":
        display_frames = read_consecutive_frames(
            video_path, num_display, start_fraction,
        )
        latent_indices = [0] * num_display
    else:
        all_frames = read_video_pyav(video_path, num_vae_frames, start_fraction)
        selected = sample_frame_indices(len(all_frames), num_display, sampling)
        display_frames = [all_frames[i] for i in selected]
        latent_indices = [
            min(int(i * num_latent / len(all_frames)), num_latent - 1)
            for i in selected
        ]

    return display_frames, latent_indices


@click.command()
@click.option("--sae-path", required=True, type=click.Path(exists=True))
@click.option("--video-path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--hook-module", default=None, help="VAE layer to hook (omit for full encoder output)")
@click.option("--vae-model", default="Lightricks/LTX-Video-0.9.5")
@click.option("--num-frames", default=25, help="Frames to feed VAE ((n-1) %% 8 == 0)")
@click.option("--display-frames", default=8, help="Frames shown per feature image")
@click.option(
    "--sampling", default="spaced",
    type=click.Choice(["consecutive", "spaced", "random"]),
)
@click.option("--min-count", default=10, help="Min fire count to include a feature")
@click.option("--max-count", default=200, help="Max fire count to include a feature")
@click.option("--topk", default=20, help="Number of top features to visualize")
@click.option("--start", default=0.0, help="Skip this fraction of the video (0.0-1.0)")
@click.option("--device", default="cuda")
def main(
    sae_path, video_path, output_dir, hook_module, vae_model,
    num_frames, display_frames, sampling, min_count, max_count, topk, start, device,
):
    os.makedirs(output_dir, exist_ok=True)

    click.echo("Loading models...")
    vae = load_vae(vae_model, device, enable_tiling=False)
    sae = MatryoshkaBatchTopKSAE.from_pretrained(sae_path, device=device)
    sae.eval()

    click.echo(f"Reading {num_frames} frames for VAE (start={start:.0%})...")
    vae_frames = read_video_pyav(video_path, num_frames, start)
    video_tensor = preprocess_frames(vae_frames, height=256, width=256, device=device)

    click.echo("Extracting feature activations...")
    feature_acts, spatial_dims = extract_feature_activations(
        video_tensor, vae, sae, hook_module,
    )
    temporal, lat_h, lat_w = spatial_dims

    click.echo("Selecting top features...")
    top_indices = select_top_features(feature_acts, min_count, max_count, topk)
    if len(top_indices) == 0:
        click.echo(f"No features fire between {min_count} and {max_count} times.")
        return

    click.echo(f"Loading {display_frames} display frames ({sampling})...")
    frames_to_show, latent_indices = load_display_frames(
        video_path, display_frames, sampling, num_frames, start,
    )

    fire_counts = (feature_acts > 0).sum(dim=0)
    click.echo(f"Rendering {len(top_indices)} feature heatmaps...")
    for feature_idx in top_indices:
        idx = feature_idx.item()
        count = fire_counts[idx].item()
        activation_map = feature_acts[:, idx].reshape(temporal, lat_h, lat_w)

        image = render_feature_heatmap(activation_map, frames_to_show, latent_indices)
        image.save(os.path.join(output_dir, f"feature_{idx:04d}_count{count}.png"))

    click.echo(f"Saved {len(top_indices)} heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
