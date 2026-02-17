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
from gather_utils import (
    TEMPORAL_STRIDE, chunk_frames_for_vae, load_model_vae,
    multi_module_hooks, preprocess_frames,
)
from video_utils import read_all_frames, read_consecutive_frames, read_video_pyav


@torch.no_grad()
def extract_feature_activations(video_tensor, vae, sae, hook_module):
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


@torch.no_grad()
def extract_chunked_feature_activations(
    all_frames, vae, sae, hook_module, chunk_size, height, width, device, temporal_stride=8,
):
    chunks = chunk_frames_for_vae(all_frames, chunk_size, temporal_stride)
    if not chunks:
        raise ValueError(f"Video too short to chunk ({len(all_frames)} frames)")
    all_features = []
    lat_h = lat_w = 0
    for chunk in chunks:
        video_tensor = preprocess_frames(chunk, height, width, device)
        features, (_, h, w) = extract_feature_activations(video_tensor, vae, sae, hook_module)
        all_features.append(features)
        lat_h, lat_w = h, w
        del video_tensor
        torch.cuda.empty_cache()
    feature_acts = torch.cat(all_features, dim=0)
    total_temporal = feature_acts.shape[0] // (lat_h * lat_w)
    return feature_acts, total_temporal, lat_h, lat_w


def select_top_features(feature_activations, min_count, max_count, topk):
    fire_counts = (feature_activations > 0).sum(dim=0)
    total_magnitude = feature_activations.sum(dim=0)
    in_range = (fire_counts >= min_count) & (fire_counts <= max_count)
    masked_magnitude = torch.where(in_range, total_magnitude, torch.tensor(-1.0))
    k = min(topk, in_range.sum().item())
    if k == 0:
        return torch.tensor([], dtype=torch.long)
    return masked_magnitude.topk(k).indices


def load_display_frames(video_path, num_display, sampling, num_vae_frames, start_fraction, temporal_stride=8):
    num_latent = (num_vae_frames - 1) // temporal_stride + 1
    if sampling == "consecutive":
        display_frames = read_consecutive_frames(video_path, num_display, start_fraction)
        latent_indices = [0] * num_display
    else:
        all_frames = read_video_pyav(video_path, num_vae_frames, start_fraction)
        selected = sample_frame_indices(len(all_frames), num_display, sampling)
        display_frames = [all_frames[i] for i in selected]
        latent_indices = [min(int(i * num_latent / len(all_frames)), num_latent - 1) for i in selected]
    return display_frames, latent_indices


@click.command()
@click.option("--sae-path", required=True, type=click.Path(exists=True))
@click.option("--video-path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--hook-module", default=None)
@click.option("--vae-model", default="Lightricks/LTX-Video-0.9.5")
@click.option("--vae-type", default="ltx", type=click.Choice(["ltx", "wan"]))
@click.option("--num-frames", default=25)
@click.option("--display-frames", default=8)
@click.option("--sampling", default="spaced", type=click.Choice(["consecutive", "spaced", "random"]))
@click.option("--min-count", default=10)
@click.option("--max-count", default=200)
@click.option("--topk", default=20)
@click.option("--start", default=0.0)
@click.option("--full-video", is_flag=True)
@click.option("--device", default="cuda")
def main(
    sae_path, video_path, output_dir, hook_module, vae_model, vae_type,
    num_frames, display_frames, sampling, min_count, max_count, topk, start,
    full_video, device,
):
    os.makedirs(output_dir, exist_ok=True)
    temporal_stride = TEMPORAL_STRIDE[vae_type]

    click.echo("Loading models...")
    vae = load_model_vae(vae_model, vae_type, device)
    sae = MatryoshkaBatchTopKSAE.from_pretrained(sae_path, device=device)
    sae.eval()

    if full_video:
        click.echo("Reading all frames for chunked encoding...")
        all_frames = read_all_frames(video_path)
        feature_acts, temporal, lat_h, lat_w = extract_chunked_feature_activations(
            all_frames, vae, sae, hook_module,
            chunk_size=num_frames, height=256, width=256, device=device,
            temporal_stride=temporal_stride,
        )
        selected = sample_frame_indices(len(all_frames), display_frames, sampling)
        frames_to_show = [all_frames[i] for i in selected]
        latent_indices = [min(i * temporal // len(all_frames), temporal - 1) for i in selected]
    else:
        vae_frames = read_video_pyav(video_path, num_frames, start)
        video_tensor = preprocess_frames(vae_frames, height=256, width=256, device=device)
        feature_acts, spatial_dims = extract_feature_activations(video_tensor, vae, sae, hook_module)
        temporal, lat_h, lat_w = spatial_dims
        frames_to_show, latent_indices = load_display_frames(
            video_path, display_frames, sampling, num_frames, start, temporal_stride,
        )

    top_indices = select_top_features(feature_acts, min_count, max_count, topk)
    if len(top_indices) == 0:
        click.echo(f"No features fire between {min_count} and {max_count} times.")
        return

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
