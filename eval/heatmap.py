"""Heatmap rendering: overlay SAE feature activations on video frames."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm


def sample_frame_indices(
    num_total_frames: int, num_display: int, sampling: str
) -> list[int]:
    if num_display >= num_total_frames:
        return list(range(num_total_frames))

    if sampling == "spaced":
        return np.linspace(0, num_total_frames - 1, num_display, dtype=int).tolist()
    elif sampling == "random":
        return sorted(np.random.choice(num_total_frames, num_display, replace=False).tolist())
    else:
        return list(range(num_display))


def render_feature_heatmap(
    activation_map: torch.Tensor,
    display_frames: list[np.ndarray],
    latent_indices: list[int],
    alpha: float = 0.5,
) -> Image.Image:
    """Render a grid of heatmap-overlaid frames for one feature.

    Args:
        activation_map: (T_lat, H_lat, W_lat) per-latent-frame activation magnitudes.
        display_frames: List of (H, W, 3) uint8 arrays to overlay on.
        latent_indices: Which latent temporal index each display frame maps to.
        alpha: Heatmap opacity.
    """
    frame_height, frame_width = display_frames[0].shape[:2]
    colormap = cm.get_cmap("hot")

    global_max = activation_map.max().item()
    if global_max == 0:
        global_max = 1.0

    panels = []
    for frame_rgb, lat_idx in zip(display_frames, latent_indices):
        heat = activation_map[lat_idx].unsqueeze(0).unsqueeze(0).float()
        heat = F.interpolate(
            heat, size=(frame_height, frame_width), mode="bilinear", align_corners=False
        )
        heat_np = (heat.squeeze().cpu().numpy() / global_max).clip(0, 1)

        heat_rgba = (colormap(heat_np)[:, :, :3] * 255).astype(np.uint8)
        blended = (
            (1 - alpha) * frame_rgb.astype(np.float32)
            + alpha * heat_rgba.astype(np.float32)
        ).astype(np.uint8)
        panels.append(blended)

    grid = np.concatenate(panels, axis=1)
    return Image.fromarray(grid)
