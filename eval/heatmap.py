"""Render SAE feature activations over video frames (heatmap or patch mode)."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

try:
    _VALUE_FONT = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 9,
    )
except (OSError, IOError):
    _VALUE_FONT = ImageFont.load_default()


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


def render_feature_patches(
    activation_map: torch.Tensor,
    display_frames: list[np.ndarray],
    latent_indices: list[int],
    min_opacity: float = 0.15,
    border_width: int = 2,
) -> Image.Image:
    """Render patch-based visualization with opacity proportional to activation.

    Each patch's brightness scales linearly from min_opacity (zero activation)
    to 1.0 (max activation). Active patches get a green border.
    """
    frame_height, frame_width = display_frames[0].shape[:2]

    global_max = activation_map.max().item()
    if global_max == 0:
        global_max = 1.0

    panels = []
    for frame_rgb, lat_idx in zip(display_frames, latent_indices):
        heat = activation_map[lat_idx].unsqueeze(0).unsqueeze(0).float()
        normalized = F.interpolate(
            heat, size=(frame_height, frame_width), mode="nearest",
        ).squeeze().cpu().numpy() / global_max

        opacity = min_opacity + (1.0 - min_opacity) * normalized.clip(0, 1)
        result = (frame_rgb.astype(np.float32) * opacity[:, :, None]).astype(np.uint8)

        if border_width > 0:
            result = _draw_active_patch_borders(
                result, activation_map[lat_idx], border_width,
            )
        panels.append(result)

    grid = np.concatenate(panels, axis=1)
    return Image.fromarray(grid)


def render_feature_values(
    activation_map: torch.Tensor,
    display_frames: list[np.ndarray],
    latent_indices: list[int],
    alpha: float = 0.4,
) -> Image.Image:
    """Heatmap overlay with per-patch activation values drawn as text."""
    frame_h, frame_w = display_frames[0].shape[:2]
    colormap = cm.get_cmap("hot")
    global_max = activation_map.max().item() or 1.0

    panels = []
    for frame_rgb, lat_idx in zip(display_frames, latent_indices):
        act = activation_map[lat_idx]
        heat = F.interpolate(
            act.float()[None, None], size=(frame_h, frame_w),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy() / global_max
        heat_rgb = (colormap(heat.clip(0, 1))[:, :, :3] * 255).astype(np.uint8)
        blended = (
            (1 - alpha) * frame_rgb.astype(np.float32)
            + alpha * heat_rgb.astype(np.float32)
        ).astype(np.uint8)

        img = Image.fromarray(blended)
        draw = ImageDraw.Draw(img)
        h_lat, w_lat = act.shape
        patch_h, patch_w = frame_h // h_lat, frame_w // w_lat
        vals = act.cpu().numpy()
        for i in range(h_lat):
            for j in range(w_lat):
                if vals[i, j] <= 0:
                    continue
                cx = j * patch_w + patch_w // 2
                cy = i * patch_h + patch_h // 2
                txt = f"{vals[i, j]:.1f}"
                draw.text((cx + 1, cy + 1), txt, fill=(0, 0, 0),
                          anchor="mm", font=_VALUE_FONT)
                draw.text((cx, cy), txt, fill=(255, 255, 255),
                          anchor="mm", font=_VALUE_FONT)
        panels.append(np.array(img))

    return Image.fromarray(np.concatenate(panels, axis=1))


def _draw_active_patch_borders(
    frame: np.ndarray,
    patch_activations: torch.Tensor,
    border_width: int,
) -> np.ndarray:
    h_lat, w_lat = patch_activations.shape
    frame_h, frame_w = frame.shape[:2]
    patch_h, patch_w = frame_h // h_lat, frame_w // w_lat
    green = np.array([0, 255, 0], dtype=np.uint8)
    bw = border_width

    for i in range(h_lat):
        for j in range(w_lat):
            if patch_activations[i, j] <= 0:
                continue
            y0, y1 = i * patch_h, min((i + 1) * patch_h, frame_h)
            x0, x1 = j * patch_w, min((j + 1) * patch_w, frame_w)
            frame[y0:y0 + bw, x0:x1] = green
            frame[y1 - bw:y1, x0:x1] = green
            frame[y0:y1, x0:x0 + bw] = green
            frame[y0:y1, x1 - bw:x1] = green
    return frame
