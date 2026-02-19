import os
from dataclasses import dataclass

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoVideoProcessor

from dictionary_learning.dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from video_utils import read_all_frames

MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"


@dataclass
class PatchGrid:
    temporal: int
    height: int
    width: int


@hydra.main(config_path="configs", config_name="eval_vjepa2", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.export_dir, exist_ok=True)

    model = AutoModel.from_pretrained(MODEL_NAME, device_map=cfg.device)
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
    sae = BatchTopKSAE.from_pretrained(cfg.sae_path, device=cfg.device)

    all_frames = read_all_frames(cfg.video_path)
    start_frame = cfg.start_frame
    end_frame = cfg.end_frame if cfg.end_frame is not None else len(all_frames)
    video_frames = np.stack(all_frames[start_frame:end_frame])
    print(f"Using frames {start_frame}:{end_frame} of {len(all_frames)} total")

    inputs = processor(video_frames, return_tensors="pt")
    inputs = {k: v.to(cfg.device) if isinstance(v, torch.Tensor)
              else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_state = outputs.hidden_states[cfg.layer + 1]
    activations = hidden_state.reshape(-1, hidden_state.shape[-1]).float()
    features = sae.encode(activations)

    grid = compute_patch_grid(model, inputs)
    dict_size = features.shape[1]
    print(f"Grid: {grid} | Dict size: {dict_size} | Video frames: {len(video_frames)}")

    start = cfg.start_feature
    end = cfg.end_feature if cfg.end_feature is not None else dict_size

    for feature_id in range(start, end, cfg.feature_step):
        export_feature_heatmap_video(
            video_frames, features, feature_id, grid, cfg.export_dir)

    num_exported = len(range(start, end, cfg.feature_step))
    print(f"Exported {num_exported} feature videos to {cfg.export_dir}")


def compute_patch_grid(model, inputs) -> PatchGrid:
    pixel_values = inputs["pixel_values_videos"]
    _, num_frames, _, height, width = pixel_values.shape
    return PatchGrid(
        temporal=num_frames // model.config.tubelet_size,
        height=height // model.config.patch_size,
        width=width // model.config.patch_size,
    )


def export_feature_heatmap_video(
    video_frames: np.ndarray,
    features: torch.Tensor,
    feature_id: int,
    grid: PatchGrid,
    export_dir: str,
):
    feature_map = (
        features[:, feature_id]
        .reshape(grid.temporal, grid.height, grid.width)
        .cpu()
        .float()
        .detach()
        .numpy()
    )

    h_vid, w_vid = video_frames.shape[1], video_frames.shape[2]
    output_path = os.path.join(export_dir, f"feature_{feature_id}.mp4")
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w_vid, h_vid))

    total_frames = len(video_frames)
    max_activation = feature_map.max() + 1e-8

    for frame_idx, frame in enumerate(video_frames):
        t_p = min(frame_idx * grid.temporal // total_frames, grid.temporal - 1)
        heatmap = cv2.resize(feature_map[t_p], (w_vid, h_vid))
        heatmap = (heatmap / max_activation * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            frame[:, :, ::-1].copy(), 0.6, colored_heatmap, 0.4, 0)
        writer.write(blended)

    writer.release()


if __name__ == "__main__":
    main()
