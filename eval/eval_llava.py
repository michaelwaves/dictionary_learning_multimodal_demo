import os
from dataclasses import dataclass

import cv2
import hydra
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

from dictionary_learning.dictionary_learning.dictionary import JumpReluAutoEncoder
from video_utils import read_all_frames

MODEL_NAME = "llava-hf/LLaVA-NeXT-Video-7B-hf"
PROMPT = "USER: <video>\nDescribe this video.\nASSISTANT:"


@dataclass
class VideoLayout:
    num_model_frames: int
    tokens_per_frame: int
    patch_height: int
    patch_width: int
    video_token_start: int


@hydra.main(config_path="configs", config_name="eval_llava", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.export_dir, exist_ok=True)

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=cfg.device)
    model.eval()
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_NAME)
    sae = load_sae(cfg.sae_path, cfg.d_model, cfg.d_sae, cfg.device)

    all_frames = read_all_frames(cfg.video_path)
    end_frame = cfg.end_frame if cfg.end_frame is not None else len(all_frames)
    video_frames = np.stack(all_frames[cfg.start_frame:end_frame])

    inputs = processor(
        text=PROMPT, videos=[video_frames], return_tensors="pt").to(cfg.device)

    features, layout = extract_video_features(model, sae, inputs, cfg)

    video_feats = features[
        layout.video_token_start:
        layout.video_token_start + layout.num_model_frames * layout.tokens_per_frame
    ]

    start = cfg.start_feature
    end = cfg.end_feature if cfg.end_feature is not None else cfg.d_sae

    for feature_id in range(start, end, cfg.feature_step):
        export_feature_heatmap_video(
            video_frames, video_feats, feature_id, layout, cfg.export_dir)

    num_exported = len(range(start, end, cfg.feature_step))
    print(f"Exported {num_exported} feature videos to {cfg.export_dir}")


def load_sae(path: str, d_model: int, d_sae: int, device: str) -> JumpReluAutoEncoder:
    sae = JumpReluAutoEncoder(activation_dim=d_model, dict_size=d_sae, device=device)
    sae.load_state_dict(load_file(path, device=device))
    return sae


def extract_video_features(model, sae, inputs, cfg):
    hidden_states = {}

    def hook_fn(module, input, output):
        hidden_states["act"] = output[0].detach()

    layer = model.model.language_model.layers[cfg.layer]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)

    handle.remove()

    activations = hidden_states["act"].reshape(-1, cfg.d_model).float()
    features = sae.encode(activations)

    layout = compute_video_layout(model, inputs, activations.shape[0])
    return features, layout


def compute_video_layout(model, inputs, total_tokens: int) -> VideoLayout:
    input_ids = inputs["input_ids"].squeeze()
    placeholder_mask = input_ids == model.config.video_token_index
    video_start = placeholder_mask.nonzero()[0].item()
    num_text_tokens = len(input_ids) - placeholder_mask.sum().item()
    num_video_tokens = total_tokens - num_text_tokens
    num_model_frames = inputs["pixel_values_videos"].shape[1]
    tokens_per_frame = num_video_tokens // num_model_frames
    patch_size = int(tokens_per_frame ** 0.5)
    return VideoLayout(
        num_model_frames=num_model_frames,
        tokens_per_frame=tokens_per_frame,
        patch_height=patch_size,
        patch_width=patch_size,
        video_token_start=video_start,
    )


def export_feature_heatmap_video(
    video_frames: np.ndarray,
    video_feats: torch.Tensor,
    feature_id: int,
    layout: VideoLayout,
    export_dir: str,
):
    feature_map = (
        video_feats[:, feature_id]
        .reshape(layout.num_model_frames, layout.patch_height, layout.patch_width)
        .cpu().float().detach().numpy()
    )

    h_vid, w_vid = video_frames.shape[1], video_frames.shape[2]
    output_path = os.path.join(export_dir, f"feature_{feature_id}.mp4")

    total_frames = len(video_frames)
    max_activation = feature_map.max() + 1e-8

    frames = []
    for frame_idx, frame in enumerate(video_frames):
        t_p = min(frame_idx * layout.num_model_frames // total_frames,
                  layout.num_model_frames - 1)
        heatmap = cv2.resize(feature_map[t_p], (w_vid, h_vid))
        heatmap = (heatmap / max_activation * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, colored_heatmap[:, :, ::-1], 0.4, 0)
        frames.append(blended)

    iio.imwrite(output_path, np.stack(frames), fps=30, codec="libx264")


if __name__ == "__main__":
    main()
