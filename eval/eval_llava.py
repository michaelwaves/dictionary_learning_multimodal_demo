import torch
import numpy as np
import cv2
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from dictionary_learning.dictionary_learning.dictionary import JumpReluAutoEncoder
from safetensors.torch import load_file
from video_utils import read_all_frames, read_video_pyav

SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-04_llava/sae_weights.safetensors"
VIDEO_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/videos_celebdf/fake/id0_id1_0000.mp4"
MODEL = "llava-hf/LLaVA-NeXT-Video-7B-hf"
HOOK_LAYER = 16
D_MODEL = 4096
D_SAE = 65536
DEVICE = "cuda"
PROMPT = "USER: <video>\nDescribe this video.\nASSISTANT:"
feature_ids = range(0, 1000, 10)


def load_sae(path: str, d_model: int, d_sae: int, device: str) -> JumpReluAutoEncoder:
    sae = JumpReluAutoEncoder(activation_dim=d_model,
                              dict_size=d_sae, device=device)
    sae.load_state_dict(load_file(path, device=device))
    return sae


print("Loading model")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map=DEVICE,
)
model.eval()
processor = LlavaNextVideoProcessor.from_pretrained(MODEL)
sae = load_sae(SAE_PATH, D_MODEL, D_SAE, DEVICE)

print("Reading video")
video_frames = np.stack(read_all_frames(VIDEO_PATH))
inputs = processor(text=PROMPT, videos=[
                   video_frames], return_tensors="pt").to(DEVICE)

print("Collecting activations")
hidden_states = {}


def make_hook(name):
    def hook_fn(module, input, output):
        hidden_states[name] = output[0].detach()
    return hook_fn


layer = model.model.language_model.layers[HOOK_LAYER]
handle = layer.register_forward_hook(make_hook(HOOK_LAYER))

with torch.no_grad():
    model.generate(**inputs, max_new_tokens=1)

handle.remove()

act = hidden_states[HOOK_LAYER].reshape(-1, D_MODEL).float()
print(f"act shape: {act.shape}, min: {act.min():.4f}, max: {act.max():.4f}")

print("Encoding SAE features")
feats = sae.encode(act)
print(
    f"feats shape: {feats.shape}, nonzero per token: {(feats > 0).float().sum(dim=1).mean():.1f}")
print(
    f"active feature ids (top 20): {feats.sum(dim=0).topk(20).indices.tolist()}")

input_ids = inputs["input_ids"].squeeze()
placeholder_mask = input_ids == model.config.video_token_index
video_start = placeholder_mask.nonzero()[0].item()
num_text_tokens = len(input_ids) - placeholder_mask.sum().item()
num_video_tokens = act.shape[0] - num_text_tokens
num_model_frames = inputs["pixel_values_videos"].shape[1]
tokens_per_frame = num_video_tokens // num_model_frames
H_P = W_P = int(tokens_per_frame ** 0.5)
print(
    f"text tokens: {num_text_tokens}, video tokens: {num_video_tokens}, "
    f"model frames: {num_model_frames}, per frame: {tokens_per_frame} ({H_P}x{W_P})")

video_feats = feats[video_start:video_start + num_model_frames * H_P * W_P]
num_all_frames = len(video_frames)
H_VID, W_VID = video_frames.shape[1], video_frames.shape[2]
fps = cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS)

for feature_id in feature_ids:
    feat_map = video_feats[:, feature_id].reshape(
        num_model_frames, H_P, W_P).cpu().float().detach().numpy()
    feat_max = feat_map.max() + 1e-8

    out = cv2.VideoWriter(
        f"export/out_{feature_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_VID, H_VID),
    )

    for t, frame in enumerate(video_frames):
        model_t = int(t * num_model_frames / num_all_frames)
        model_t = min(model_t, num_model_frames - 1)
        heatmap = cv2.resize(feat_map[model_t], (W_VID, H_VID))
        heatmap = (heatmap / feat_max * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            frame[:, :, ::-1].copy(), 0.6, colored, 0.4, 0)
        out.write(blended)

    out.release()
