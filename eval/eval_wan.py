import torch
import numpy as np
import cv2
from diffusers import AutoencoderKLWan
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from dictionary_learning.dictionary_learning import AutoEncoder
from video_utils import read_all_frames
from gather_utils import preprocess_frames

# SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-17_04-47-16_wan/resid_post_layer_all/trainer_1/ae.pt"
SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-19_05-39-14_wan_standard/vae_latent_mean/trainer_6/ae.pt"
VIDEO_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/videos_celebdf/fake/id0_id1_0000.mp4"
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
D_MODEL = 48
DEVICE = "cuda"
HEIGHT = WIDTH = 256
TEMPORAL_STRIDE = 4
feature_ids = range(0, 4096, 100)


def align_frame_count(n: int, stride: int = TEMPORAL_STRIDE) -> int:
    if n <= 1:
        return 1
    return ((n - 2) // stride) * stride + 1


vae = AutoencoderKLWan.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.float32,
).to(DEVICE)
vae.eval()
# sae = MatryoshkaBatchTopKSAE.from_pretrained(SAE_PATH).to(DEVICE)

sae = AutoEncoder.from_pretrained(SAE_PATH).to(DEVICE)
print("Reading video")
all_frames = read_all_frames(VIDEO_PATH)
aligned_count = align_frame_count(len(all_frames))
video_frames = np.stack(all_frames[:aligned_count])
video_tensor = preprocess_frames(list(video_frames), HEIGHT, WIDTH, DEVICE)

print("Encoding VAE latents")
with torch.no_grad():
    latent_dist = vae.encode(video_tensor, return_dict=True).latent_dist
    latent_mean = latent_dist.mean

B, C, T_P, H_P, W_P = latent_mean.shape
act = latent_mean.permute(0, 2, 3, 4, 1).reshape(-1, C).float()

print(
    f"act shape: {act.shape}, min: {act.min():.4f}, max: {act.max():.4f}, mean: {act.mean():.4f}")

print("Encoding SAE features")
feats = sae.encode(act)

print(
    f"feats shape: {feats.shape}, min: {feats.min():.4f}, max: {feats.max():.4f}")
print(
    f"nonzero features per token: {(feats > 0).float().sum(dim=1).mean():.1f}")
print(
    f"active feature ids (top 20): {feats.sum(dim=0).topk(20).indices.tolist()}")

H_VID, W_VID = video_frames.shape[1], video_frames.shape[2]

for feature_id in feature_ids:
    feat_map = feats[:, feature_id].reshape(
        T_P, H_P, W_P).cpu().float().detach().numpy()

    out = cv2.VideoWriter(
        f"export/wan/standard/out_{feature_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), 30, (W_VID, H_VID),
    )

    for t, frame in enumerate(video_frames):
        t_p = min(t // TEMPORAL_STRIDE, T_P - 1)
        heatmap = cv2.resize(feat_map[t_p], (W_VID, H_VID))
        heatmap = (heatmap / (feat_map.max() + 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            frame[:, :, ::-1].copy(), 0.6, colored, 0.4, 0)
        out.write(blended)

    out.release()

breakpoint()
