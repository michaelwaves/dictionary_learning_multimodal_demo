import torch
import numpy as np
import cv2
from diffusers import AutoencoderKLLTXVideo
from dictionary_learning.dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from video_utils import read_all_frames
from gather_utils import preprocess_frames

# SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-06_08-18-08_lightricks/trainer_2/ae.pt"

SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-19_05-37-03_lightricks_standard/vae_latent_mean/trainer_8/ae.pt"
VIDEO_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/videos_celebdf/fake/id0_id1_0000.mp4"
MODEL = "Lightricks/LTX-Video-0.9.5"
D_MODEL = 128
DEVICE = "cuda"
HEIGHT = WIDTH = 256
feature_ids = range(0, 8000, 100)


vae = AutoencoderKLLTXVideo.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.float32,
).to(DEVICE)
vae.eval()
sae = AutoEncoder.from_pretrained(SAE_PATH).to(DEVICE)

TEMPORAL_STRIDE = 8


def align_frame_count(n: int, stride: int = TEMPORAL_STRIDE) -> int:
    if n <= 1:
        return 1
    return ((n - 2) // stride) * stride + 1


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

print("Encoding SAE features")
feats = sae.encode(act)

H_VID, W_VID = video_frames.shape[1], video_frames.shape[2]

for feature_id in feature_ids:
    feat_map = feats[:, feature_id].reshape(
        T_P, H_P, W_P).cpu().float().detach().numpy()

    out = cv2.VideoWriter(
        f"export/lightricks/standard/out_{feature_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), 30, (W_VID, H_VID),
    )

    for t, frame in enumerate(video_frames):
        t_p = min(t // 8, T_P - 1)
        heatmap = cv2.resize(feat_map[t_p], (W_VID, H_VID))
        heatmap = (heatmap / (feat_map.max() + 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            frame[:, :, ::-1].copy(), 0.6, colored, 0.4, 0)
        out.write(blended)

    out.release()

breakpoint()
