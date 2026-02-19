

from transformers import AutoModel, AutoVideoProcessor
from dictionary_learning.dictionary_learning.dictionary import AutoEncoder
from video_utils import read_all_frames
import torch
import numpy as np
import cv2

SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-18_17-39-25_vjepa_standard/layer_16/trainer_3/ae.pt"
VIDEOS_DIR = "videos_celebdf"
VIDEO_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/videos_celebdf/fake/id0_id1_0000.mp4"
MODEL = "facebook/vjepa2-vitl-fpc64-256"
D_MODEL = 1024
DEVICE = 'cuda'

model = AutoModel.from_pretrained(MODEL, device_map=DEVICE)
processor = AutoVideoProcessor.from_pretrained(MODEL)
sae = AutoEncoder.from_pretrained(SAE_PATH).to(DEVICE)

print("Reading vid")
video_frames = np.stack(read_all_frames(VIDEO_PATH))
video_tensor = processor(video_frames, return_tensors="pt", device=DEVICE)

print("Collecting acts")
with torch.no_grad():
    outputs = model(**video_tensor,
                    output_hidden_states=True)


print("Collected acts")
act = outputs.hidden_states[17].reshape(-1, D_MODEL).float()
print("Getting sae features")
feats = sae.encode(act)
print("Got sae features")


T_P = 234
H_P = W_P = 16

feature_ids = range(420, 1000, 10)
for feature_id in feature_ids:
    feat_map = feats[:, feature_id].reshape(
        T_P, H_P, W_P).cpu().float().detach().numpy()

    H_VID, W_VID = video_frames.shape[1], video_frames.shape[2]
    out = cv2.VideoWriter(f"export/out_{feature_id}.mp4", cv2.VideoWriter_fourcc(
        *"mp4v"), 30, (W_VID, H_VID))

    for t, frame in enumerate(video_frames):
        t_p = min(t // 2, T_P - 1)
        heatmap = cv2.resize(feat_map[t_p], (W_VID, H_VID))
        heatmap = (heatmap / (feat_map.max() + 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            frame[:, :, ::-1].copy(), 0.6, colored, 0.4, 0)
        out.write(blended)

    out.release()


breakpoint()
