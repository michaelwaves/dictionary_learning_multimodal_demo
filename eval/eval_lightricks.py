import torch
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

SAE_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-06_08-18-08_lightricks/trainer_2/ae.pt"
VIDEOS_DIR = "videos_celebdf"
VIDEO_PATH = "/mnt/nw/home/m.yu/repos/dictionary_learning_demo/videos_celebdf/fake/id0_id1_0000.mp4"
MODEL = "Lightricks/LTX-Video-0.9.5"
D_MODEL = 128
DEVICE = 'cuda'


vae = AutoencoderKLLTXVideo.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.float32,
).to(DEVICE)
