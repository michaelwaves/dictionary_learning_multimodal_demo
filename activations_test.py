"""Quick probe: SVD effective dimensionality for various VAE encoder layers."""

import torch
from gather_ltx_activations import (
    load_vae,
    multi_module_hooks,
    reshape_vae_activation,
    preprocess_frames,
)
from video_utils import read_video_pyav, scan_video_directory

LAYERS_TO_PROBE = [
    "encoder.down_blocks.0",
    "encoder.down_blocks.0.resnets.3",
    "encoder.down_blocks.1",
    "encoder.down_blocks.1.resnets.2",
    "encoder.down_blocks.2.resnets.2",
    "encoder.mid_block",
]

vae = load_vae("Lightricks/LTX-Video", "cuda", enable_tiling=False)

# Grab first video from your directory
video_dir = "/mnt/nw/home/m.yu/repos/multimodal_sae/videos"
video_path = scan_video_directory(video_dir)[0]
frames = read_video_pyav(video_path, num_frames=25)
video_tensor = preprocess_frames(frames, height=256, width=256, device="cuda")

# Forward pass with hooks on all layers at once
with torch.no_grad(), multi_module_hooks(vae, LAYERS_TO_PROBE) as captured:
    vae.encode(video_tensor)

# SVD analysis per layer
for path in LAYERS_TO_PROBE:
    act = reshape_vae_activation(captured[path]).float()
    S = torch.linalg.svdvals(act)
    cumvar = (S**2).cumsum(0) / (S**2).sum()

    print(f"\n{path}  shape={act.shape}  d={act.shape[-1]}")
    for k in [1, 5, 10, 20, 50]:
        if k <= len(cumvar):
            print(f"  top {k:>3d} SVs: {cumvar[k-1]:.4%}")
