import datetime
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
import torch
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.loading_utils import load_video
from diffusers.utils.export_utils import export_to_video
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.video_processor import VideoProcessor
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torchvision.utils import save_image


def load_vae(model_name: str, device: str, enable_tiling: bool):

    print(f"Loading VAE from {model_name} (float32 for stability)...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32, device=device
    )
    vae.eval()
    if enable_tiling:
        vae.enable_tiling()
    return vae


vae = load_vae("Lightricks/LTX-Video-0.9.5", "cuda", enable_tiling=False)
video = load_video("sample_videos/jam.mp4")
video_processor = VideoProcessor()
video_tensor = video_processor.preprocess_video(video, 256, 256)

z = vae.encode(video_tensor, return_dict=True)
recon = None
if isinstance(z, AutoencoderKLOutput):
    z = z.latent_dist.mean
    recon = vae.decode(z)
    if isinstance(recon, DecoderOutput):
        recon = recon.sample
if recon is None:
    raise ValueError("recon is none")

frame = 1000
original_frame = (video_tensor[0, :, frame]+1)/2
recon_frame = (recon[0, :, frame]+1)/2
save_image([original_frame, recon_frame],
           f"recon_check_{datetime.datetime.now().strftime("%d-%m%Y_%H-%M-%S")}.png")

breakpoint()
