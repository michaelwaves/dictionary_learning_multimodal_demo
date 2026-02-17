"""Config and model dispatch for diffusion activation gathering."""

import logging

import torch

from gather_utils import (
    DEFAULT_MODEL_NAMES, SPATIAL_DIVISOR, TEMPORAL_STRIDE,
    _DTYPE_MAP, reshape_vae_activation,
)

logger = logging.getLogger(__name__)


class DiffusionGatherConfig:
    def __init__(self, model_type="ltx", hook_target="vae_latent_mean", hook_modules=None,
                 model_name="", video_dir="", image_dir="", output_dir="",
                 device="cuda:0", dtype="bfloat16", num_frames=17, height=256, width=256,
                 text_prompt="", max_text_seq_len=128, num_train_timesteps=1000,
                 shift=1.0, shard_size=50, max_norm_multiple=10, prefetch_workers=4,
                 full_video=False, max_videos=0):
        self.model_type = model_type
        self.hook_target = hook_target
        self.hook_modules = hook_modules
        self.model_name = model_name or DEFAULT_MODEL_NAMES[model_type]
        self.video_dir = video_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.device = device
        self.dtype = dtype
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.text_prompt = text_prompt
        self.max_text_seq_len = max_text_seq_len
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.shard_size = shard_size
        self.max_norm_multiple = max_norm_multiple
        self.prefetch_workers = prefetch_workers
        self.full_video = full_video
        self.max_videos = max_videos
        self._validate()

    @property
    def temporal_stride(self) -> int:
        return TEMPORAL_STRIDE[self.model_type]

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)

    def _validate(self):
        stride = self.temporal_stride
        if (self.num_frames - 1) % stride != 0:
            raise ValueError(f"num_frames must satisfy (n-1)%{stride}==0, got {self.num_frames}")
        divisor = SPATIAL_DIVISOR[self.model_type]
        if self.height % divisor != 0 or self.width % divisor != 0:
            raise ValueError(f"height/width must be divisible by {divisor}")
        if self.hook_target in ("transformer", "vae_encoder") and not self.hook_modules:
            raise ValueError(f"--hook-modules required for hook_target={self.hook_target}")


def load_diffusion_models(config: DiffusionGatherConfig):
    if config.model_type == "wan":
        from wan_model import compute_wan_text_embeddings, load_wan_transformer, load_wan_vae
        vae = load_wan_vae(config.model_name, config.device)
        if config.hook_target == "transformer":
            transformer = load_wan_transformer(config.model_name, config.device, config.torch_dtype)
            prompt_embeds, prompt_mask = compute_wan_text_embeddings(
                config.model_name, config.text_prompt, config.max_text_seq_len,
                config.device, next(transformer.parameters()).dtype,
            )
            return vae, transformer, prompt_embeds, prompt_mask
        return vae, None, None, None
    from ltx_model import compute_ltx_text_embeddings, load_ltx_transformer, load_ltx_vae
    vae = load_ltx_vae(config.model_name, config.device,
                       enable_tiling=(config.hook_target == "transformer"))
    if config.hook_target == "transformer":
        transformer = load_ltx_transformer(config.model_name, config.device, config.torch_dtype)
        prompt_embeds, prompt_mask = compute_ltx_text_embeddings(
            config.model_name, config.text_prompt, config.max_text_seq_len,
            config.device, next(transformer.parameters()).dtype,
        )
        return vae, transformer, prompt_embeds, prompt_mask
    return vae, None, None, None


def run_forward_pass(config, video_tensor, vae, transformer, prompt_embeds, prompt_mask):
    if config.hook_target == "vae_latent_mean":
        latent_mean = vae.encode(video_tensor, return_dict=True).latent_dist.mean
        return {"vae_latent_mean": reshape_vae_activation(latent_mean).float()}
    if config.model_type == "wan":
        from wan_model import process_wan_transformer
        return process_wan_transformer(
            video_tensor, vae, transformer, config.hook_modules,
            prompt_embeds, prompt_mask, config.num_train_timesteps,
        )
    from ltx_model import process_ltx_transformer, process_ltx_vae_encoder
    if config.hook_target == "vae_encoder":
        return process_ltx_vae_encoder(video_tensor, vae, config.hook_modules)
    return process_ltx_transformer(
        video_tensor, vae, transformer, config.hook_modules,
        prompt_embeds, prompt_mask, config.num_train_timesteps, config.shift,
    )


def probe_d_in(config, vae, transformer, prompt_embeds, prompt_mask):
    if config.hook_target == "vae_latent_mean":
        dummy = torch.randn(1, 3, 5, 128, 128, device=config.device).clamp_(-1, 1)
        result = vae.encode(dummy, return_dict=True).latent_dist.mean
        d_in = reshape_vae_activation(result).shape[-1]
        logger.info(f"Probed d_in={d_in} for vae_latent_mean")
        return {"vae_latent_mean": d_in}
    if config.model_type == "wan":
        from wan_model import probe_wan_d_in
        return probe_wan_d_in(
            config.device, vae, transformer, config.hook_modules,
            prompt_embeds, prompt_mask, config.num_train_timesteps,
        )
    from ltx_model import probe_ltx_d_in
    return probe_ltx_d_in(
        config.hook_target, config.device, vae, transformer, config.hook_modules,
        prompt_embeds, prompt_mask, config.num_train_timesteps, config.shift,
    )
