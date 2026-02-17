"""LTX-Video model loading and forward passes for activation gathering."""

import logging

import torch

from gather_utils import multi_module_hooks, reshape_transformer_activation, reshape_vae_activation

logger = logging.getLogger(__name__)


def load_ltx_vae(model_name: str, device: str, enable_tiling: bool = False):
    from diffusers import AutoencoderKLLTXVideo

    logger.info(f"Loading LTX VAE from {model_name}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    if enable_tiling:
        vae.enable_tiling()
    return vae


def load_ltx_transformer(model_name: str, device: str, dtype: torch.dtype):
    from diffusers import LTXVideoTransformer3DModel

    logger.info(f"Loading LTX transformer from {model_name} ({dtype})...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    transformer.eval()
    return transformer


def compute_ltx_text_embeddings(
    model_name: str, text_prompt: str, max_seq_len: int,
    device: str, cast_dtype: torch.dtype, free_after: bool = True,
):
    from transformers import AutoTokenizer, T5EncoderModel

    logger.info("Loading T5 text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=cast_dtype,
    ).to(device)
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_inputs = tokenizer(
        text_prompt, padding="max_length", max_length=max_seq_len,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=cast_dtype)
    attention_mask = text_inputs.attention_mask.to(device)
    if free_after:
        del text_encoder, tokenizer
        torch.cuda.empty_cache()
    return prompt_embeds, attention_mask


@torch.no_grad()
def process_ltx_vae_encoder(video_tensor, vae, hook_modules):
    with multi_module_hooks(vae, hook_modules) as captured:
        vae.encode(video_tensor)
    return {p: reshape_vae_activation(a).float() for p, a in captured.items()}


@torch.no_grad()
def process_ltx_transformer(
    video_tensor, vae, transformer, hook_modules,
    prompt_embeds, prompt_mask, num_train_timesteps, shift,
):
    device = video_tensor.device
    transformer_dtype = next(transformer.parameters()).dtype

    latent_dist = vae.encode(video_tensor).latent_dist
    latents = latent_dist.sample()
    scaling_factor = vae.config.scaling_factor
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * scaling_factor

    B, C, Fl, Hl, Wl = latents.shape
    latents = latents.permute(0, 2, 3, 4, 1).reshape(B, Fl * Hl * Wl, C)

    sigma = torch.rand(1, device=device, dtype=latents.dtype)
    if shift != 1.0:
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
    noised = ((1.0 - sigma) * latents + sigma * torch.randn_like(latents)).to(transformer_dtype)
    timestep = (sigma * num_train_timesteps).to(transformer_dtype)

    with multi_module_hooks(transformer, hook_modules) as captured:
        transformer(
            hidden_states=noised, encoder_hidden_states=prompt_embeds,
            timestep=timestep, encoder_attention_mask=prompt_mask,
            num_frames=Fl, height=Hl, width=Wl,
            return_dict=False,
        )
    return {p: reshape_transformer_activation(a).float() for p, a in captured.items()}


@torch.no_grad()
def probe_ltx_d_in(
    hook_target, device, vae, transformer, hook_modules,
    prompt_embeds, prompt_mask, num_train_timesteps, shift,
):
    dummy = torch.randn(1, 3, 9, 128, 128, device=device, dtype=torch.float32).clamp_(-1, 1)
    if hook_target == "vae_encoder":
        results = process_ltx_vae_encoder(dummy, vae, hook_modules)
    else:
        results = process_ltx_transformer(
            dummy, vae, transformer, hook_modules, prompt_embeds, prompt_mask,
            num_train_timesteps, shift,
        )
    dims = {p: a.shape[-1] for p, a in results.items()}
    for p, d in dims.items():
        logger.info(f"Probed d_in={d} for hook '{p}'")
    return dims
