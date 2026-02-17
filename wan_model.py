"""Wan2.2 model loading and forward pass for activation gathering."""

import logging

import torch

from gather_utils import multi_module_hooks, reshape_transformer_activation

logger = logging.getLogger(__name__)


def load_wan_vae(model_name: str, device: str):
    from diffusers import AutoencoderKLWan
    logger.info(f"Loading Wan VAE from {model_name} (float32)...")
    vae = AutoencoderKLWan.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    return vae


def load_wan_transformer(model_name: str, device: str, dtype: torch.dtype):
    from diffusers import WanTransformer3DModel
    logger.info(f"Loading Wan transformer from {model_name} ({dtype})...")
    transformer = WanTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    transformer.eval()
    return transformer


def compute_wan_text_embeddings(
    model_name: str, text_prompt: str, max_seq_len: int,
    device: str, cast_dtype: torch.dtype, free_after: bool = True,
):
    from transformers import AutoTokenizer, UMT5EncoderModel
    logger.info("Loading UMT5 text encoder...")
    text_encoder = UMT5EncoderModel.from_pretrained(
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
        logger.info("Freed UMT5 text encoder")
    return prompt_embeds, attention_mask


def build_ti2v_input(
    noised_latents: torch.Tensor, image_latent: torch.Tensor,
) -> torch.Tensor:
    """Build 48-channel TI2V input: [noised(16ch), image(16ch), mask(16ch)].

    image_latent is the first frame's VAE encoding, broadcast across time.
    mask is 1 for the first temporal position, 0 for the rest.
    """
    B, C, T, H, W = noised_latents.shape
    image_broadcast = image_latent.expand(B, C, T, H, W)
    mask = torch.zeros(B, C, T, H, W, device=noised_latents.device, dtype=noised_latents.dtype)
    mask[:, :, 0, :, :] = 1.0
    return torch.cat([noised_latents, image_broadcast, mask], dim=1)


@torch.no_grad()
def process_wan_transformer(
    video_tensor: torch.Tensor,
    vae: torch.nn.Module,
    transformer: torch.nn.Module,
    hook_modules: list[str],
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_train_timesteps: int,
) -> dict[str, torch.Tensor]:
    """VAE encode -> noise -> build TI2V input -> transformer forward with hooks."""
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

    image_latent = latents[:, :, :1, :, :]

    sigma = torch.rand(1, device=device, dtype=latents.dtype)
    noised = ((1.0 - sigma) * latents + sigma * torch.randn_like(latents)).to(transformer_dtype)
    timestep = (sigma * num_train_timesteps).to(transformer_dtype)

    ti2v_input = build_ti2v_input(noised, image_latent.to(transformer_dtype))

    with multi_module_hooks(transformer, hook_modules) as captured:
        transformer(
            hidden_states=ti2v_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            return_dict=False,
        )

    return {p: reshape_transformer_activation(a).float() for p, a in captured.items()}


@torch.no_grad()
def probe_wan_d_in(
    device: str, vae, transformer, hook_modules, prompt_embeds, prompt_mask, num_train_timesteps,
):
    dummy = torch.randn(1, 3, 5, 128, 128, device=device, dtype=torch.float32).clamp_(-1, 1)
    results = process_wan_transformer(
        dummy, vae, transformer, hook_modules, prompt_embeds, prompt_mask, num_train_timesteps,
    )
    dims = {p: a.shape[-1] for p, a in results.items()}
    for p, d in dims.items():
        logger.info(f"Probed d_in={d} for hook '{p}'")
    return dims
