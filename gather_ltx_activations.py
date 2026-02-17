"""Stage 1: Gather video activations from LTX-Video and save to disk.

Usage:
    python gather_ltx_activations.py \
        --video_dir /path/to/videos \
        --hook_target transformer \
        --hook_modules transformer_blocks.14 transformer_blocks.20 \
        --num_frames 17 --device cuda:0

Hookable transformer paths: transformer_blocks.{0-27} (d=2048),
  transformer_blocks.{i}.attn1/attn2/ff, proj_in

Hookable vae_encoder paths: encoder.conv_in (d=128),
  encoder.down_blocks.{0-3}, encoder.mid_block (d=128-512)
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

from gather_utils import (
    METADATA_FILENAME,
    FramePrefetcher,
    ProgressTracker,
    ShardWriter,
    _DTYPE_MAP,
    make_run_dir,
    multi_module_hooks,
    preprocess_frames,
    remove_norm_outliers,
    reshape_transformer_activation,
    reshape_vae_activation,
    save_run_config,
)
from video_utils import scan_video_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GatherLTXConfig:
    model_name: str = "Lightricks/LTX-Video"
    hook_target: str = "transformer"
    hook_modules: list[str] = None
    video_dir: str = ""
    output_dir: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    num_frames: int = 17
    height: int = 256
    width: int = 256
    text_prompt: str = ""
    max_text_seq_len: int = 128
    num_train_timesteps: int = 1000
    shift: float = 1.0
    shard_size: int = 50
    max_norm_multiple: int = 10
    prefetch_workers: int = 4

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)

    def __post_init__(self):
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames must satisfy (n-1)%8==0, got {self.num_frames}")
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(f"height/width must be divisible by 32, got {self.height}x{self.width}")


def load_vae(model_name: str, device: str, enable_tiling: bool):
    from diffusers import AutoencoderKLLTXVideo
    logger.info(f"Loading LTX VAE from {model_name}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    if enable_tiling:
        vae.enable_tiling()
    return vae


def load_transformer(model_name: str, device: str, dtype: torch.dtype):
    from diffusers import LTXVideoTransformer3DModel
    logger.info(f"Loading LTX transformer from {model_name} ({dtype})...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    transformer.eval()
    return transformer


def compute_text_embeddings(model_name, text_prompt, max_seq_len, device, cast_dtype, free_after=True):
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
def process_video_vae(video_tensor, vae, hook_modules):
    with multi_module_hooks(vae, hook_modules) as captured:
        vae.encode(video_tensor)
    return {p: reshape_vae_activation(a).float() for p, a in captured.items()}


@torch.no_grad()
def process_video_transformer(
    video_tensor, vae, transformer, hook_modules,
    prompt_embeds, prompt_mask, num_train_timesteps, shift,
    latent_num_frames, latent_height, latent_width,
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
            num_frames=latent_num_frames, height=latent_height, width=latent_width,
            return_dict=False,
        )
    return {p: reshape_transformer_activation(a).float() for p, a in captured.items()}


@torch.no_grad()
def probe_d_in(config, vae, transformer, prompt_embeds, prompt_mask):
    dummy = torch.randn(1, 3, 9, 128, 128, device=config.device, dtype=torch.float32).clamp_(-1, 1)
    if config.hook_target == "vae_encoder":
        results = process_video_vae(dummy, vae, config.hook_modules)
    else:
        results = process_video_transformer(
            dummy, vae, transformer, config.hook_modules, prompt_embeds, prompt_mask,
            config.num_train_timesteps, config.shift, 2, 4, 4,
        )
    dims = {p: a.shape[-1] for p, a in results.items()}
    for p, d in dims.items():
        logger.info(f"Probed d_in={d} for hook '{p}'")
    return dims


def _module_dir_name(module_path: str) -> str:
    return module_path.replace(".", "_")


def gather_activations(config: GatherLTXConfig):
    video_paths = scan_video_directory(config.video_dir)
    if not video_paths:
        raise RuntimeError(f"No video files found in {config.video_dir}")
    logger.info(f"Found {len(video_paths)} videos")

    if not config.output_dir:
        config.output_dir = make_run_dir()
    os.makedirs(config.output_dir, exist_ok=True)
    save_run_config(config.output_dir, asdict(config))

    progress = ProgressTracker(config.output_dir)
    remaining = [p for p in video_paths if not progress.is_done(p)]
    logger.info(f"Already processed: {len(video_paths) - len(remaining)}, remaining: {len(remaining)}")
    if not remaining:
        return

    enable_tiling = config.hook_target != "vae_encoder"
    vae = load_vae(config.model_name, config.device, enable_tiling)
    transformer, prompt_embeds, prompt_mask = None, None, None

    if config.hook_target == "transformer":
        transformer = load_transformer(config.model_name, config.device, config.torch_dtype)
        prompt_embeds, prompt_mask = compute_text_embeddings(
            config.model_name, config.text_prompt, config.max_text_seq_len,
            config.device, next(transformer.parameters()).dtype, free_after=True,
        )
        latent_num_frames = (config.num_frames - 1) // 8 + 1
        latent_height = config.height // 32
        latent_width = config.width // 32

    d_in_map = probe_d_in(config, vae, transformer, prompt_embeds, prompt_mask)
    shard_writers = {
        p: ShardWriter(os.path.join(config.output_dir, _module_dir_name(p)))
        for p in config.hook_modules
    }
    prefetcher = FramePrefetcher(remaining, config.num_frames, config.prefetch_workers)
    videos_since_flush = 0

    for idx, video_path in enumerate(tqdm(remaining, desc="Gathering LTX activations")):
        frames = prefetcher.get_frames(idx)
        if frames is None:
            progress.mark_done(video_path)
            continue
        try:
            video_tensor = preprocess_frames(frames, config.height, config.width, config.device)
        except Exception:
            progress.mark_done(video_path)
            continue
        try:
            if config.hook_target == "vae_encoder":
                activations = process_video_vae(video_tensor, vae, config.hook_modules)
            else:
                activations = process_video_transformer(
                    video_tensor, vae, transformer, config.hook_modules,
                    prompt_embeds, prompt_mask, config.num_train_timesteps, config.shift,
                    latent_num_frames, latent_height, latent_width,
                )
        except Exception as e:
            logger.warning(f"Forward pass failed for {video_path}: {e}")
            progress.mark_done(video_path)
            continue

        for module_path in config.hook_modules:
            act = activations.get(module_path)
            if act is not None and act.numel() > 0:
                if config.max_norm_multiple > 0:
                    act = remove_norm_outliers(act, config.max_norm_multiple)
                if act.numel() > 0:
                    shard_writers[module_path].append(act)

        del activations, video_tensor
        torch.cuda.empty_cache()
        progress.mark_done(video_path)
        videos_since_flush += 1
        if videos_since_flush >= config.shard_size:
            for w in shard_writers.values():
                w.flush()
            progress.save()
            videos_since_flush = 0

    for w in shard_writers.values():
        w.flush()
    progress.save()

    for module_path, writer in shard_writers.items():
        metadata = {
            "model_name": config.model_name, "hook_target": config.hook_target,
            "hook_module": module_path, "d_model": d_in_map.get(module_path, -1),
            "num_frames": config.num_frames, "height": config.height, "width": config.width,
            "text_prompt": config.text_prompt, "shift": config.shift,
            "max_norm_multiple": config.max_norm_multiple,
            "total_tokens": writer.total_tokens, "num_shards": writer.shard_index,
            "num_videos": len(video_paths), "save_dtype": "float32",
        }
        with open(os.path.join(writer.module_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"{module_path}: {writer.total_tokens} tokens, {writer.shard_index} shards")

    prefetcher.shutdown()
    logger.info(f"Done. Output: {config.output_dir}")


def parse_args() -> GatherLTXConfig:
    parser = argparse.ArgumentParser(description="Gather LTX-Video activations")
    parser.add_argument("--model_name", default="Lightricks/LTX-Video")
    parser.add_argument("--hook_target", default="transformer", choices=["transformer", "vae_encoder"])
    parser.add_argument("--hook_modules", nargs="+", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--text_prompt", default="")
    parser.add_argument("--max_text_seq_len", type=int, default=128)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--max_norm_multiple", type=int, default=10)
    parser.add_argument("--prefetch_workers", type=int, default=4)
    return GatherLTXConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    gather_activations(parse_args())
