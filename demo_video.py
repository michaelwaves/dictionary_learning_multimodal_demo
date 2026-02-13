"""
Stage 2: Train a Matryoshka BatchTopK SAE on pre-gathered video activations.

Reads sharded activation files from disk (produced by gather_video_activations.py)
and trains SAEs using the existing training infrastructure.

Usage:
    python demo_video.py \
        --activation_dir ./activations/layer_18 \
        --save_dir video_saes \
        --architectures matryoshka_batch_top_k \
        --device cuda:0 \
        --use_wandb
"""

import os
import json
import argparse
import random
import time

import torch as t
import torch.multiprocessing as mp

import demo_config
from disk_buffer import DiskActivationBuffer
from dictionary_learning.dictionary_learning.training import trainSAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAE on pre-gathered video activations"
    )
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="Directory with sharded activations and metadata.json")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to save trained SAEs")
    parser.add_argument("--architectures", type=str, nargs="+", required=True,
                        choices=[e.value for e in demo_config.TrainerType],
                        help="SAE architectures to train")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--num_tokens", type=int, default=None,
                        help="Override training token count (default: from demo_config)")
    parser.add_argument("--shards_in_memory", type=int, default=4,
                        help="Number of shards to keep in GPU memory")
    return parser.parse_args()


def load_activation_metadata(activation_dir: str) -> dict:
    metadata_path = os.path.join(activation_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


def compute_training_steps(num_tokens: int, sae_batch_size: int) -> int:
    return int(num_tokens / sae_batch_size)


def compute_checkpoint_steps(total_steps: int) -> list[int]:
    """Log-spaced checkpoints from 0.1% to 100% of training."""
    desired_fractions = [0.0] + t.logspace(-3, 0, 7).tolist()[:-1]
    return sorted(int(total_steps * frac) for frac in desired_fractions)


def run_video_sae_training(
    activation_dir: str,
    save_dir: str,
    architectures: list[str],
    device: str,
    num_tokens: int,
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    shards_in_memory: int = 4,
):
    metadata = load_activation_metadata(activation_dir)
    activation_dim = metadata["d_model"]
    model_name = metadata["model_name"]
    layer = metadata.get("layer", metadata.get("hook_module", "unknown"))
    submodule_name = metadata.get("submodule_name", f"resid_post_layer_{layer}")

    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    steps = compute_training_steps(num_tokens, sae_batch_size)
    log_steps = 100

    save_steps = compute_checkpoint_steps(steps) if save_checkpoints else None

    print(f"Training config: {activation_dim=}, {layer=}, {steps=}, {sae_batch_size=}")
    print(f"Activation source: {metadata['total_tokens']} tokens from {metadata['num_shards']} shards")

    activation_buffer = DiskActivationBuffer(
        activation_dir=activation_dir,
        out_batch_size=sae_batch_size,
        device=device,
        dtype=t.bfloat16,
        shards_in_memory=shards_in_memory,
    )

    trainer_configs = demo_config.get_trainer_configs(
        architectures=architectures,
        learning_rates=demo_config.learning_rates,
        seeds=demo_config.random_seeds,
        activation_dim=activation_dim,
        dict_sizes=demo_config.dictionary_widths,
        model_name=model_name,
        device=device,
        layer=layer,
        submodule_name=submodule_name,
        steps=steps,
    )

    print(f"Training {len(trainer_configs)} SAE(s)")
    assert len(trainer_configs) > 0
    full_save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=full_save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.wandb_project,
            normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
            backup_steps=1000,
        )


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    num_tokens = args.num_tokens if args.num_tokens else demo_config.num_tokens

    start_time = time.time()

    run_video_sae_training(
        activation_dir=args.activation_dir,
        save_dir=args.save_dir,
        architectures=args.architectures,
        device=args.device,
        num_tokens=num_tokens,
        dry_run=args.dry_run,
        use_wandb=args.use_wandb,
        save_checkpoints=args.save_checkpoints,
        shards_in_memory=args.shards_in_memory,
    )

    print(f"Total time: {time.time() - start_time:.1f}s")
