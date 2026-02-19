● All changes are done. Here's a summary:

  Changes made

  demo_config.py
  - num_tokens: 500M → 50M (matches available VJEPA2 data — ~1 epoch)
  - TARGET_L0s: [10, 20] → [20, 50] (reasonable sparsity for d=1024)

  eval/eval_vjepa2.py — full rewrite:
  - BatchTopKSAE.from_pretrained() instead of AutoEncoder.from_pretrained()
  - Dynamic patch grid computed from model.config.patch_size/tubelet_size and processor output — no more hardcoded
  T_P=234, H_P=W_P=16
  - Proper device handling — process on CPU, move tensors to device (matching vjepa_model.py pattern)
  - CLI via Click — --sae_path, --video_path, --layer, --export_dir, --feature_step, --device
  - Broader feature scan — every 50th feature across full dictionary (configurable via --feature_step)
  - Correct frame→patch temporal mapping using proportional indexing

  Training commands (one per GPU)

  # Layer 8
  CUDA_VISIBLE_DEVICES=0 python demo_video.py \
    --activation_dir activations/runs/2026-02-17_23-23-33/layer_8 \
    --architectures batch_top_k --device cuda:0 --use_wandb

  # Layer 16
  CUDA_VISIBLE_DEVICES=1 python demo_video.py \
    --activation_dir activations/runs/2026-02-17_23-23-33/layer_16 \
    --architectures batch_top_k --device cuda:0 --use_wandb

  # Layer 23
  CUDA_VISIBLE_DEVICES=2 python demo_video.py \
    --activation_dir activations/runs/2026-02-17_23-23-33/layer_23 \
    --architectures batch_top_k --device cuda:0 --use_wandb

  Eval command (after training)

  CUDA_VISIBLE_DEVICES=3 python eval/eval_vjepa2.py \
    --sae_path <path_to_trained_sae>/ae.pt \
    --video_path videos_celebdf/fake/id0_id1_0000.mp4 \
    --layer 16