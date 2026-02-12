
  # Stage 1: Gather activations (run once)
  python gather_video_activations.py \
      --model_name "Qwen/Qwen3-VL-8B-Instruct" \
      --video_dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos/ \
      --output_dir ./activations \
      --layers 12 18 24 \
      --num_frames 16 \
      --device cuda:0

  # Stage 2: Train SAE (repeatable with different configs)
  python demo_video.py \
      --activation_dir ./activations/layer_18 \
      --save_dir video_saes \
      --architectures matryoshka_batch_top_k \
      --device cuda:0 \
      --use_wandb