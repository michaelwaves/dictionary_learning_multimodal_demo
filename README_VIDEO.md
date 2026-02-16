
 # Prerequisites

 uv installed on system, GPU with at least 40GB VRAM
```sh
#install dependencies
uv sync
```

  # Stage 1: Gather activations (run once)

- gather_video_activations.py: Qwen/ViTs

Ltx/DiTs
- gather_ltx_activations.py: more complex script that lets you choose which encoder block(s) to hook
- gather_ltx_encoder_activations.py: simpler script that just gets activations from full encoder, sampling from latent distribution with vae.encode(tensor).latent_dist.mean 
```sh

  python gather_video_activations.py \
      --model_name "Qwen/Qwen3-VL-8B-Instruct" \
      --video_dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos/ \
      --output_dir ./activations \
      --layers 12 18 24 \
      --num_frames 16 \
      --device cuda:0

  python gather_ltx_activations.py  --video_dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos  --output_dir ./ltx_activations_test --hook_target vae_encoder --hook_modules  encoder.mid_block  --num_frames 25

python gather_ltx_activations.py  --video_dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos  --output_dir ./ltx_activations --hook_target vae_encoder --hook_modules  encoder.down_blocks.2.resnets.2  --num_frames 321

python gather_ltx_activations.py  --video_dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos  --output_dir ./ltx_activations --hook_target vae_encoder --hook_modules  encoder.down_blocks.0.resnets.3  --num_frames 321

python gather_ltx_encoder_activations.py  --video-dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos  --output-dir ./ltx_activations_vae  --num-frames 321


  python gather_ltx_encoder_activations.py \
      --video-dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos \
      --output-dir ./ltx_activations_vae_full \
      --num-frames 33 \
      --full-video \
      --max-videos 100

```

  # Stage 2: Train SAE (repeatable with different configs)

  demo_video.py works for all architectures lightricks and qwen

  ```sh
  python demo_video.py \
      --activation_dir ./activations/layer_18 \
      --save_dir video_saes \
      --architectures matryoshka_batch_top_k \
      --device cuda:0 \
      --use_wandb


 python demo_video.py --activation_dir ./ltx_activations/encoder-down_blocks-2-resnets-2 --save_dir video_saes  --architectures matryoshka_batch_top_k  --device cuda:0   --num_tokens 500000000  --shards_in_memory 4   --use_wandb   --save_checkpoints

 python demo_video.py --activation_dir ./ltx_activations/encoder_down_blocks_0_resnets_3 --save_dir video_saes  --architectures matryoshka_batch_top_k  --device cuda:0   --num_tokens 72000000  --shards_in_memory 4   --use_wandb   --save_checkpoints

 
 python demo_video.py --activation_dir ./ltx_activations_vae --save_dir video_saes  --architectures matryoshka_batch_top_k  --device cuda:0   --num_tokens 72000000  --shards_in_memory 4   --use_wandb   --save_checkpoints
```

 
 # Stage 3: Eval

```sh
 python eval/visualize_features.py --sae-path video_saes/resid_post_layer_encoder.down_blocks.2.resnets.2/trainer_3/checkpoints/ae_11117.pt --video-path sample_videos/jam.mp4 --output-dir ./eval/output --hook-module encoder.down_blocks.2.resnets.2 --topk 20 --min-count 10 --max-count 200  --display-frames 8 --sampling spaced --start 0.2

  python eval/visualize_features.py --sae-path video_saes/resid_post_layer_unknown --video-path sample_videos/jam.mp4 --output-dir ./eval/output/encoder  --topk 20 --min-count 10 --max-count 200  --display-frames 8 --sampling spaced --start 0.2

 ```

 # Utility scripts:

- recon_test.py: check if the vae encoding/decoding is working
- activations_test.py: analyze layer activations with singular value decomposition (SVD) to see which ones have variance that is not 99% explained by like top 10 features (to not make it way too easy for SAE).