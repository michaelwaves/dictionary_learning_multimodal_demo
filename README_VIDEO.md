# Video SAE Pipeline

```sh
uv sync
```

## Stage 1: Gather activations

Diffusion models (LTX, Wan):
```sh
# VAE latent mean
python gather_diffusion_activations.py \
    --model-type wan --hook-target vae_latent_mean \
    --video-dir /path/to/videos --full-video

# Transformer block hooks
python gather_diffusion_activations.py \
    --model-type ltx --hook-target transformer \
    --hook-modules transformer_blocks.14 \
    --video-dir /path/to/videos

# VAE encoder intermediate hooks
python gather_diffusion_activations.py \
    --model-type ltx --hook-target vae_encoder \
    --hook-modules encoder.mid_block \
    --video-dir /path/to/videos
```

Vision models (Qwen3-VL, VJEPA2):
```sh

      python gather_vl_activations.py --model-type vjepa --video-dir  /mnt/nw/home/m.yu/repos/multimodal_sae/videos_celebdf --layers 8,16,23

```

## Stage 2: Train SAE

```sh
python demo_video.py \
    --activation_dir activations/runs/<datetime>/vae_latent_mean \
    --architectures matryoshka_batch_top_k \
    --use_wandb
```

## Stage 3: Eval

```sh
python eval/visualize_features.py \
    --sae-path /path/to/sae.pt \
    --video-path /path/to/video.mp4 \
    --output-dir ./viz

python eval/top_samples.py \
    --sae-path /path/to/sae.pt \
    --video-dir /path/to/videos \
    --output-dir ./top_samples


python eval/top_samples.py  --sae-path /mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-06_08-18-08_lightricks/trainer_1/ae.pt --video-dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos_celebdf --selection active  --render-mode patch --topk 500


  python eval/top_samples.py --sae-path /mnt/nw/home/m.yu/repos/dictionary_learning_demo/video_saes/runs/2026-02-17_04-47-16_wan/resid_post_layer_all/trainer_1/ae.pt  --video-dir /mnt/nw/home/m.yu/repos/multimodal_sae/videos_celebdf  --vae-type wan  --vae-model Wan-AI/Wan2.2-TI2V-5B-Diffusers  --selection sparse --render-mode heatmap  --topk-features 500 --max-videos 10 --min-fire-count 20
```

## Utility scripts

- `recon_test.py` — VAE encode/decode sanity check
- `activations_test.py` — SVD analysis of layer activations
