"""VJEPA2 model loading and forward pass via HuggingFace Transformers."""

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

logger = logging.getLogger(__name__)


def load_vjepa_model_and_processor(
    model_name: str, torch_dtype: torch.dtype, device: str,
):
    logger.info(f"Loading VJEPA2 model: {model_name}")
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device, attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoVideoProcessor.from_pretrained(model_name)
    return model, processor


def run_vjepa_forward(
    model: torch.nn.Module,
    processor: AutoVideoProcessor,
    frames: list[np.ndarray],
    layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    video = np.stack(frames)
    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=True, output_hidden_states=True)

    d_model = outputs.last_hidden_state.shape[-1]
    result = {}
    for layer_idx in layers:
        act = outputs.hidden_states[layer_idx + 1].reshape(-1, d_model)
        result[layer_idx] = act.float()
    return result
