"""Qwen3-VL model loading, visual token selection, and layer hooks."""

import logging
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


def load_qwen_model_and_processor(model_name: str, torch_dtype: torch.dtype, device: str, num_frames: int):
    logger.info(f"Loading {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    processor.video_processor.min_frames = num_frames
    processor.video_processor.max_frames = num_frames
    return model, processor


def build_chat_text(processor: AutoProcessor, prompt: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_video_inputs(processor, frames, chat_text):
    try:
        return processor(text=[chat_text], videos=[frames], return_tensors="pt", padding=True)
    except Exception as e:
        logger.debug(f"Processor failed: {e}")
        return None


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def select_visual_tokens(activations: torch.Tensor, input_ids: torch.Tensor, video_pad_token_id: int):
    return activations[input_ids == video_pad_token_id]


def run_qwen_forward(
    model: torch.nn.Module,
    processor: AutoProcessor,
    frames: list[np.ndarray],
    prompt: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    chat_text = build_chat_text(processor, prompt)
    inputs = prepare_video_inputs(processor, frames, chat_text)
    if inputs is None:
        return {}
    device = next(model.parameters()).device
    inputs = move_inputs_to_device(inputs, device)
    input_ids = inputs["input_ids"]
    video_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    with multi_layer_hooks(model, layers) as captured:
        with torch.no_grad():
            model(**inputs)

    result = {}
    for layer_idx in layers:
        if layer_idx in captured:
            act = select_visual_tokens(captured[layer_idx], input_ids, video_pad_token_id)
            result[layer_idx] = act.float()
    return result


@contextmanager
def multi_layer_hooks(model: torch.nn.Module, layer_indices: list[int]):
    captured = {}
    handles = []
    for layer_idx in layer_indices:
        submodule = model.model.language_model.layers[layer_idx]

        def _make_hook(idx):
            def hook_fn(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[idx] = tensor.detach()
            return hook_fn

        handles.append(submodule.register_forward_hook(_make_hook(layer_idx)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
