import torch
import numpy as np

from transformers import AutoVideoProcessor, AutoModelForVideoClassification
from accelerate import Accelerator
from video_utils import read_video_pyav


device = Accelerator().device

hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

video_url = "sample_videos/jam.mp4"

frames = read_video_pyav(video_url, 8)
inputs = processor(frames, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names")
topk = logits.topk(5).indices[0]
topk_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for i, prob in zip(topk, topk_probs):
    text_label = model.config.id2label[i.item()]
    print(f"[{i}]-{text_label}: {prob:.2f}")
