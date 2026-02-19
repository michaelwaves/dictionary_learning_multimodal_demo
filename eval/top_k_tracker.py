"""Track top-k scoring chunks per SAE feature."""

from dataclasses import dataclass

import torch


@dataclass
class ChunkRef:
    score: float
    video_path: str
    start: int
    count: int


class TopKTracker:
    def __init__(self, num_features: int, k: int):
        self.scores = torch.full((num_features, k), -float("inf"))
        self.refs: list[list[ChunkRef | None]] = [
            [None] * k for _ in range(num_features)
        ]

    def update(self, mean_acts: torch.Tensor, path: str, start: int, count: int):
        min_score_values, min_score_indices = self.scores.min(dim=1)
        for i in (mean_acts > min_score_values).nonzero(as_tuple=True)[0].tolist():
            slot = min_score_indices[i].item()
            self.scores[i, slot] = mean_acts[i]
            self.refs[i][slot] = ChunkRef(
                mean_acts[i].item(), path, start, count,
            )
            breakpoint()

    def top_features(self, n: int) -> list[int]:
        totals = self.scores.clone()
        totals[totals == -float("inf")] = 0
        return totals.sum(dim=1).topk(min(n, self.scores.shape[0])).indices.tolist()

    def top_chunks(self, feature: int) -> list[ChunkRef]:
        return sorted(
            [r for r in self.refs[feature] if r], key=lambda r: -r.score,
        )
