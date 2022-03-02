import torch
import torch.nn as nn
from einops import rearrange


class DangerPLoss(nn.Module):
    def __init__(self, max_danger_lvl: int):
        super().__init__()
        self.max_danger_lvl = max_danger_lvl
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: The way the dual masks are handled is a mess. Find a better solution.
        cls_preds = rearrange(preds[0], "b c w h -> b w h c")
        danger_preds = rearrange(preds[1], "b c w h -> b w h c")

        oh_cls_labels = labels[..., 0]
        oh_danger_labels = labels[..., 1]
        oh_danger_labels = oh_danger_labels[..., :self.max_danger_lvl]  # Remove padding

        cls_loss = self.loss_fn(cls_preds, oh_cls_labels)
        danger_loss = self.loss_fn(danger_preds, oh_danger_labels)

        return cls_loss + danger_loss
