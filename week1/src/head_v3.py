from typing import Optional
from torch.nn import functional as F
import torch


class HeadVer3:
    def __init__(self):
        self.wei: Optional[torch.Tensor] = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: out (B, T, C)
        """
        # --- TODO 4 --- #
        B, T, C = x.shape
        tril = torch.tril(torch.ones(T, T))
        weight = torch.zeros((T, T))
        weight = weight.masked_fill(tril == 0, float('-inf'))
        weight = torch.softmax(weight, dim=-1)
        out = weight @ x
        self.wei = weight.detach()
        # -------------- #
        return out