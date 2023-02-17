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
        #          | 1 0 0 |                 | 0 -inf -inf |
        # tril  =  | 1 1 0 |  --> weight  =  | 0   0  -inf |
        #          | 1 1 1 |                 | 0   0    0  |
        weight = torch.softmax(weight, dim=-1)
        #  |  1   0   0  |
        #  | 1/2 1/2  0  |
        #  | 1/3 1/3 1/3 |
        out = weight @ x
        self.wei = weight.detach()
        # -------------- #
        return out