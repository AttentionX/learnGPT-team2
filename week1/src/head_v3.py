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

        # Softmax: e ^ x / (e ^ max(x) * sum(e ^ x / e ^ max(x)))
        # Weight must be negative infinity to ensure the output is 0 and therefore masked
        _, T, _ = x.shape
        tril = torch.tril(torch.ones(T, T))
        wei = torch.zeros((T, T))

        # Negative infinity instead of zeros for masked parts
        wei = wei.masked_fill(tril == 0, float('-inf'))
        # Now add softmax
        wei = torch.softmax(wei, dim=-1)
        out = wei @ x

        # Computations will come to a stop
        self.wei = wei.detach()
        # -------------- #
        return out