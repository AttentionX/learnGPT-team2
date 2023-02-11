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
        wei = ...
        self.wei = wei.detach()
        out = ...
        raise NotImplementedError
        # -------------- #
        return out