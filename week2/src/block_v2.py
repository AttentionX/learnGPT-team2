import torch
from .block_v1 import BlockVer1


class BlockVer2(BlockVer1):
    """ Block with residual connection"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO 2-3 --- #
        residual = x
        head_output = self.head(x)
        head_output += x

        residual = head_output
        x = self.ffwd(head_output)
        x += residual
        # --------------- #
        return x
