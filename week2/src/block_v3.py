"""
adding LayerNorm to the block
"""
import torch
from typing import Union
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2
from .block_v2 import BlockVer2
from .layer_normalization import LayerNorm


class BlockVer3(BlockVer2):

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int):
        super().__init__(head, embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO 3-2 --- #
        # transformer 의 초기 버전에서는 Layer Normalization 이 layer 뒤에 있었지만
        # 최신 모델은 앞에 붙는다.
        # -> 
        x = x + self.head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        # ---------------- #
        return x
