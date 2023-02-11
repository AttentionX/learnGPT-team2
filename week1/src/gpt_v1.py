from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTVer1(nn.Module):

    def __init__(self, vocab_size: int, block_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> \
            tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx and targets are both (B, T) tensor of integers
        logits = self.logits(idx)  # (B, T) ->  (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross_entropy 함수는 Channel 이 두번째 element에 있는 걸 요구하기 때문에 텐서 shape 을 바꾼다
            logits = logits.view(B * T, C)  # (B, T, C) ->  (B * T, C)
            targets = targets.view(B * T)  # (B, T) -> (B * T)
            loss = F.cross_entropy(logits, targets)  # (B * T, C), (B * T) -> scalar
        return logits, loss

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # --- TODO 1 --- #
            last_idx = idx[:, -self.block_size:] # (B, T + a) -> (B, T)
            logits, loss = self(last_idx) # (B, T, C)
            # see only last token
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            # -------------- #
        return idx