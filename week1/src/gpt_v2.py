from typing import Union
from . import GPTVer1, HeadVer1, HeadVer2, HeadVer3, HeadVer4
import torch
import torch.nn as nn


class GPTVer2(GPTVer1):

    def __init__(self, head: Union[HeadVer1, HeadVer2, HeadVer3, HeadVer4],
                 vocab_size: int, embed_size: int, block_size: int):
        super().__init__(vocab_size, block_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.head = head
        # keeps tensor for weights
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)  # (|V|, C) 65, 32(dimension)
        self.lm_head = nn.Linear(embed_size, vocab_size)  # (C, |V|)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        # --- TODO 2 --- #
        print("idx", idx.shape) # idx: 32,8
        # looks up embeddings for given 32, 8 and for each of them returns values with the corresponding dimension
        # Q: what is seq len?
        embedding = self.token_embedding_table(idx) # return embedding: 32, 8, 32

        # Attention head for past(average)
        past = self.head(embedding)
        # Return logits with input of 32 -> 65
        print("past: ",past.shape)
        logits = self.lm_head(past)
        print("logits: ", logits.shape)
        # And with these logits, our gpt model will sample chars with our distribution
        return logits