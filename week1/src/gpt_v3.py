import torch
from . import GPTVer2


class GPTVer3(GPTVer2):

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        B, T = idx.shape
        C = self.token_embedding_table.weight.shape[1]
        # --- TODO 6 --- #
        tok_emb = self.token_embedding_table(idx) # (B,T) -> (B, T, V) -> (B, T, C)
        pos_emb = self.pos_encodings(T, C) # (T, C)
        # pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1)
        x = self.head(tok_emb + pos_emb)
        logits = self.lm_head(x) # (B, T, C) -> (B, T ,V)
        # ------------- #
        return logits

    @staticmethod
    def pos_encodings(block_size: int, embed_size: int) -> torch.Tensor:
        """
        :param block_size: length of the sequence (T)
        :param embed_size: number of embedding dimensions (C)
        :return: (L, H)
        """
        # --- TODO 6 --- #
        # this is the original implementation
        encodings = torch.zeros(block_size, embed_size) # (T, C)
        pos = torch.arange(0, block_size) # (T)
        pos = pos.float().unsqueeze(dim=1) # (1, T)
        _2i = torch.arange(0, embed_size, step=2).float()
        encodings[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        encodings[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

        # -------------- #
        return encodings
