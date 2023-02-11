import torch


class HeadVer1:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (B, T, C)
        :return: out (B, T, C)
        """
        B, T, C = x.shape
        # --- TODO 2 --- #
        # use nested for loops to take an average of the past into account
        out = torch.zeros((B,T,C))
        for idx_b in range(B):
            for idx_t in range(T):
                past_x = x[idx_b, :idx_t + 1]
                out[idx_b, idx_t] = torch.mean(past_x, 0)
        return out
