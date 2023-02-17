import torch


class HeadVer2:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: out (B, T, C)
        """
        # --- TODO 3 --- #
        # vectorize HeadVer1.__call__()
        B, T, C = x.shape 
        weight = torch.tril(torch.ones(T, T))
        weight = weight / weight.sum(1, keepdim=True)
        out = weight @ x
        # ------------ #
        return out
