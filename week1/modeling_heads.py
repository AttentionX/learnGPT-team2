import torch


class HeadVer1(torch.nn.Module):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (B, T, C)
        :return: out (B, T, C)
        """
        B, T, C = x.shape
        # --- TODO --- #
        out = ...
        # ------------ #
        return out


class HeadVer2(torch.nn.Module):
    """

    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return:
        """
        # ---- TODO --- #
        out = ...
        # ------------ #
        return out


class HeadVer3(torch.nn.Module):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO --- #
        out = ...
        # ------------ #
        return out


class HeadVer4(torch.nn.Module):
    """ i.e. one head of self-attention """
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.query = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.value = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.var = None
        self.wei = None

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        :param x: (B, T, C)
        :param debug: for testing purposes
        :return: (B, T, C)
        """
        B, T, C = x.shape
        if debug:
            k = torch.randn(B, T, C)  # (B, T, C)
            q = torch.randn(B, T, C)  # (B, T, C)
            v = torch.randn(B, T, C)  # (B, T, C)
        else:
            k = self.key(x)  # (B, T, C)
            q = self.query(x)  # (B, T, C)
            v = self.value(x)  # (B, T, C)
        # --- TODO --- #
        out = ...
        # ------------ #
        return out
