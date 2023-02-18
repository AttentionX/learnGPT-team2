import torch


class HeadVer2:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: out (B, T, C)
        """
        # --- TODO 3 --- #
        # vectorize HeadVer1.__call__()
        # Softmax version of weights
        T= x.shape[1]

        # Generate weight matrix with 1s and return the lower triangle part
        # Rest will be 0
        weights = torch.tril(torch.ones(T, T))
        # Weights shape: 128 * 128, x: 4 * 128 * 1024
        print("weights: ", weights.shape)
        print("weights.sum: ", weights.sum(1, keepdim=True).shape)

        print(weights)
        print(weights.sum(1, keepdim=True))
        weights = weights / weights.sum(1, keepdim=True)
        print(weights)
        out = weights @ x
        return out
