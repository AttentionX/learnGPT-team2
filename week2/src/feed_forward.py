import torch.nn


class FeedForward(torch.nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        # --- TODO 2-1 --- #
        # 유튜브 영상과 다르게 구현했음
        # 유튜브 영상은 ReLU 뒤에 Linear를 붙이지 않음
        # -> 뒷 부분에서 residual 과 함께 추가함

        # 차원을 확장-축소해야 하지 않나?
        # -> 왜 해야 할까? 왜 축소->확장이 아니라 확장->축소일까?
        #   -> FFN 의 목적은 더 많은 정보를 capture 하는 것이지 중요한 정보만 남기는 것이 아님
        
        # attention 을 통과한 단어 간의 관계에 대해 더 많은 정보를 추출해야 하기 때문에 차원을 확장
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 4 * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_size, embed_size),
        )
        # ---------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        return self.net(x)
