"""
an implementation of layer normalization
"""
import torch


class LayerNorm(torch.nn.Module):
    """
    why do we need layer norm? - how should we test for this?
    - https://arxiv.org/pdf/1607.06450.pdf (Ba, Kiros & Hinton, 2016)
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # --- learnable parameters --- #
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO 3-1 --- #
        # normalize the row = feature vector 의 정규화
        # column (batch) = 입력 문장 내 정규화
        
        # batch 대신 layer 을 normalize 하는 이유
        # -> training / test 타임의 작동을 분리할 필요도 없고
        # 입력된 문장 길이에 상관없이 정규화를 적용할 수 있다.
        # -> decoder model 의 경우에는 문장을 생성하기 때문에 길이가 계속 변하기 때문에,
        # batch norm 을 적용하는 것은 어렵다

        # column (batch) 였으면 dim = 0
        xmean = x.mean(dim=-1, keepdim=True) # layer mean   (B, T, C) -> (B, T, 1)
        # std 를 사용하는 것과 sqrt(var) 을 사용하는 것 사이의 차이가 크다.
        # eps 를 그냥 사용하는 것과 sqrt 해서 사용하는 것의 차이라기에는 그 차이가 엄청 크다
        # std = x.std(dim=-1, keepdim=True) # layer variance
        # xhat = (x - xmean) / (std + self.eps) # normalized
        # return self.gamma * xhat + self.beta

        # 정답 예시
        std = x.std(-1, keepdim=True)  # (B, T, C) ->  (B, T, 1)
        return self.gamma * (x - xmean) / (std + self.eps) + self.beta
        # ---------------- #
