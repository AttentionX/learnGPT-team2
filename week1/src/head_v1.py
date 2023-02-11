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

        # normalized past masking @ input 은 average of the past 와 같다
        # 근데 normalized past masking 은 softmax(negative infinite future masking) 이다
        # 따라서 past average = softmax(negarive infinite future mask) @ input 이다.
        # 이게 무슨 뜻일까?
        
        # self attention은 input_query 와 input_key 사이의 dot product 를 구해서
        # input_value 를 weighted sum 한다
        # 이 때 q-k dot product 를 weight 라고 하면
        # 이 weight 에 negative infinice future mask 를 한 뒤에 softmax 한 값과
        # value-input 을 곱해야 한다.

        # 즉 이 head-1 은 그 self-attention을 weight이 동일하게 다 1이라고 가정한 상태에서 구현한 셈이다

        # (B, T, T) @ (B, T, C) ---> (B, T, C) 대신에
        # 바로 (B, T, C) ---> (B, T ,C) 를 한다
        out = torch.zeros((B, T, C)).to(x.device)
        for b in range(B):
            for t in range(T):
                # batch 내에서, 0~t 까지를 잘라낸 뒤에
                xprev = x[b, :t + 1] # (t,C)
                # average 를 구해 저장한다
                out[b, t] = torch.mean(xprev, 0)
        # -------------- #
        return out
