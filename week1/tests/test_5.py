import torch
from ..src import HeadVer4, GPTVer2
from .test_utils import config, train, generate

# 
def test_head_v4_attention_has_no_notion_of_space():
    """
    :return:
    """
    # 단어 3개로 구성된 문장인 x1, x2
    # x1 = abc, x2 = bac
    x1 = torch.Tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
    x2 = torch.Tensor([[[4, 5, 6],
                        [1, 2, 3],
                        [7, 8, 9]]])
    _, T, C = x1.shape # (1, 3, 3)
    head = HeadVer4(T, C)
    # 문장의 self-attention 의 결과는?
    # y1 = softmax( Q(a, b, c) @ K(a, b, c) ) @ V(a, b, c)
    #    [[a-a, a-b, a-c], [b-a, b-b, b-c], [c-a, c-b, c-c]] 간의 유사도 @ V(a, b, c)
    # y1 의 마지막 단어는 a * (c-a 유사도) + b * (c - b 유사도) + c * (c-c 유사도)
    y1 = head(x1)  # (B, T, C)
    # y2 = softmax( Q(b, a, c) @ K(b, a, c) ) @ V(b, a, c)
    #    [[b-a, b-b, b-c], [a-b, a-a, a-c], [c-b, c-a, c-c]] 간의 유사도 @ V(b, a, c)
    # y2의 마지막 단어는 b * (c - b 유사도) + a * (c-a 유사도) + c * (c-c 유사도)
    # 결국 같다
    # 즉, self-attention은 단어간의 유사도 * 단어 의 구조기 때문에 위치 정보가 고려되지 않는다.
    y2 = head(x2)  # (B, T, C)
    # 마지막 단어의 결과를 비교
    assert torch.allclose(y1[:, -1, :], y2[:, -1, :])


# future masking 이 제대로 되었는지 확인
def test_head_v4_logits_are_properly_masked():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    T, C = x.shape[1:] # (3, 3)
    head = HeadVer4(T, C)
    head(x)
    expected = torch.IntTensor([[[0,  1,  1],
                                 [0,  0,  1],
                                 [0,  0,  0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)


# softmax를 사용해 구현해 softmax (Q @ V) 의 합이 1인지 확인
def test_head_v4_logits_are_properly_normalized():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer4(T, C)
    head(x) # (B, T, C)
    expected = torch.ones(B, T)
    # C에 걸쳐 분포되어있는 weight 를 합치면 1이 되는지 확인
    was = head.wei.sum(dim=-1)  # (B, T)
    assert torch.allclose(expected, was)


# Q @ K / sqrt(C) 
# scale 값이 GPT3 API 의 hyperparameter 'temperature' 와 같은 역할을 할 수 있는 것 같다고 함
def test_head_v4_the_variance_of_wei_after_scale_is_1():
    B, T, C = 4, 128, 1024
    x = torch.randn(B, T, C)
    head = HeadVer4(T, C)
    head(x, test=True)  # (B, T, C)
    assert 1 == torch.round(head.var)


def test_gpt_v2_and_head_v4_generates_text_given_a_context():
    torch.manual_seed(1337)
    head = HeadVer4(config['block_size'], config['embed_size'])
    lm = GPTVer2(head, config['vocab_size'], config['embed_size'], config['block_size'])
    train(lm)  # may take a while
    expected = "The quick brown fox jumps over the lazyor th manot utou s l spaif ant"
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    assert expected == was
