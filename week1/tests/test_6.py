"""
check if ver_1, ver_2, ver_3 preserves order.
"""
import torch
from .test_utils import config, train, generate
from ..src import HeadVer1, HeadVer4, GPTVer1, GPTVer2, GPTVer3

# 같은 단어 "7"에 대해 logit이 같은 결과가 나온다.
def test_gpt_v1_logits_order_is_not_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v2_logits_order_is_not_preserved():
    torch.manual_seed(1337)
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer2(HeadVer1(), V, C, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])

# PE 는 각 값이 달라야 하고
def test_gpt_v3_pos_encodings_each_pos_is_different():
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    assert not torch.allclose(encodings[2], encodings[3])


# PE 는 두 위치의 차이가 거리를 의미하기 때문에,
# 같은 거리에 있는 두 단어의 PE 차이는 같아야 한다
def test_gpt_v3_pos_encodings_dist_stays_constant():
    T, C = 10, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))
    assert torch.allclose(torch.norm(encodings[5] - encodings[3]), torch.norm(encodings[6] - encodings[4]))
    assert torch.allclose(torch.norm(encodings[7] - encodings[5]), torch.norm(encodings[8] - encodings[6]))


# PE 를 적용하면 같은 단어 7이더라도 위치에 따라 다른 결과가 나온다.
# 사실 이건 Absolute PE 의 문제점이다. absolute 한 단어의 위치에 따라 logit이 달라지는 것이기 때문.
# 만약 relative PE를 적용한다면? PE 가 input에 더해지는 대신에, query와 key 사이에 적용된다.
# 그럴 경우에, 주변 단어와의 관계만 고려하기 때문에 [7,7,7,7] 에서 중간 7 두개는 같은 결과가 나올 수도 있을 것이다
# TODO:  Alibi PE, relative PE, Rotary PE 셋 다 구현해서 적용하고 결과를 비교해보자.
# week2 에서 Absolute PE 를 learnable Embedding 으로 구현할 예정이라고 하니, 이 때 구현을 시도해보자.
def test_gpt_v3_logits_order_is_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer1(), V, C, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert not torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert not torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert not torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_and_head_v4_generates_text_given_a_context():
    torch.manual_seed(1337)
    head = HeadVer4(config['block_size'], config['embed_size'])
    lm = GPTVer3(head, config['vocab_size'], config['embed_size'], config['block_size'])
    train(lm)  # may take a while
    expected = "The quick brown fox jumps over the lazy stt, manot utou st the if ant"
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    assert expected == was

