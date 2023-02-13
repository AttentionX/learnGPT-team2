import torch
import timeit
from ..src import HeadVer1, HeadVer3

# past average 의 softmax(negative infinite future masking) 형태 구현 테스트
# 왜 negative infinite 로 masking을 할까?
# -> softmax 는 분자가 e^x 인데, x = -inf 이면 분자가 0이 되어 weight 가 0이 되기 때문에


# 값이 같은지 확인
# past average = softmax(negarive infinite future mask) @ input
def test_head_v3_logically_the_same_as_head_v1():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head_v1 = HeadVer1()
    head_v3 = HeadVer3()
    y_v1 = head_v1(x)
    y_v3 = head_v3(x)
    assert torch.allclose(y_v1, y_v3)

# 행렬 곱셈을 통해 구현해 속도가 더 빠른지 확인
def test_head_v3_faster_than_head_v1():
    x = torch.rand(4, 128, 1024)
    # head_v1 은 2중 for 문으로 느리지만,
    head_v1 = HeadVer1()
    # head_v3 은 행렬 곱셈으로 빠르다.
    head_v3 = HeadVer3()
    time_taken_v1 = timeit.timeit(lambda: head_v1(x), number=10)
    time_taken_v3 = timeit.timeit(lambda: head_v3(x), number=10)
    assert time_taken_v3 < time_taken_v1

# weight 의 합이 1인지 확인 (softmax 를 통해 구현했는지 확인)
def test_head_v3_logits_are_properly_masked():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer3()
    head(x) # (B, T, C)
    expected = torch.ones(B, T)
    # C 차원에 sum, C는 softmax의 결과이므로 합은 1
    was = head.wei.sum(dim=-1) # (B, T, C) -> (B, T)
    assert torch.allclose(expected, was)


# future masking이 제대로 되어있는지 확인
def test_head_v3_logits_are_properly_normalized():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head = HeadVer3()
    head(x)
    expected = torch.IntTensor([[[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)
