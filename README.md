# learnGPT - team 2

## week 1

### `test_2.py`? (`HeadVer1`)

> taking the past into account - why? 

- Compare: `GPTVer1`-generated completion vs `GPTVer2`-generated completion
- What difference do you notice?
- Why is there a difference?

### `test_3.py`? (`HeadVer2`)

>  vectorizing for loops - why?

- How is `HeadVer2` logically the same as `HeadVer1`?
- Why is `HeadVer2` faster than `HeadVer1`?

### `test_4.py`? (`HeadVer3`)

> taking the past into account with masking & normalization - how?

- How is `HeadVer3` logically the same as `HeadVer1`?
- Why mask `wei` with `-inf`? Why not `0`?


## week 2 - `test_8.py` (Team 2)

```shell
pytest tests/test_8.py -s -vv
```

### test_ffn_helps

<img src='img/BlockVer1.png' width=250>

`BlockVer1`은 위에서 구현한 Multi-Head Attention과 FeedForward를 수행합니다.

> TODO 2-1: `FeedForward`를 구현해주세요.

> TODO 2-2: `BlockVer1.forward`을 구현해주세요. Multi-Head Attention을 통과한 뒤 FeedForward layer를 통과시키면 됩니다.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer2`와  `BlockVer1` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?


### test_residual_conn_helps_when_network_is_deep

<img src='img/BlockVer2.png' width=250>

`BlockVer2`는 BlockVer1에서 Residual connection을 추가한 Block입니다.

> TODO 2-3: `BlockVer2.forward`를 구현해주세요. Multi-Head Attention과 FeedForward에 대해 각각 residual Connection을 추가하면 됩니다.


테스트를 돌려보고 다음의 질문에 답해주세요.
1. `BlockVer1`과 `BlockVer2` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?





