1. MultiHeadVer2와 BlockVer1 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

A. MultiHead를 통과시켜준 다음, feedforward를 해준 BlockVer1의 성능이 더 좋다. Multihead는 다른 attention head들의 output을 concatenate해주는데 이때 이 결과들을 취합하여 학습 가능한 상태로 다음 block에 전달하는 역할을 한다. 

```
Then the whole process become like training a "stacked ensemble learning" where each model get different weight. This is not the best analogy; but the purpose of FFN is to parameterize self-attention modules.
```

2. BlockVer1과 BlockVer2 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

A. Residual connection을 더해준 BlockVer2가 더 좋다. 그 이유는 모델이 크고 깊어질수록 propagation을 하는 과정에서 gradient들이 깊은 곳까지 전파되지 않고 vanish 할 수 있다. 따라서 연산을 해주기 전 값을 후의 값에 추가로 더해주며 정보 손실을 막을 수 있다.