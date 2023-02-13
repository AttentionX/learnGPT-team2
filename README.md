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
  
Softmax(negative infinite future masking) 은 past average 와 같다.  
e^(-inf) = 0 이고, softmax의 결과의 합은 1이기 때문

- Why mask `wei` with `-inf`? Why not `0`?  
  
e^(-inf) = 0 이기 때문이다.  
e^(0) = 1 이기 때문에 softmax 연산 이후에도 값이 남게 되어 미래 토큰과 communitaction 이 생기기 때문

## week 2

...

## Contributors



