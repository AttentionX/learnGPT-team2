"""
running question: why do we need Dropout?
"""
import torch
from ..src.block_v4 import BlockVer4
from ..src.block_v3 import BlockVer3
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.multi_head_v1 import MultiHeadVer1
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train

# Dropout에 의해 랜덤으로 parameter 가 사용되지 않기 때문에
# 매 run 만다 결과가 달라진다.
def test_block_ver_4_output_is_always_different_in_train_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.train()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert not torch.allclose(out_1, out_2)
    assert not torch.allclose(out_2, out_3)
    assert not torch.allclose(out_3, out_4)

# eval 모드에서는 dropout 이 적용되지 않기 때문에 결과가 항상 같다.
def test_block_ver_4_output_is_always_the_same_in_eval_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.eval()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert torch.allclose(out_1, out_2)
    assert torch.allclose(out_2, out_3)
    assert torch.allclose(out_3, out_4)

# drop out 적용 이전에는 overfit이 발생하지만, 이후에는 발생하지 않는다는 것을 테스트
# FAILED
def test_dropout_helps():
    """
    dropout mitigates overfitting
    """
    torch.manual_seed(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    # push the model to overfit
    config['max_iters'] = 10000
    config['learning_rate'] = 0.01
    # --- BlockVer3: layers of multi-head + ffn + residual + layer norm --- #
    contextualizer = BlockVer3(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer4: layers of multi-head + ffn + residual + layer norm + dropout --- #
    contextualizer = BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    # "mitigates overfitting" = train loss is bigger but validation loss is smaller.
    assert losses_1['train'] < losses_2['train']
    assert losses_1['val'] > losses_2['val']




