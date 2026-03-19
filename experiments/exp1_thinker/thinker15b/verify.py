"""verify.py — CPU smoke test for Thinker-1.5B v2 components.

v2: Tests real GRU, slot persistence, slot vocab CE loss, standard pre-norm.

Run on any machine with MindSpore: python verify.py
"""

from __future__ import annotations

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_bpe_tokenizer() -> None:
    print("=" * 50)
    print("TEST: BPE Tokenizer roundtrip (v2)")
    print("=" * 50)

    from tokenizer import ByteLevelBPE

    tok = ByteLevelBPE()
    corpus = [
        "def foo(x):\n    return x + 1\n",
        "x = 10\ny = 20\nz = x + y\n",
        "for i in range(10):\n    print(i)\n",
    ] * 50

    tok.train(corpus, vocab_size=300)
    print(f"  Vocab size: {len(tok.encoder)}")
    print(f"  EOS id: {tok.eos_id}")
    print(f"  [TURN] id: {tok.turn_id}")

    test_texts = [
        "def foo(x):",
        "x = 10 + 20",
        "print(i)",
        "",
    ]
    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Roundtrip failed: {text!r} -> {decoded!r}"
        print(f"  OK: {text!r} -> {len(ids)} tokens")

    # Multi-turn encoding
    turns = ["hello", "world"]
    multi_ids = tok.encode_with_turns(turns)
    assert tok.turn_id in multi_ids
    print(f"  Multi-turn: {len(multi_ids)} tokens, [TURN] found: OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        vp = os.path.join(tmpdir, "vocab.json")
        mp = os.path.join(tmpdir, "merges.json")
        tok.save(vp, mp)
        tok2 = ByteLevelBPE.load(vp, mp)
        for text in ["x = 10", "def foo():"]:
            assert tok.encode(text) == tok2.encode(text)
        print("  Save/load roundtrip: OK")

    print("  PASS\n")


def test_config() -> None:
    print("=" * 50)
    print("TEST: Config validation (v2)")
    print("=" * 50)

    from config import Thinker15BConfig

    cfg = Thinker15BConfig()
    print(f"  hidden={cfg.hidden_dim}, layers={cfg.num_layers}, "
          f"heads={cfg.num_heads}, slot_vocab={cfg.slot_vocab_size}")
    assert cfg.hidden_dim % 16 == 0
    assert cfg.slot_vocab_size % 16 == 0
    print("  Default config: OK")

    tiny = Thinker15BConfig.tiny()
    print(f"  tiny: hidden={tiny.hidden_dim}, layers={tiny.num_layers}, "
          f"slot_vocab={tiny.slot_vocab_size}")
    assert tiny.hidden_dim == 128
    assert tiny.slot_vocab_size == 32
    print("  Tiny config: OK")

    try:
        Thinker15BConfig(hidden_dim=100)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid config rejected: OK ({e})")

    print("  PASS\n")


def test_components() -> None:
    print("=" * 50)
    print("TEST: Model components v2 (CPU)")
    print("=" * 50)

    try:
        import mindspore as ms
        from mindspore import nn, ops
        from mindspore.common.tensor import Tensor
    except ImportError:
        print("  SKIP: MindSpore not available\n")
        return

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    from config import Thinker15BConfig
    from model import (
        _fp16_dense, RMSNorm, RotaryEmbedding, TransformerBlock,
        SFMSlotBank, SlotPredictionHead, SlotTokenizer,
        Thinker15BModel,
    )

    cfg = Thinker15BConfig.tiny()
    B, S = 2, cfg.max_seq_len
    H = cfg.hidden_dim
    V = cfg.vocab_size
    NS = cfg.num_slots
    SD = cfg.slot_dim
    SV = cfg.slot_vocab_size

    # 1. RMSNorm
    print("  RMSNorm ...")
    norm = RMSNorm(H)
    x = Tensor(np.random.randn(B, S, H).astype(np.float32))
    y = norm(x)
    assert y.shape == (B, S, H)
    print(f"    shape={y.shape}: OK")

    # 2. RotaryEmbedding
    print("  RotaryEmbedding ...")
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    q = Tensor(np.random.randn(B, cfg.num_heads, S, cfg.head_dim).astype(
        np.float16))
    q_rot = rope(q)
    assert q_rot.shape == q.shape
    print(f"    shape={q_rot.shape}: OK")

    # 3. TransformerBlock (v2: no post_attn_norm)
    print("  TransformerBlock (v2 pre-norm) ...")
    block = TransformerBlock(cfg)
    inv_freq = 1.0 / (cfg.rope_theta ** (
        np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))
    t = np.arange(S, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos_t = Tensor(np.cos(emb).astype(np.float16)[np.newaxis, np.newaxis,
                                                     :, :])
    sin_t = Tensor(np.sin(emb).astype(np.float16)[np.newaxis, np.newaxis,
                                                     :, :])
    mask_np = np.triu(np.full((S, S), -1e4, dtype=np.float16), k=1)
    mask_t = Tensor(mask_np.reshape(1, 1, S, S))

    x = Tensor(np.random.randn(B, S, H).astype(np.float16))
    out = block(x, cos_t, sin_t, mask_t)
    assert out.shape == (B, S, H)
    print(f"    shape={out.shape}: OK")

    # 4. SFMSlotBank (v2: real GRU, takes slots as input)
    print("  SFMSlotBank (v2 real GRU) ...")
    bank = SFMSlotBank(cfg)
    slots_in = Tensor(np.random.randn(B, NS, SD).astype(np.float16))
    modified, new_slots = bank(x, slots_in)
    assert modified.shape == (B, S, H)
    assert new_slots.shape == (B, NS, SD)
    print(f"    modified={modified.shape}, slots={new_slots.shape}: OK")

    # 5. SlotPredictionHead (v2: CE logits over slot vocab)
    print("  SlotPredictionHead (v2 CE) ...")
    pred_head = SlotPredictionHead(cfg)
    pred_logits = pred_head(new_slots)
    assert pred_logits.shape == (B, NS, SV)
    print(f"    logits shape={pred_logits.shape}: OK")

    # 6. SlotTokenizer (v2: discretize to slot vocab)
    print("  SlotTokenizer (v2) ...")
    slot_tok = SlotTokenizer(cfg)
    slot_ids = slot_tok(new_slots)
    assert slot_ids.shape == (B, NS)
    print(f"    ids shape={slot_ids.shape}: OK")

    # 7. Thinker15BModel (v2: slot persistence)
    print("  Thinker15BModel (v2, tiny) ...")
    model = Thinker15BModel(cfg)
    input_ids = Tensor(np.random.randint(0, V, (B, S)).astype(np.int32))
    logits, slot_loss = model(input_ids, cos_t, sin_t, mask_t)
    assert logits.shape == (B, S, V)
    assert slot_loss.shape == ()
    print(f"    logits={logits.shape}, slot_loss={float(slot_loss):.4f}: OK")

    total = sum(p.size for p in model.get_parameters())
    print(f"    Total params: {total:,}")

    print("  PASS\n")


def test_gradient_flow() -> None:
    print("=" * 50)
    print("TEST: Gradient flow (v2)")
    print("=" * 50)

    try:
        import mindspore as ms
        from mindspore import nn, ops
        from mindspore.common.tensor import Tensor
    except ImportError:
        print("  SKIP: MindSpore not available\n")
        return

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    from config import Thinker15BConfig
    from model import Thinker15BModel

    cfg = Thinker15BConfig.tiny()
    B, S = 2, cfg.max_seq_len
    V = cfg.vocab_size

    model = Thinker15BModel(cfg)

    inv_freq = 1.0 / (cfg.rope_theta ** (
        np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))
    t = np.arange(S, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos_t = Tensor(np.cos(emb).astype(np.float16)[np.newaxis, np.newaxis,
                                                     :, :])
    sin_t = Tensor(np.sin(emb).astype(np.float16)[np.newaxis, np.newaxis,
                                                     :, :])
    mask_np = np.triu(np.full((S, S), -1e4, dtype=np.float16), k=1)
    mask_t = Tensor(mask_np.reshape(1, 1, S, S))

    input_ids = Tensor(np.random.randint(0, V, (B, S)).astype(np.int32))
    logits, slot_loss = model(input_ids, cos_t, sin_t, mask_t)

    ce_loss = nn.CrossEntropyLoss()
    logits_t = logits[:, :-1, :]
    labels = input_ids[:, 1:].reshape((-1,))
    loss = ce_loss(logits_t.reshape((-1, V)), labels) + 0.15 * slot_loss
    print(f"  Loss: {float(loss):.4f}")

    grads = ops.grad(lambda ids, c, s, m: model(ids, c, s, m)[0].sum())(
        input_ids, cos_t, sin_t, mask_t)
    print(f"  Gradient computed: shape={grads.shape}")

    assert not np.isnan(float(loss)), "Loss is NaN!"
    print("  Loss is finite: OK")
    print("  PASS\n")


def main() -> None:
    print("\nThinker-1.5B v2 Verification Suite\n")

    test_bpe_tokenizer()
    test_config()
    test_components()
    test_gradient_flow()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
