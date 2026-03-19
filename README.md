# State-Flow Machine (SFM)

A novel post-transformer architecture for code intelligence, optimized for Huawei Ascend 910 NPUs.

## Architecture

State-Flow Machine replaces the single-transformer paradigm with 4 specialized systems. The core insight is that coding is about **state transformations** — what a program does vs what it should do — and explicit state tracking generalizes to longer programs in ways that implicit token-level models provably cannot (TC0 circuit complexity limit, Siems et al. ICLR 2025).

| System | Role | Mechanism |
|--------|------|-----------|
| **Perception** | Token reading | Linear-attention decoder, O(n) |
| **Execution** | State tracking | State Slot Bank with Gated DeltaNet (the breakthrough) |
| **Structure** | Dependency map | Dynamic graph neural network |
| **Meta** | Self-correction | Recurrent controller with verification head |

**System 2 (Execution)** is the breakthrough. It uses a State Slot Bank — explicit memory registers that bind to variables and track their values through execution. Each slot uses a Gated DeltaNet recurrent cell with eigenvalues in [-1,1] (the negative eigenvalues enable state tracking that transformers and standard RNNs cannot do). The system processes statements sequentially with adaptive compute (1-8 internal ticks per statement). Cross-system bridges exchange information every 2 perception layers via projection to a shared 256d space.

## Experiment 0: State Tracking (PASS)

**Task**: Predict the final value of a target variable after a sequence of arithmetic operations. 101-class classification. Trained on short programs (10-27 ops), evaluated at length multipliers up to 32x.

| Length | State Slots (961K) | Transformer-Fair (443K) | Transformer-Large (2.2M) |
|--------|--------------------|--------------------------|---------------------------|
| 1x | 99.9% | **100.0%** | **100.0%** |
| 2x | 92.9% | **99.0%** | 99.5% |
| 4x | **62.0%** | 1.9% | 3.1% |
| 8x | **35.3%** | 1.3% | 1.0% |
| 16x | **5.1%** | 0.9% | 0.7% |
| 32x | **5.0%** | 1.0% | 0.8% |

Transformers dominate in-distribution but collapse to ~2% at 4x length. State Slots retains 62% at 4x and 35% at 8x — a **30x gap** in generalization ratio. The 2.2M Transformer-Large performs no better than the 443K Transformer-Fair at extrapolation, confirming this is an architectural limitation, not a scale issue.

See `experiments/exp0_state_tracking/` for the full experiment code and `length_generalization.png` for the visualization.

## Experiment 1: SFM-Enhanced LLM Training (IN PROGRESS)

**Task**: Fine-tune Qwen2.5-Coder-7B with LoRA adapters + SFM Slot Banks on OpenThoughts-114k.

**Setup**:
- **Base model**: Qwen2.5-Coder-7B (frozen, ~7B params)
- **Adapters**: LoRA rank=32 alpha=64 on all attention + FFN projections (~75M trainable)
- **SFM**: Slot Banks (8 slots each) inserted at layers 7, 15, 23, 31
- **Hardware**: 4x Ascend 910 (32GB HBM each), MindSpore 2.2 + CANN 7
- **Training**: DATA_PARALLEL, gradient checkpointing, cosine LR with warmup
- **Platform**: OpenI (grampus launcher + c2net)

**Run**:
```bash
python train.py   # Self-contained, runs unattended with convergence detection
```

The training script (`train.py`) is fully self-contained — no pip installs needed. It:
- Discovers paths via c2net/grampus platform integration
- Loads Qwen2.5-Coder-7B from HuggingFace safetensors
- Tokenizes OpenThoughts-114k with proper BPE
- Trains with convergence-based stopping (24h safety limit)
- Uploads results via c2net

## Repo Structure

```
train.py                           # Exp 1: Self-contained 7B training script
sfm/                               # Core library
  systems/                         #   4 systems (perception, execution, structure, meta)
  components/                      #   Reusable blocks (DeltaNet, state slots, GNN, etc.)
  tokenizer/                       #   Code tokenizer
  utils/                           #   Device & distributed utilities
experiments/
  exp0_state_tracking/             #   Exp 0: State tracking (complete, PASS)
    finish_experiment.py           #     Self-contained training + eval
    generate_data.py               #     Synthetic execution trace generator
    evaluate.py                    #     Generalization evaluation (1x-32x)
    baseline_transformer.py        #     Transformer baselines
  exp1_thinker/                    #   Exp 1: Full SFM integration
    train_throughput.py            #     Throughput benchmark script
scripts/
  visualize.py                     # Visualization utilities
```

## Hardware

Huawei Ascend 910 NPUs only (910ProA recommended). DaVinci Cube unit: 16x16x16 MAC array — optimize matmul shapes for multiples of 16.

## License

MIT License
