# State-Flow Machine (SFM)

A novel post-transformer architecture for code intelligence.

## What This Is

Not a wrapper, not a fine-tune, not RAG — a new architecture with 4 specialized systems that replaces the transformer paradigm for code tasks.

## Why It Exists

Every coding LLM (GPT, Claude, Gemini, DeepSeek) processes code as flat text. They predict the next token. But coding is about state transformations — what a program does vs what it should do. Transformers are provably incapable of tracking program state (TC0 circuit complexity limit, Siems et al. ICLR 2025). This architecture fixes that.

## The 4 Systems

### System 1 — Perception
Linear-attention decoder. O(n) not O(n^2). Reads tokens. That's it.

### System 2 — Execution (the breakthrough)
State Slot Bank: 16 registers that explicitly bind to variables and track values through execution. Uses **Gated DeltaNet** recurrent cell with eigenvalues in [-1,1] (not [0,1] — the negative eigenvalues enable state tracking that transformers and standard RNNs cannot do). The forget gate (from ICLR 2025 Gated DeltaNet paper) controls state retention vs injection, enabling proper variable reassignment tracking. Processes statements sequentially with adaptive compute (1-8 internal ticks per statement).

### System 3 — Structure
Dynamic graph neural network. Nodes = functions, classes, variables, files. Edges = calls, imports, mutates, reads. Updated via sparse message-passing. Gives the model a live dependency map so it doesn't break existing code when editing.

### System 4 — Meta
Small recurrent controller. Maintains a hypothesis register (what it thinks is wrong) and a plan stack (what it intends to do). Has a verification head that checks its own output before emitting. Prevents death-spirals (repeated failed attempts).

### Cross-System Bridges
Every 2 perception layers, all 4 systems exchange information via projection to a shared 256d space.

## Experiment 0 Results: State Tracking

**Task**: Predict the final value of a variable after a sequence of arithmetic operations. Programs range from 10 to 80 operations. Models trained on 10-operation programs, evaluated at 1x/2x/4x/8x length.

### Length Generalization (Exact Match Accuracy — v2)

| Length | State Slots | Transformer-Fair | Transformer-Large |
|--------|-------------|------------------|-------------------|
| 1x (10 ops) | **100.0%** | 90.9% | 98.0% |
| 2x (20 ops) | **99.6%** | 91.3% | 93.7% |
| 4x (40 ops) | **87.8%** | 1.2% | 2.2% |
| 8x (80 ops) | **44.9%** | 0.6% | 0.4% |

### Generalization Ratios (accuracy at Nx relative to 1x)

| Model | 4x Ratio | 8x Ratio |
|-------|----------|----------|
| State Slots | **87.8%** | **44.9%** |
| Transformer-Fair | 1.3% | 0.7% |
| Transformer-Large | 2.2% | 0.4% |

![Length Generalization Results](length_generalization.png)

### Key Findings

- **State Slots achieve 100% in-distribution accuracy** (v2 classification + intermediate supervision closed the gap completely).
- **State Slots retain 87.8% of accuracy at 4x length** vs 1-2% for transformers — a **67x improvement** in generalization ratio.
- **At 8x (80 operations), State Slots still achieve 44.9%** while both transformers have collapsed to <1%.
- Transformers are architecturally incapable of length generalization due to the TC0 circuit complexity limit (Siems et al. ICLR 2025).

### v2 Improvements over v1

| Aspect | v1 | v2 |
|--------|----|----|
| Loss function | MSE regression | 101-class CE + intermediate aux loss |
| In-distribution accuracy | 11.2% | **100.0%** |
| 4x generalization | 8.9% (79% retention) | **87.8%** (87.8% retention) |
| 8x generalization | 5.1% (46% retention) | **44.9%** (44.9% retention) |

### v3 Changes (pending training)

- Fair comparison: transformers use 101-class CE (same loss as execution model)
- Mixed-length training data: 60% 10-27, 20% 28-50, 15% 51-80, 5% 81-120 ops
- State passing: carry detached recurrent state across training batches
- LR grid search for both execution and transformer models
- Extended evaluation up to 32x (320 ops)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from sfm import create_sfm, SFMConfig

# Create model with small config for testing
model = create_sfm(SFMConfig.small())

# Or use default config
model = create_sfm()

# Forward pass
import torch
tokens = torch.randint(0, 32000, (1, 64))  # batch=1, seq_len=64
logits = model(tokens)  # (1, 64, 32000)
```

## Running the Experiments

### Experiment 0: State Tracking

Proves System 2 (State Slots) works better than transformers for tracking program state.

```bash
# Full experiment: trains 3 models + evaluates at 1x-32x (single NPU)
python experiments/exp0_state_tracking/finish_experiment.py

# Skip training, re-evaluate from saved checkpoints
python experiments/exp0_state_tracking/finish_experiment.py --skip_training

# Skip LR grid search, use defaults
python experiments/exp0_state_tracking/finish_experiment.py --skip_lr_search
```

**PASS criteria**: State Slots generalize to 4x program length, transformer doesn't. (Achieved — 87.8% vs 1-2%)

## Ascend NPU Optimization

This architecture is **optimized for Huawei Ascend NPUs** with the DaVinci Cube unit:

### Dimension Alignment
All tensor dimensions are multiples of 16 to match the Cube unit's 16x16x16 MAC array:
- `d_model`: 256 (16 x 16)
- `hidden_dim`: 256 (16 x 16)
- `num_slots`: 16 (16 x 1)
- `slot_dim`: 128 (16 x 8)

### Cube-First Parallel Scan
DeltaNet uses a matrix-based parallel scan:
1. Reshape sequence into 16-step chunks
2. Within-chunk recurrence as matrix multiply (Cube unit)
3. Between-chunk carry on Vector unit
4. Pipeline parallelism: Cube computes next chunk while Vector carries

### Batched Operations
- All slot operations use `torch.bmm` for batched matmul
- MATCH: `scores = torch.matmul(query, slot_keys.T)` — one Cube call
- READ: `values = torch.matmul(weights, slot_values)` — one Cube call
- WRITE: `torch.baddbmm` for erase-then-write update

### Performance Targets
- Quick mode (100 samples, 1 epoch, 1 NPU): < 60 seconds
- Full mode (10k samples, 50 epochs, 4 NPUs): < 30 minutes

## Project Structure

```
sfm/
├── sfm/                          # Core library
│   ├── config.py                 # All hyperparameters (16-aligned)
│   ├── model.py                  # Full SFM assembly
│   ├── systems/                  # 4 systems
│   │   ├── perception.py         # System 1
│   │   ├── execution.py          # System 2 (Cube-optimized, Gated DeltaNet)
│   │   ├── structure.py          # System 3
│   │   └── meta.py               # System 4
│   ├── components/               # Reusable building blocks
│   │   ├── deltanet_cell.py      # Gated DeltaNet + Cube-first parallel scan
│   │   ├── state_slots.py        # Cube-optimized slots (default 16)
│   │   ├── linear_attention.py   # O(n) attention
│   │   ├── graph_attention.py    # GNN for structure
│   │   ├── adaptive_halting.py   # Dynamic compute
│   │   └── cross_system_bridge.py # System communication
│   ├── tokenizer/
│   │   └── code_tokenizer.py
│   └── utils/
│       └── device.py             # NPU optimization
├── experiments/                  # Experiments
│   ├── exp0_state_tracking/      # Prove System 2 works
│   │   ├── run.py                # Main entry point
│   │   ├── train.py              # Unified training (single + multi-NPU)
│   │   ├── finish_experiment.py  # Single-NPU transformer training + eval
│   │   ├── dataset.py            # Data loading
│   │   ├── generate_data.py      # Synthetic data
│   │   ├── baseline_transformer.py
│   │   └── evaluate.py
│   └── exp1_full_sfm/            # Full integration
└── scripts/
    └── visualize.py
```

## Hardware Requirements

**Designed for Huawei Ascend NPUs only.**

Requires:
- Huawei Ascend NPU (910ProA recommended)
- torch_npu package
- CANN software stack

No CUDA or CPU fallback — this architecture is optimized for Ascend hardware.

## Distributed Training Features

- **DistributedDataParallel (DDP)** - Multi-process, best performance
- **HCCL Backend** - Huawei's collective communication library
- **Gradient Accumulation** - effective_batch = batch_size x npus x accumulation_steps
- **Mixed Precision (AMP)** - Automatic loss scaling for FP16 (transformers only)
- **Cosine Annealing with Warmup** - Execution: lr=1e-3, warmup=200; Transformers: lr=3e-4, warmup=500

```bash
# Single node, 4 NPUs (effective batch = 32 x 4 x 2 = 256)
torchrun --nproc_per_node=4 experiments/exp0_state_tracking/train.py --epochs 50
```

## Testing

Each component has a standalone smoke test:

```bash
# Test individual components
python sfm/components/deltanet_cell.py    # Gated DeltaNet + Cube-first parallel scan
python sfm/components/state_slots.py      # Cube-optimized slots (16 slots)
python sfm/components/linear_attention.py
python sfm/components/graph_attention.py
python sfm/components/adaptive_halting.py
python sfm/components/cross_system_bridge.py

# Test systems
python sfm/systems/perception.py
python sfm/systems/execution.py           # Full execution system
python sfm/systems/structure.py
python sfm/systems/meta.py

# Test full model (prints dimension alignment)
python sfm/model.py
```

## Key Design Principles

1. **All operations differentiable** — No hard discrete ops, soft attention everywhere
2. **Sequential execution** — System 2 processes statements sequentially (execution is sequential)
3. **Parallel perception** — System 1 processes tokens in parallel with linear attention
4. **Fixed bridge dimension** — Cross-system bridges use fixed 256d vectors
5. **Modular testing** — Every component runnable standalone
6. **Cube-optimized dimensions** — All dimensions multiples of 16

## Roadmap

- [x] Core architecture implementation
- [x] Cube-first NPU optimization
- [x] Experiment 0: State tracking proof (regression task)
- [x] Gated DeltaNet forget gate (ICLR 2025)
- [x] Slot reduction 64→16 + skip connection
- [x] Experiment 0 v2: Classification + intermediate supervision (100% in-distribution, 87.8% at 4x)
- [x] Experiment 0 v3: Fair comparison + mixed-length training + state passing
- [ ] Experiment 0 v3 training run + 32x evaluation
- [ ] Experiment 1: Full SFM integration
- [ ] Experiment 2: SWE-bench Lite evaluation

## Citation

If you use this architecture, please cite:

```bibtex
@software{sfm2025,
  title = {State-Flow Machine: A Post-Transformer Architecture for Code Intelligence},
  year = {2025},
  note = {Novel architecture with 4 specialized systems, optimized for Ascend NPUs}
}
```

## License

MIT License
