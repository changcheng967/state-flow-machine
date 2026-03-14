# State-Flow Machine (SFM)

A novel post-transformer architecture for code intelligence.

## What This Is

Not a wrapper, not a fine-tune, not RAG — a new architecture with 4 specialized systems that replaces the transformer paradigm for code tasks.

## Why It Exists

Every coding LLM (GPT, Claude, Gemini, DeepSeek) processes code as flat text. They predict the next token. But coding is about state transformations — what a program does vs what it should do. Transformers are provably incapable of tracking program state (TC0 circuit complexity limit, Siems et al. ICLR 2025). This architecture fixes that.

## The 4 Systems

### System 1 — Perception
Linear-attention decoder. O(n) not O(n²). Reads tokens. That's it.

### System 2 — Execution (the breakthrough)
State Slot Bank: 64 registers that explicitly bind to variables and track values through execution. Uses DeltaNet recurrent cell with eigenvalues in [-1,1] (not [0,1] — the negative eigenvalues enable state tracking that transformers and standard RNNs cannot do). Processes statements sequentially with adaptive compute (1-8 internal ticks per statement).

### System 3 — Structure
Dynamic graph neural network. Nodes = functions, classes, variables, files. Edges = calls, imports, mutates, reads. Updated via sparse message-passing. Gives the model a live dependency map so it doesn't break existing code when editing.

### System 4 — Meta
Small recurrent controller. Maintains a hypothesis register (what it thinks is wrong) and a plan stack (what it intends to do). Has a verification head that checks its own output before emitting. Prevents death-spirals (repeated failed attempts).

### Cross-System Bridges
Every 2 perception layers, all 4 systems exchange information via projection to a shared 256d space.

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
# Quick smoke test (single NPU) - should complete in < 60 seconds
python experiments/exp0_state_tracking/run.py --quick

# Full experiment (single NPU, 50 epochs)
python experiments/exp0_state_tracking/run.py --epochs 50 --samples 10000

# Multi-NPU distributed training (4 NPUs)
python experiments/exp0_state_tracking/run.py --npus 4 --epochs 50 --samples 10000
```

**PASS criteria**: State Slots generalize to 4× program length, transformer doesn't.

## Ascend NPU Optimization

This architecture is **optimized for Huawei Ascend NPUs** with the DaVinci Cube unit:

### Dimension Alignment
All tensor dimensions are multiples of 16 to match the Cube unit's 16×16×16 MAC array:
- `d_model`: 256 (16 × 16)
- `hidden_dim`: 256 (16 × 16)
- `num_slots`: 64 (16 × 4)
- `slot_dim`: 128 (16 × 8)

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
- State Slots accuracy on HARD programs: > 50% by epoch 50
- Transformer accuracy on HARD programs: < 30% by epoch 50

## Project Structure

```
sfm/
├── sfm/                          # Core library
│   ├── config.py                 # All hyperparameters (16-aligned)
│   ├── model.py                  # Full SFM assembly
│   ├── systems/                  # 4 systems
│   │   ├── perception.py         # System 1
│   │   ├── execution.py          # System 2 (Cube-optimized)
│   │   ├── structure.py          # System 3
│   │   └── meta.py               # System 4
│   ├── components/               # Reusable building blocks
│   │   ├── deltanet_cell.py      # Cube-first parallel scan
│   │   ├── state_slots.py        # Cube-optimized slot ops
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
- **Gradient Accumulation** - effective_batch = batch_size × npus × accumulation_steps
- **Mixed Precision (AMP)** - Automatic loss scaling for FP16
- **Cosine Annealing with Warmup** - Linear warmup 500 steps, then cosine decay

```bash
# Single node, 4 NPUs (effective batch = 32 × 4 × 2 = 256)
torchrun --nproc_per_node=4 experiments/exp0_state_tracking/train.py --epochs 50
```

## Testing

Each component has a standalone smoke test:

```bash
# Test individual components
python sfm/components/deltanet_cell.py    # Cube-first parallel scan
python sfm/components/state_slots.py      # Cube-optimized slots
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

## Training Configuration

Default training uses:
- **Task**: Regression (MSE loss) predicting final variable value [0, 100]
- **Learning rate**: 3e-4 with cosine annealing
- **Warmup**: 500 steps linear warmup
- **Epochs**: 50
- **Batch size**: 32 per NPU
- **Gradient accumulation**: 2 steps
- **Effective batch size**: 32 × 4 × 2 = 256 (with 4 NPUs)

## Roadmap

- [x] Core architecture implementation
- [x] Cube-first NPU optimization
- [x] Experiment 0: State tracking proof (regression task)
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
