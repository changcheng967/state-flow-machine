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
# Quick smoke test
python experiments/exp0_state_tracking/run.py --quick

# Full experiment
python experiments/exp0_state_tracking/run.py --epochs 10 --samples 10000
```

**PASS criteria**: State Slots generalize to 4× program length, transformer doesn't.

## Project Structure

```
sfm/
├── sfm/                          # Core library
│   ├── config.py                 # All hyperparameters
│   ├── model.py                  # Full SFM assembly
│   ├── systems/                  # 4 systems
│   │   ├── perception.py         # System 1
│   │   ├── execution.py          # System 2
│   │   ├── structure.py          # System 3
│   │   └── meta.py               # System 4
│   ├── components/               # Reusable building blocks
│   │   ├── deltanet_cell.py      # Core recurrent cell
│   │   ├── state_slots.py        # Variable tracking
│   │   ├── linear_attention.py   # O(n) attention
│   │   ├── graph_attention.py    # GNN for structure
│   │   ├── adaptive_halting.py   # Dynamic compute
│   │   └── cross_system_bridge.py # System communication
│   ├── tokenizer/
│   │   └── code_tokenizer.py
│   └── utils/
│       └── device.py             # Auto device selection
├── experiments/                  # Experiments
│   ├── exp0_state_tracking/      # Prove System 2 works
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

## Testing

Each component has a standalone smoke test:

```bash
# Test individual components
python sfm/components/deltanet_cell.py
python sfm/components/state_slots.py
python sfm/components/linear_attention.py
python sfm/components/graph_attention.py
python sfm/components/adaptive_halting.py
python sfm/components/cross_system_bridge.py

# Test systems
python sfm/systems/perception.py
python sfm/systems/execution.py
python sfm/systems/structure.py
python sfm/systems/meta.py

# Test full model
python sfm/model.py
```

## Key Design Principles

1. **All operations differentiable** — No hard discrete ops, soft attention everywhere
2. **Sequential execution** — System 2 processes statements sequentially (execution is sequential)
3. **Parallel perception** — System 1 processes tokens in parallel with linear attention
4. **Fixed bridge dimension** — Cross-system bridges use fixed 256d vectors
5. **Modular testing** — Every component runnable standalone

## Roadmap

- [x] Core architecture implementation
- [x] Experiment 0: State tracking proof
- [ ] Experiment 1: Full SFM integration
- [ ] Experiment 2: SWE-bench Lite evaluation

## Citation

If you use this architecture, please cite:

```bibtex
@software{sfm2025,
  title = {State-Flow Machine: A Post-Transformer Architecture for Code Intelligence},
  year = {2025},
  note = {Novel architecture with 4 specialized systems}
}
```

## License

MIT License
