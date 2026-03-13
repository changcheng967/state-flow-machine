# CLAUDE.md — State-Flow Machine (SFM)

## What This Is
A novel post-transformer architecture for code intelligence. Not a wrapper, not a fine-tune, not RAG — a new architecture with 4 specialized systems that replaces the transformer paradigm for code tasks.

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

## Repo Structure
```
sfm/
├── sfm/                          # Core library
│   ├── config.py                 # All hyperparameters
│   ├── model.py                  # Full SFM assembly
│   ├── systems/                  # 4 systems
│   │   ├── perception.py
│   │   ├── execution.py
│   │   ├── structure.py
│   │   └── meta.py
│   ├── components/               # Reusable building blocks
│   │   ├── deltanet_cell.py
│   │   ├── state_slots.py
│   │   ├── linear_attention.py
│   │   ├── graph_attention.py
│   │   ├── cross_system_bridge.py
│   │   └── adaptive_halting.py
│   ├── tokenizer/
│   │   └── code_tokenizer.py
│   └── utils/
│       └── device.py
├── experiments/                  # Modular experiments to prove each system
│   ├── exp0_state_tracking/      # Prove System 2 works (State Slots vs Transformer)
│   └── exp1_full_sfm/            # Full integration test
└── scripts/
    └── visualize.py
```

## Experiments Roadmap
- **Exp 0**: Train System 2 (State Slots) standalone on synthetic execution traces. Compare against transformer baseline at same param count (~50M). PASS = State Slots generalize to 4× program length, transformer doesn't.
- **Exp 1**: Full SFM integration on code completion and bug-fix tasks.
- **Exp 2**: SWE-bench Lite evaluation.

## Hardware Target
4× Huawei Ascend 910ProA NPUs (32 GB each). All code auto-falls back to CUDA → CPU.

## Key Design Rules
- All operations differentiable. No hard discrete ops. Soft attention everywhere.
- System 2 processes statements SEQUENTIALLY. This is intentional — execution is sequential.
- System 1 processes tokens in PARALLEL with linear attention.
- Cross-system bridges use fixed 256d vectors. Systems are modular and testable independently.
- Every component must be runnable standalone: `python sfm/components/state_slots.py` should run a smoke test.

## Coding Standards
- Type hints on every function.
- Docstring on every module and class.
- No placeholder implementations. Every forward() must compute real outputs.
- Seeds fixed for reproducibility.
- `python sfm/model.py` instantiates full SFM and prints param counts per system.