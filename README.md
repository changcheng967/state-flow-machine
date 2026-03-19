# State-Flow Machine (SFM)

A novel post-transformer architecture for code intelligence. Not a wrapper, not a fine-tune, not RAG — a new architecture with 4 specialized systems that replaces the transformer paradigm for code tasks.

## Why

Every coding LLM (GPT, Claude, Gemini, DeepSeek) processes code as flat text. They predict the next token. But coding is about **state transformations** — what a program does vs what it should do. Transformers are provably incapable of tracking program state at length (TC0 circuit complexity limit, Siems et al. ICLR 2025). This architecture fixes that.

## The 4 Systems

| System | Role | Mechanism |
|--------|------|-----------|
| **Perception** | Token reading | Linear-attention decoder, O(n) |
| **Execution** | State tracking | State Slot Bank with Gated DeltaNet |
| **Structure** | Dependency map | Dynamic graph neural network |
| **Meta** | Self-correction | Recurrent controller with verification head |

**System 2 (Execution)** is the breakthrough. It uses a State Slot Bank — explicit memory registers that bind to variables and track values through execution. Each slot uses a Gated DeltaNet recurrent cell with eigenvalues in [-1,1] (the negative eigenvalues enable state tracking that transformers and standard RNNs cannot do). Processes statements sequentially with adaptive compute (1-8 internal ticks per statement). Cross-system bridges exchange information every 2 perception layers via projection to shared 256d space.

## Experiment 0: State Tracking (PASS)

**Task**: Predict the final value of a target variable after a sequence of arithmetic operations. 101-class classification. Trained on short programs (10-27 ops), evaluated at length multipliers up to 32x.

| Length | State Slots (961K) | Transformer-Fair (443K) | Transformer-Large (2.2M) |
|--------|--------------------|--------------------------|---------------------------|
| 1x | 99.9% | 100.0% | 100.0% |
| 2x | 92.9% | 99.0% | 99.5% |
| 4x | **62.0%** | 1.9% | 3.1% |
| 8x | **35.3%** | 1.3% | 1.0% |
| 16x | **5.1%** | 0.9% | 0.7% |
| 32x | **5.0%** | 1.0% | 0.8% |

Transformers dominate in-distribution but collapse to ~2% at 4x length. State Slots retains 62% at 4x and 35% at 8x — a **30x gap** in generalization ratio. The 2.2M Transformer-Large performs no better than the 443K Transformer-Fair at extrapolation, confirming this is an architectural limitation, not a scale issue.

## Repo Structure

```
sfm/                                    # Core library (PyTorch, reference implementation)
  config.py                             #   All hyperparameters
  model.py                              #   Full 4-system assembly
  systems/                              #   4 systems
    perception.py                      #     System 1: Linear-attention decoder
    execution.py                       #     System 2: State Slot Bank + DeltaNet
    structure.py                       #     System 3: Dynamic GNN
    meta.py                            #     System 4: Recurrent controller
  components/                           #   Reusable building blocks
    deltanet_cell.py                   #     Gated DeltaNet with parallel scan
    state_slots.py                     #     State Slot Bank (16 typed registers)
    linear_attention.py                 #     Linear attention
    graph_attention.py                  #     Graph neural network ops
    cross_system_bridge.py              #     Inter-system communication
    adaptive_halting.py                #     Adaptive compute gating
  tokenizer/                            #   Code tokenizer
    code_tokenizer.py
  utils/                                #   Device & distributed utilities

experiments/
  exp0_state_tracking/                 #   Exp 0: State tracking (complete, PASS)
    finish_experiment.py               #     Training + eval (MindSpore, Ascend)
    generate_data.py                   #     Synthetic execution trace generator
    evaluate.py                        #     Generalization eval (1x-32x)
    baseline_transformer.py            #     Transformer baselines
    dataset.py                         #     Dataset loader

scripts/
  visualize.py                         #   Result visualization

length_generalization.png             #   Exp 0 result chart
evaluation_results.json                #   Exp 0 numerical results
```

## Hardware Target

Huawei Ascend 910 NPUs. DaVinci Cube unit: 16x16x16 MAC array — all dimensions optimized for multiples of 16. Experiments run on OpenI (4x Ascend 910, MindSpore 2.2, CANN 7).

## License

MIT
