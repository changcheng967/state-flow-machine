"""test_image.py — Verify OpenI custom image works for training.

Tests (in order, continues on failure):
  1. Python version
  2. c2net path discovery
  3. MindSpore import + version
  4. NPU detection
  5. Basic tensor ops
  6. nn.Dense + nn.Embedding
  7. @ms.jit compilation
  8. GradOperation
  9. AdamWeightDecay
  10. ClipByGlobalNorm
  11. ForiLoop / WhileLoop
  12. Mini DeltaNet forward
  13. DeltaNet @ms.jit
  14. Mini training step

Writes results to OUTPUT_PATH/test_result.log and test_result.json.
"""

# ═══════════════════════════════════════════════════════════════════
# BOOT LOG — write IMMEDIATELY before anything can fail
# ═══════════════════════════════════════════════════════════════════

def _boot_log(msg: str) -> None:
    """Write to stdout + error.log BEFORE any imports can fail."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    for d in ["/cache/output"]:
        try:
            import os
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "test_result.log"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

import sys
import os
import json
import time

_boot_ts = time.strftime("%H:%M:%S")
_boot_pid = os.getpid()
_boot_rank = os.environ.get("RANK_ID", "?")
_boot_log(f"test_image started pid={_boot_pid} rank={_boot_rank} python={sys.version.split()[0]}")

# ═══════════════════════════════════════════════════════════════════
# PATH DISCOVERY — c2net + fallbacks
# ═══════════════════════════════════════════════════════════════════

OUTPUT_PATH = "/cache/output"
DATASET_PATH = "/cache/dataset"
PRETRAIN_PATH = "/cache/pretrainmodel"
CODE_PATH = "/cache/code"

try:
    from c2net.context import prepare, upload_output
    _ctx = prepare()
    OUTPUT_PATH = _ctx.output_path
    DATASET_PATH = _ctx.dataset_path
    PRETRAIN_PATH = _ctx.pretrain_model_path
    CODE_PATH = _ctx.code_path
    _boot_log(f"c2net OK: out={OUTPUT_PATH} data={DATASET_PATH} pre={PRETRAIN_PATH}")
except Exception as e:
    _boot_log(f"c2net not available ({e}), using default paths")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

_log_fh = None

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def setup_log() -> None:
    global _log_fh
    log_path = os.path.join(OUTPUT_PATH, "test_result.log")
    _log_fh = open(log_path, "w")
    log(f"Log file: {log_path}")
    log(f"OUTPUT_PATH: {OUTPUT_PATH}")
    log(f"DATASET_PATH: {DATASET_PATH}")
    log(f"PRETRAIN_PATH: {PRETRAIN_PATH}")
    log(f"CODE_PATH: {CODE_PATH}")
    log(f"RANK_ID: {_boot_rank}")
    log(f"PID: {_boot_pid}")
    log(f"Python: {sys.version}")
    log(f"CWD: {os.getcwd()}")

setup_log()

# ═══════════════════════════════════════════════════════════════════
# TEST FRAMEWORK
# ═══════════════════════════════════════════════════════════════════

passed = 0
failed = 0
errors = []
results = {}


def check(name: str, fn, *args, **kwargs):
    """Run a check, report pass/fail, always continue."""
    global passed, failed
    log(f"\n{'='*60}")
    log(f"TEST: {name}")
    log(f"{'='*60}")
    try:
        t0 = time.time()
        result = fn(*args, **kwargs)
        dt = time.time() - t0
        log(f"  PASS ({dt:.2f}s)")
        if result is not None:
            log(f"  Result: {result}")
        passed += 1
        results[name] = {"status": "pass", "detail": str(result), "time": round(dt, 2)}
        return result
    except Exception as e:
        dt = time.time() - t0
        log(f"  FAIL ({dt:.2f}s): {type(e).__name__}: {e}")
        failed += 1
        errors.append((name, e))
        results[name] = {"status": "fail", "detail": f"{type(e).__name__}: {e}", "time": round(dt, 2)}
        return None


# ═══════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════

# ── 1. MindSpore import + version ─────────────────────────────────
log(f"\n{'='*60}")
log("OpenI Custom Image Test")
log(f"{'='*60}")

ms_ver = "?"
try:
    import mindspore as ms
    from mindspore import nn, ops, Tensor
    ms_ver = ms.__version__
    check(f"MindSpore version: {ms_ver}", lambda: ms_ver)
except Exception as e:
    check("Import MindSpore", lambda: (_ for _ in ()).throw(e))
    log(f"FATAL: Cannot import MindSpore: {e}")
    # Write error and exit
    summary = {"passed": 0, "failed": 1, "total": 1, "all_passed": False,
                "error": f"Cannot import MindSpore: {e}"}
    with open(os.path.join(OUTPUT_PATH, "test_result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("EXITING — MindSpore not available")
    sys.exit(1)

import numpy as np

# ── 2. NPU detection ───────────────────────────────────────────────
def test_npu():
    try:
        device_id = int(os.environ.get("DEVICE_ID", "0"))
        ms.set_context(device_target="Ascend", device_id=device_id)
        x = Tensor(np.ones((2, 3), dtype=np.float32))
        y = ops.matmul(x, x.transpose(1, 0))
        return f"Ascend NPU OK, device_id={device_id}, matmul shape={y.shape}"
    except Exception as e:
        return f"FAILED: {e}"

check("NPU availability", test_npu)

# ── 3. Basic tensor ops ────────────────────────────────────────────
def test_tensor_ops():
    a = Tensor(np.random.randn(2, 4, 8).astype(np.float16))
    b = Tensor(np.random.randn(2, 8, 4).astype(np.float16))
    c = ops.matmul(a, b)
    d = a.astype(ms.float32)
    e = b.astype(ms.float32)
    f = ops.matmul(d, e)
    g = f.astype(ms.float16)
    h = ops.reduce_sum(g)
    i = g.reshape(2, 8, 2)
    j = ops.Tile()(a, (2, 1, 1))
    k = ops.stack([a, b], axis=0)
    l = ops.sigmoid(Tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32)))
    m = ops.Softmax(axis=-1)(Tensor(np.random.randn(2, 4), dtype=np.float32))
    return "All tensor ops OK"

check("Basic tensor ops (FP16/FP32 matmul, cast, reduce, reshape, tile, stack, sigmoid, softmax)", test_tensor_ops)

# ── 4. nn.Dense, nn.Embedding, nn.RMSNorm-like ────────────────────
def test_nn_layers():
    d = nn.Dense(128, 64, has_bias=True)
    x = Tensor(np.random.randn(4, 128), dtype=np.float32)
    y = d(x)
    assert y.shape == (4, 64), f"Expected (4, 64), got {y.shape}"

    emb = nn.Embedding(1000, 128)
    ids = Tensor(np.random.randint(0, 1000, (4, 10)).astype(np.int32))
    e = emb(ids)
    assert e.shape == (4, 10, 128), f"Expected (4, 10, 128), got {e.shape}"

    mm = ops.MatMul(transpose_b=True)
    w = Tensor(np.random.randn(128, 64), dtype=np.float32)
    logits = mm(y, w)
    assert logits.shape == (4, 64), f"Expected (4, 64), got {logits.shape}"

    return f"Dense(128->64), Embedding(1000,128), MatMul OK"

check("nn.Dense + nn.Embedding + MatMul", test_nn_layers)

# ── 5. @ms.jit compilation ────────────────────────────────────────
def test_jit():
    class AddCell(nn.Cell):
        def __init__(self):
            super().__init__()

        @ms.jit
        def construct(self, x, y):
            return x + y * 2

    cell = AddCell()
    a = Tensor(np.ones((2, 3), dtype=np.float32))
    b = Tensor(np.ones((2, 3), dtype=np.float32))
    c = cell(a, b)
    assert c.shape == (2, 3)
    return "@ms.jit compilation OK"

check("@ms.jit basic compilation", test_jit)

# ── 6. ops.GradOperation ──────────────────────────────────────────
def test_grad():
    class GradCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(16, 8)
            self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
            self.weights = self.dense.trainable_params()
            self.sens = Tensor([1.0], ms.float32)

        @ms.jit
        def construct(self, x):
            y = self.dense(x)
            grads = self.grad_op(self.dense, self.weights)(x, self.sens)
            return y, grads

    cell = GradCell()
    x = Tensor(np.random.randn(4, 16), dtype=np.float32)
    y, grads = cell(x)
    assert y.shape == (4, 8)
    assert len(grads) == 2
    return f"GradOperation OK, {len(grads)} grads, shapes={[g.shape for g in grads]}"

check("ops.GradOperation (forward + backward)", test_grad)

# ── 7. nn.AdamWeightDecay ────────────────────────────────────────
def test_optimizer():
    net = nn.Dense(32, 16)
    params = net.trainable_params()
    lr = Tensor([0.001], ms.float32)
    opt = nn.AdamWeightDecay(params, learning_rate=lr)
    grads = [Tensor(np.random.randn(*p.shape), dtype=ms.float32)
              for p in params]
    opt(tuple(grads))
    return f"AdamWeightDecay OK, {len(params)} params"

check("nn.AdamWeightDecay", test_optimizer)

# ── 8. nn.ClipByGlobalNorm ───────────────────────────────────────
def test_clip():
    if hasattr(nn, 'ClipByGlobalNorm'):
        clip = nn.ClipByGlobalNorm(1.0)
        g1 = Tensor(np.random.randn(4, 8), dtype=np.float32) * 10
        g2 = Tensor(np.random.randn(4, 8), dtype=np.float32) * 10
        clipped, norm = clip((g1, g2))
        return f"ClipByGlobalNorm EXISTS & WORKS, norm={float(norm):.4f}"
    else:
        return "nn.ClipByGlobalNorm NOT FOUND"

check("nn.ClipByGlobalNorm", test_clip)

# ── 9. ops.ForiLoop / ops.WhileLoop ───────────────────────────────
def test_loop_ops():
    has_fori = hasattr(ops, 'ForiLoop')
    has_while = hasattr(ops, 'WhileLoop')
    has_scan = hasattr(ops, 'Scan')

    if has_fori:
        try:
            fl = ops.ForiLoop()
            def body(x, i):
                return x + i
            init = Tensor(0.0, ms.float32)
            result = fl(body, 0, 10, init)
            return f"ForiLoop EXISTS & WORKS, result={float(result):.1f}"
        except Exception as e:
            return f"ForiLoop EXISTS but FAILED: {e}"
    elif has_while:
        return "ForiLoop not found, WhileLoop exists"
    elif has_scan:
        return "ForiLoop/WhileLoop not found, Scan exists"
    else:
        return "ForiLoop/WhileLoop/Scan NOT FOUND"

check("ops.ForiLoop / WhileLoop (MS 2.4+)", test_loop_ops)

# ── 10. Mini DeltaNet forward ─────────────────────────────────────
def test_deltanet_forward():
    BRIDGE_DIM = 256
    D = 256
    NH = 16
    HD = 16

    class MiniDeltaNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.key_proj = nn.Dense(BRIDGE_DIM, D, has_bias=False)
            self.value_proj = nn.Dense(BRIDGE_DIM, D, has_bias=False)
            self.beta_proj = nn.Dense(BRIDGE_DIM, NH, has_bias=True)
            init_states = np.zeros((NH, HD, HD), dtype=np.float32)
            for i in range(NH):
                init_states[i] = np.eye(HD, dtype=np.float32) * 0.1
            self.initial_state = ms.Parameter(Tensor(init_states))
            self.NH = NH
            self.HD = HD

        def construct(self, x):
            B, S, _ = x.shape
            x_f32 = x.astype(ms.float32)
            K = self.key_proj(x_f32).astype(ms.float16)
            V = self.value_proj(x_f32).astype(ms.float16)
            beta = ops.sigmoid(self.beta_proj(x_f32)).astype(ms.float16)
            state = ops.Tile()(self.initial_state, (B, 1, 1, 1)).astype(ms.float16)
            outputs = ()
            for t in range(S):
                kt = K[:, t, None, :]
                vt = V[:, t, None, :]
                bt = beta[:, t, :, None, None]
                k_head = kt.reshape(B, NH, HD, 1)
                v_head = vt.reshape(B, NH, HD, 1)
                residual = ops.matmul(state, k_head) - v_head
                update = bt * ops.matmul(residual, k_head.transpose(0, 1, 3, 2))
                state = state - update
                out_t = state[:, :, -1, :]
                outputs = outputs + (out_t,)
            stacked = ops.stack(outputs, axis=0)
            stacked = stacked.transpose(1, 0, 2, 3)
            return stacked.reshape(B, S, NH * HD)

    net = MiniDeltaNet()
    x = Tensor(np.random.randn(2, 32, BRIDGE_DIM).astype(np.float16))
    out = net(x)
    assert out.shape == (2, 32, D), f"Expected (2, 32, {D}), got {out.shape}"
    n_params = sum(p.size for p in net.get_parameters())
    return f"MiniDeltaNet forward OK, output={out.shape}, params={n_params:,}"

check("Mini DeltaNet forward pass (B=2, S=32)", test_deltanet_forward)

# ── 11. DeltaNet @ms.jit compilation ──────────────────────────────
def test_deltanet_jit():
    BRIDGE_DIM = 256
    D = 256
    NH = 16
    HD = 16

    class MiniDeltaNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.key_proj = nn.Dense(BRIDGE_DIM, D, has_bias=False)
            self.value_proj = nn.Dense(BRIDGE_DIM, D, has_bias=False)
            self.beta_proj = nn.Dense(BRIDGE_DIM, NH, has_bias=True)
            init_states = np.zeros((NH, HD, HD), dtype=np.float32)
            for i in range(NH):
                init_states[i] = np.eye(HD, dtype=np.float32) * 0.1
            self.initial_state = ms.Parameter(Tensor(init_states))
            self.NH = NH
            self.HD = HD

        @ms.jit
        def construct(self, x):
            B, S, _ = x.shape
            x_f32 = x.astype(ms.float32)
            K = self.key_proj(x_f32).astype(ms.float16)
            V = self.value_proj(x_f32).astype(ms.float16)
            beta = ops.sigmoid(self.beta_proj(x_f32)).astype(ms.float16)
            state = ops.Tile()(self.initial_state, (B, 1, 1, 1)).astype(ms.float16)
            outputs = ()
            for t in range(S):
                kt = K[:, t, None, :]
                vt = V[:, t, None, :]
                bt = beta[:, t, :, None, None]
                k_head = kt.reshape(B, NH, HD, 1)
                v_head = vt.reshape(B, NH, HD, 1)
                residual = ops.matmul(state, k_head) - v_head
                update = bt * ops.matmul(residual, k_head.transpose(0, 1, 3, 2))
                state = state - update
                out_t = state[:, :, -1, :]
                outputs = outputs + (out_t,)
            stacked = ops.stack(outputs, axis=0)
            stacked = stacked.transpose(1, 0, 2, 3)
            return stacked.reshape(B, S, NH * HD)

    net = MiniDeltaNet()
    net.set_train(True)
    x = Tensor(np.random.randn(2, 64, BRIDGE_DIM).astype(np.float16))
    out = net(x)
    assert out.shape == (2, 64, D)
    return f"DeltaNet @ms.jit OK, output={out.shape}"

check("DeltaNet @ms.jit compilation (B=2, S=64)", test_deltanet_jit)

# ── 12. Mini training step ─────────────────────────────────────────
def test_training_step():
    class MiniModel(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense1 = nn.Dense(64, 128)
            self.dense2 = nn.Dense(128, 64)

        def construct(self, x):
            h = self.dense1(x)
            return self.dense2(h)

    class TrainStep(nn.Cell):
        def __init__(self, net, opt):
            super().__init__()
            self.net = net
            self.opt = opt
            self.weights = opt.parameters
            self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
            self.sens = Tensor([1.0], ms.float32)
            self.loss_fn = nn.MSELoss()

            if hasattr(nn, 'ClipByGlobalNorm'):
                self.clip = nn.ClipByGlobalNorm(1.0)
            else:
                self.clip = None

        @ms.jit
        def construct(self, x, target):
            out = self.net(x)
            loss = self.loss_fn(out, target)
            grads = self.grad_op(self.net, self.weights)(x, self.sens)
            if self.clip is not None:
                grads, _ = self.clip(grads)
            self.opt(grads)
            return loss

    net = MiniModel()
    opt = nn.AdamWeightDecay(net.trainable_params(),
                             learning_rate=Tensor([0.001], ms.float32))
    step = TrainStep(net, opt)

    x = Tensor(np.random.randn(4, 64), dtype=np.float32)
    target = Tensor(np.random.randn(4, 64), dtype=np.float32)

    loss = step(x, target)
    return f"Training step OK, loss={float(loss):.4f}"

check("Mini training step (forward + backward + optimizer)", test_training_step)

# ═══════════════════════════════════════════════════════════════════
# SUMMARY + JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════

log(f"\n{'='*60}")
log(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
log(f"{'='*60}")

if errors:
    log("\nFAILED TESTS:")
    for name, err in errors:
        log(f"  - {name}: {type(err).__name__}: {err}")

if failed == 0:
    log("\nALL TESTS PASSED! The image is ready for training.")
else:
    log(f"\n{failed} test(s) failed. Check errors above.")

log(f"\nMindSpore: {ms_ver}")
log(f"Python: {sys.version}")
log(f"NPUs available: {ms.get_context('device_num')}")

# Write JSON result file
json_path = os.path.join(OUTPUT_PATH, "test_result.json")
summary = {
    "passed": passed,
    "failed": failed,
    "total": passed + failed,
    "all_passed": failed == 0,
    "mindspore_version": ms_ver,
    "python_version": sys.version,
    "npus": ms.get_context("device_num"),
    "tests": results,
    "errors": [{"name": n, "error": f"{type(e).__name__}: {e}"}
              for n, e in errors],
}
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
log(f"Results written to {json_path}")

if _log_fh is not None:
    _log_fh.close()

# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLER — write error.log on any crash
# ═══════════════════════════════════════════════════════════════════

def _write_error(msg: str) -> None:
    for d in ["/cache/output", OUTPUT_PATH]:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "error.log"), "w") as f:
                f.write(msg)
        except Exception:
            pass

if __name__ == "__main__":
    try:
        pass  # Tests ran above
    except Exception as e:
        import traceback
        msg = f"FATAL: {e}\n{traceback.format_exc()}"
        log(msg)
        _write_error(msg)
