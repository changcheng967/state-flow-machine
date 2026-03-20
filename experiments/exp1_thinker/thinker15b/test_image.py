"""test_image.py — Verify OpenI custom image works for training.

Writes to MULTIPLE directories for maximum compatibility with OpenI platform.

Output files (at least one should appear):
  - /cache/output/test_result.log
  - ./test_result.log  (current working directory)
  - /tmp/test_result.log
"""

# ═══════════════════════════════════════════════════════════════════
# SANITY CHECK — write to as many places as possible, using ONLY stdlib
# This runs BEFORE any MindSpore/c2net imports that might fail.
# ═════════════════════════════════════════════════════════════════

import sys
import os
import json
import time
import traceback

# Collect all possible output directories to write logs
_LOG_DIRS = ["/cache/output"]
_cwd = os.getcwd()

# Also try the OpenI standard paths
for p in ["/cache/output", "/home/ma-user/work/output",
            "/home/ma-user/modelarts/workspace/device0",
            _cwd, "/tmp"]:
    if os.path.isdir(p) or p == "/tmp":
        _LOG_DIRS.insert(0, p)

_LOG_DIRS = list(dict.fromkeys(_LOG_DIRS).keys())  # deduplicate

def _boot_log(msg: str) -> None:
    """Write to ALL possible output locations + stderr. Never fails."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    for d in _LOG_DIRS:
        try:
            with open(os.path.join(d, "test_result.log"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


_boot_log(f"=== test_image.py started ===")
_boot_log(f"  pid={os.getpid()}")
_boot_log(f"  python={sys.version}")
_boot_log(f"  cwd={os.getcwd()}")
_boot_log(f"  writable dirs: {_LOG_DIRS}")
_boot_log(f"  HOME={os.environ.get('HOME', '?')}")
_boot_log(f"  RANK_ID={os.environ.get('RANK_ID', '?')}")
_boot_log(f"  DEVICE_ID={os.environ.get('DEVICE_ID', '?')}")
_boot_log(f"  OUTPUT_PATH env={os.environ.get('OUTPUT_PATH', 'not set')}")
_boot_log("  sys.path[:3]={sys.path[:3]}")

# ── Discover output path (try c2net, fall back to /cache/output) ───
OUTPUT_PATH = "/cache/output"

try:
    from c2net.context import prepare
    _boot_log("c2net import OK, calling prepare()...")
    _ctx = prepare()
    OUTPUT_PATH = _ctx.output_path
    _boot_log(f"c2net prepare() OK, output_path={OUTPUT_PATH}")
    _LOG_DIRS.insert(0, OUTPUT_PATH)
except Exception as e:
    _boot_log(f"c2net not available: {type(e).__name__}: {e}")
    _boot_log("Using fallback output path: /cache/output")

os.makedirs(OUTPUT_PATH, exist_ok=True)
_boot_log(f"Final OUTPUT_PATH: {OUTPUT_PATH}")
_boot_log(f"OUTPUT_PATH exists: {os.path.isdir(OUTPUT_PATH)}")
_boot_log(f"OUTPUT_PATH isdir: {os.path.isdir(OUTPUT_PATH)}")

# Write a test file to verify write access
_test_file = os.path.join(OUTPUT_PATH, "_write_test.txt")
try:
    with open(_test_file, "w") as f:
        f.write(f"write test OK at {time.strftime('%H:%M:%S')}\n")
    _boot_log(f"Write test OK: {_test_file}")
except Exception as e:
    _boot_log(f"Write test FAILED: {e}")

# ═════════════════════════════════════════════════════════════════
# LOGGING (uses discovered OUTPUT_PATH)
# ═════════════════════════════════════════════════════════════════

_log_fh = None

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def setup_log() -> None:
    global _log_fh
    _LOG_DIRS.insert(0, OUTPUT_PATH)
    log_path = os.path.join(OUTPUT_PATH, "test_result.log")
    _log_fh = open(log_path, "w")
    log(f"Log file: {log_path}")

setup_log()

# ═════════════════════════════════════════════════════════════════
# TEST FRAMEWORK
# ═════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════

log(f"\n{'='*60}")
log("OpenI Custom Image Compatibility Test")
log(f"{'='*60}")

# ── 1. MindSpore import ────────────────────────────────────────────
ms_ver = "?"
try:
    import mindspore as ms
    from mindspore import nn, ops, Tensor
    ms_ver = ms.__version__
    check(f"MindSpore import + version ({ms_ver})", lambda: ms_ver)
except Exception as e:
    log(f"FATAL: Cannot import MindSpore: {type(e).__name__}: {e}")
    log(f"sys.path: {sys.path[:3]}")
    # Write error and exit
    _boot_log(f"FATAL: Cannot import MindSpore: {e}\nsys.path={sys.path[:3]}")
    sys.exit(1)

import numpy as np

# ── 2. NPU detection ───────────────────────────────────────────────
def test_npu():
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    ms.set_context(device_target="Ascend", device_id=device_id)
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = ops.matmul(x, x.transpose(1, 0))
    return f"Ascend NPU OK, device_id={device_id}, matmul shape={y.shape}"

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
    return "All tensor ops OK"

check("Basic tensor ops", test_tensor_ops)

# ── 4. nn.Dense + nn.Embedding ────────────────────────────────────
def test_nn_layers():
    d = nn.Dense(128, 64, has_bias=True)
    x = Tensor(np.random.randn(4, 128), dtype=np.float32)
    y = d(x)
    emb = nn.Embedding(1000, 128)
    ids = Tensor(np.random.randint(0, 1000, (4, 10)).astype(np.int32))
    e = emb(ids)
    mm = ops.MatMul(transpose_b=True)
    logits = mm(y, Tensor(np.random.randn(128, 64), dtype=np.float32))
    return f"Dense({y.shape}), Emb({e.shape}), MatMul({logits.shape})"

check("nn.Dense + nn.Embedding + MatMul", test_nn_layers)

# ── 5. @ms.jit compilation ────────────────────────────────────────
def test_jit():
    class AddCell(nn.Cell):
        @ms.jit
        def construct(self, x, y):
            return x + y * 2

    cell = AddCell()
    c = cell(Tensor(np.ones((2, 3), dtype=np.float32)),
              Tensor(np.ones((2, 3), dtype=np.float32)))
    assert c.shape == (2, 3)
    return "@ms.jit compilation OK"

check("@ms.jit compilation", test_jit)

# ── 6. GradOperation ──────────────────────────────────────────────
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
    y, grads = cell(Tensor(np.random.randn(4, 16), dtype=np.float32))
    return f"GradOperation OK, {len(grads)} grads"

check("ops.GradOperation", test_grad)

# ── 7. AdamWeightDecay ───────────────────────────────────────────
def test_optimizer():
    net = nn.Dense(32, 16)
    opt = nn.AdamWeightDecay(net.trainable_params(),
                             learning_rate=Tensor([0.001], ms.float32))
    grads = [Tensor(np.random.randn(*p.shape), dtype=ms.float32) for p in net.trainable_params()]
    opt(tuple(grads))
    return f"AdamWeightDecay OK, {len(net.trainable_params())} params"

check("nn.AdamWeightDecay", test_optimizer)

# ── 8. ClipByGlobalNorm ───────────────────────────────────────────
def test_clip():
    if hasattr(nn, 'ClipByGlobalNorm'):
        clip = nn.ClipByGlobalNorm(1.0)
        g1 = Tensor(np.random.randn(4, 8), dtype=np.float32) * 10
        g2 = Tensor(np.random.randn(4, 8), dtype=np.float32) * 10
        clipped, norm = clip((g1, g2))
        return f"ClipByGlobalNorm EXISTS, norm={float(norm):.4f}"
    else:
        return "nn.ClipByGlobalNorm NOT FOUND (need alternative)"

check("nn.ClipByGlobalNorm", test_clip)

# ── 9. ForiLoop / WhileLoop ───────────────────────────────────────
def test_loop_ops():
    has_fori = hasattr(ops, 'ForiLoop')
    has_while = hasattr(ops, 'WhileLoop')
    if has_fori:
        fl = ops.ForiLoop()
        def body(x, i):
            return x + i
        result = fl(body, 0, 10, Tensor(0.0, ms.float32))
        return f"ForiLoop EXISTS, result={float(result):.1f}"
    elif has_while:
        return "ForiLoop not found, WhileLoop exists"
    else:
        return "ForiLoop/WhileLoop NOT FOUND"

check("ops.ForiLoop / WhileLoop (MS 2.4+)", test_loop_ops)

# ── 10. Mini DeltaNet forward ─────────────────────────────────────
def test_deltanet_forward():
    BRIDGE_DIM, D, NH, HD = 256, 256, 16, 16

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
            return ops.stack(outputs, axis=0).transpose(1, 0, 2, 3).reshape(B, S, NH * HD)

    net = MiniDeltaNet()
    out = net(Tensor(np.random.randn(2, 32, BRIDGE_DIM).astype(np.float16)))
    return f"MiniDeltaNet OK, output={out.shape}"

check("Mini DeltaNet forward (B=2, S=32)", test_deltanet_forward)

# ── 11. DeltaNet @ms.jit ──────────────────────────────────────────
def test_deltanet_jit():
    BRIDGE_DIM, D, NH, HD = 256, 256, 16, 16

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
            return ops.stack(outputs, axis=0).transpose(1, 0, 2, 3).reshape(B, S, NH * HD)

    net = MiniDeltaNet()
    net.set_train(True)
    out = net(Tensor(np.random.randn(2, 64, BRIDGE_DIM).astype(np.float16)))
    return f"DeltaNet @ms.jit OK, output={out.shape}"

check("DeltaNet @ms.jit (B=2, S=64)", test_deltanet_jit)

# ── 12. Mini training step ─────────────────────────────────────────
def test_training_step():
    class MiniModel(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense1 = nn.Dense(64, 128)
            self.dense2 = nn.Dense(128, 64)

        def construct(self, x):
            return self.dense2(self.dense1(x))

    class TrainStep(nn.Cell):
        def __init__(self, net, opt):
            super().__init__()
            self.net = net
            self.opt = opt
            self.weights = opt.parameters
            self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
            self.sens = Tensor([1.0], ms.float32)
            self.loss_fn = nn.MSELoss()
            self.clip = nn.ClipByGlobalNorm(1.0) if hasattr(nn, 'ClipByGlobalNorm') else None

        @ms.jit
        def construct(self, x, target):
            loss = self.loss_fn(self.net(x), target)
            grads = self.grad_op(self.net, self.weights)(x, self.sens)
            if self.clip is not None:
                grads, _ = self.clip(grads)
            self.opt(grads)
            return loss

    net = MiniModel()
    opt = nn.AdamWeightDecay(net.trainable_params(),
                             learning_rate=Tensor([0.001], ms.float32))
    step = TrainStep(net, opt)
    loss = step(Tensor(np.random.randn(4, 64), dtype=np.float32),
                   Tensor(np.random.randn(4, 64), dtype=np.float32))
    return f"Training step OK, loss={float(loss):.4f}"

check("Mini training step (forward + backward + optimizer)", test_training_step)

# ═════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════

log(f"\n{'='*60}")
log(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
log(f"{'='*60}")

if errors:
    log("\nFAILED TESTS:")
    for name, err in errors:
        log(f"  - {name}: {type(err).__name__}: {err}")

if failed == 0:
    log("\nALL TESTS PASSED! Image ready for training.")
else:
    log(f"\n{failed} test(s) failed.")

log(f"\nMindSpore: {ms_ver}")
log(f"Python: {sys.version}")
log(f"NPUs: {ms.get_context('device_num')}")

# Write JSON
json_path = os.path.join(OUTPUT_PATH, "test_result.json")
with open(json_path, "w") as f:
    json.dump({
        "passed": passed, "failed": failed, "total": passed + failed,
        "all_passed": failed == 0, "mindspore_version": ms_ver,
        "python_version": sys.version,
        "npus": ms.get_context("device_num"),
        "tests": results,
        "errors": [{"name": n, "error": f"{type(e).__name__}: {e}"} for n, e in errors],
    }, f, indent=2)
log(f"JSON written to {json_path}")

if _log_fh is not None:
    _log_fh.close()
