"""Chapter 3 — Accounting: parameters, FLOPs, memory."""
from .schema import task

CH = "3 · Accounting: parameters, FLOPs, memory"

TASKS = [

task(
    id="param-count",
    title="Count a transformer's parameters",
    chapter=CH,
    section="3.1 Parameter count",
    level=1,
    entry="param_count",
    statement=(
        "Return the parameter count of a decoder-only transformer under this "
        "exact convention: token embedding V·d; per layer, attention projections "
        "q,k,v,o each d·d, a SwiGLU feed-forward with three matrices of d·d_ff, "
        "and two RMSNorm gains of d each; then a final norm of d and an untied "
        "output head of V·d. No biases anywhere. Interviewers ask this to see "
        "whether you know where the parameters actually sit."
    ),
    shapes="v_size · d_model · n_layers · d_ff int  ->  int",
    stub=("def param_count(v_size, d_model, n_layers, d_ff):\n"
          "    # -> total number of parameters\n    pass\n"),
    hints=[
        "Split it into three pieces: embedding, the repeated block, and the head.",
        "One block is 4·d² (attention) + 3·d·d_ff (SwiGLU) + 2·d (two norm gains).",
        "Total = V·d + n_layers·block + d + V·d. The embedding and the head are "
        "counted separately because they are untied.",
    ],
    solution=(
        "def param_count(v_size, d_model, n_layers, d_ff):\n"
        "    block = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model\n"
        "    return v_size * d_model + n_layers * block + d_model + v_size * d_model\n"
    ),
    solution_np=(
        "def param_count(v_size, d_model, n_layers, d_ff):\n"
        "    block = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model\n"
        "    return v_size * d_model + n_layers * block + d_model + v_size * d_model\n"
    ),
    traps=[
        "Counting two feed-forward matrices instead of three — SwiGLU has a gate, "
        "an up, and a down projection.",
        "Tying the embedding and the head when the spec says untied, halving the "
        "vocabulary contribution.",
        "Forgetting that the block term scales with d² while the embedding scales "
        "with V·d, so which dominates depends on depth.",
    ],
    tests='''
def checks(fn, check):
    def ref(v, d, n, f):
        return v*d + n*(4*d*d + 3*d*f + 2*d) + d + v*d
    check("matches the stated convention", lambda: fn(32000, 512, 8, 1376) == ref(32000, 512, 8, 1376))
    check("scales linearly in depth",
          lambda: fn(1000, 64, 4, 256) - fn(1000, 64, 2, 256) == 2 * (4*64*64 + 3*64*256 + 2*64))
    check("zero layers leaves embedding + head + final norm",
          lambda: fn(1000, 64, 0, 256) == 1000*64 + 64 + 1000*64)
    check("counts three FFN matrices, not two",
          lambda: fn(10, 8, 1, 32) != 10*8 + (4*64 + 2*8*32 + 16) + 8 + 10*8)
    check("returns an integer", lambda: isinstance(fn(100, 16, 1, 32), int))
''',
),

task(
    id="training-flops",
    title="Training FLOPs and the 6ND rule",
    chapter=CH,
    section="3.2 FLOPs, and where 6ND comes from",
    level=1,
    entry="flops",
    statement=(
        "Return forward, backward and total training FLOPs for N parameters over "
        "D tokens, using the standard estimate: 2ND for the forward pass, twice "
        "that for the backward, hence 6ND in total. The factor 2 in the forward "
        "is one multiply and one add per parameter per token; the backward costs "
        "two passes because it computes gradients with respect to both the inputs "
        "and the weights."
    ),
    shapes="n_params · n_tokens int  ->  dict 'forward', 'backward', 'total'",
    stub=("def flops(n_params, n_tokens):\n"
          "    # -> {'forward': ..., 'backward': ..., 'total': ...}\n    pass\n"),
    hints=[
        "Forward is 2·N·D.",
        "Backward is twice the forward.",
        "Total is their sum, which is the familiar 6ND.",
    ],
    solution=(
        "def flops(n_params, n_tokens):\n"
        "    fwd = 2 * n_params * n_tokens\n"
        "    bwd = 2 * fwd\n"
        "    return {'forward': fwd, 'backward': bwd, 'total': fwd + bwd}\n"
    ),
    solution_np=(
        "def flops(n_params, n_tokens):\n"
        "    fwd = 2 * n_params * n_tokens\n"
        "    bwd = 2 * fwd\n"
        "    return {'forward': fwd, 'backward': bwd, 'total': fwd + bwd}\n"
    ),
    traps=[
        "Using 2ND as the total rather than as the forward alone.",
        "Assuming the backward costs the same as the forward — it costs twice.",
        "Applying the rule to inference, where only the forward term applies.",
    ],
    tests='''
def checks(fn, check):
    o = fn(7e9, 2e12)
    check("forward is 2ND", lambda: abs(o["forward"] - 2*7e9*2e12) < 1)
    check("backward is twice the forward", lambda: abs(o["backward"] - 2*o["forward"]) < 1)
    check("total is 6ND", lambda: abs(o["total"] - 6*7e9*2e12) < 1)
    check("linear in tokens", lambda: abs(fn(1e9, 2e12)["total"] - 2*fn(1e9, 1e12)["total"]) < 1)
    check("linear in parameters", lambda: abs(fn(2e9, 1e12)["total"] - 2*fn(1e9, 1e12)["total"]) < 1)
''',
),

task(
    id="kv-cache-bytes",
    title="Size the KV cache",
    chapter=CH,
    section="3.3 Memory",
    level=2,
    entry="kv_bytes",
    statement=(
        "Compute the bytes held by the KV cache: two tensors (K and V) per layer, "
        "each of shape (batch, n_kv_heads, seq_len, head_dim), at the given bytes "
        "per element. This number, not the parameter count, is what limits how "
        "many sequences you can serve concurrently — it grows linearly with both "
        "batch and context length, while the weights stay fixed."
    ),
    shapes="batch · n_layers · seq_len · n_kv_heads · head_dim · bytes_per_elem int  ->  int",
    stub=("def kv_bytes(batch, n_layers, seq_len, n_kv_heads, head_dim, bytes_per_elem=2):\n"
          "    # -> total bytes held by the KV cache\n    pass\n"),
    hints=[
        "One layer holds two tensors, K and V, of identical shape.",
        "Each tensor has batch · n_kv_heads · seq_len · head_dim elements.",
        "Multiply by the number of layers and by bytes per element.",
    ],
    solution=(
        "def kv_bytes(batch, n_layers, seq_len, n_kv_heads, head_dim, bytes_per_elem=2):\n"
        "    return 2 * batch * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem\n"
    ),
    solution_np=(
        "def kv_bytes(batch, n_layers, seq_len, n_kv_heads, head_dim, bytes_per_elem=2):\n"
        "    return 2 * batch * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem\n"
    ),
    traps=[
        "Forgetting the factor 2 for K and V.",
        "Using the number of query heads instead of key/value heads, which "
        "overstates the cache by H/H_kv under grouped-query attention.",
        "Assuming fp16 when the deployment is fp8 or bf16 — the factor is a "
        "parameter, not a constant.",
    ],
    tests='''
def checks(fn, check):
    check("K and V are both counted",
          lambda: fn(1, 1, 1, 1, 1, 1) == 2)
    check("linear in sequence length",
          lambda: fn(1, 32, 2048, 8, 128, 2) == 2 * fn(1, 32, 1024, 8, 128, 2))
    check("linear in batch",
          lambda: fn(4, 32, 1024, 8, 128, 2) == 4 * fn(1, 32, 1024, 8, 128, 2))
    check("GQA with 8 kv heads is a quarter of 32",
          lambda: fn(1, 32, 1024, 8, 128, 2) * 4 == fn(1, 32, 1024, 32, 128, 2))
    check("bytes per element scales it",
          lambda: fn(1, 4, 16, 2, 8, 4) == 2 * fn(1, 4, 16, 2, 8, 2))
''',
),

task(
    id="arithmetic-intensity",
    title="Arithmetic intensity and the memory wall",
    chapter=CH,
    section="3.4 Arithmetic intensity: why decoding is memory-bound",
    level=2,
    entry="intensity",
    statement=(
        "Given FLOPs performed and bytes moved, return the arithmetic intensity "
        "and whether the operation is compute-bound on a device with a given "
        "ratio of peak FLOP/s to memory bandwidth. An operation is compute-bound "
        "when its intensity exceeds the device's ratio — below it, the arithmetic "
        "units idle while data arrives, which is why single-token decoding is "
        "memory-bound however large the matrices are."
    ),
    shapes=("flops · bytes float · device_ratio float (FLOP per byte)"
            "  ->  dict 'intensity', 'compute_bound' (bool)"),
    stub=("def intensity(flops, bytes_moved, device_ratio):\n"
          "    # -> {'intensity': float, 'compute_bound': bool}\n    pass\n"),
    hints=[
        "Intensity is FLOPs per byte moved.",
        "Compare it against the device ratio: above means compute-bound.",
        "Guard against zero bytes moved rather than dividing by zero.",
    ],
    solution=(
        "def intensity(flops, bytes_moved, device_ratio):\n"
        "    ai = flops / bytes_moved if bytes_moved else float('inf')\n"
        "    return {'intensity': ai, 'compute_bound': bool(ai >= device_ratio)}\n"
    ),
    solution_np=(
        "def intensity(flops, bytes_moved, device_ratio):\n"
        "    ai = flops / bytes_moved if bytes_moved else float('inf')\n"
        "    return {'intensity': ai, 'compute_bound': bool(ai >= device_ratio)}\n"
    ),
    traps=[
        "Inverting the ratio and calling memory-bound work compute-bound.",
        "Dividing by zero when nothing is moved.",
        "Treating a big matrix as automatically compute-bound — a batch-1 "
        "matrix-vector product reads the whole weight matrix for two FLOPs per "
        "element, so its intensity is about 1 regardless of size.",
    ],
    tests='''
def checks(fn, check):
    check("intensity is FLOPs per byte", lambda: abs(fn(1000., 100., 10.)["intensity"] - 10.) < 1e-9)
    check("above the ratio is compute-bound", lambda: fn(1000., 10., 10.)["compute_bound"] is True)
    check("below the ratio is memory-bound", lambda: fn(100., 100., 10.)["compute_bound"] is False)
    check("batch-1 decoding is memory-bound",
          lambda: fn(2 * 7e9, 7e9 * 2, 200.)["compute_bound"] is False)
    check("zero bytes does not divide by zero",
          lambda: math.isinf(fn(5., 0., 10.)["intensity"]))
''',
),

]
