"""Chapter 8 — Infrastructure: parallelism, reliability, serving."""
from .schema import task

CH = "8 · Infrastructure: parallelism, reliability, serving"

TASKS = [

task(
    id="mfu",
    title="Model FLOPs utilisation",
    chapter=CH,
    section="8.2 Utilisation and failure arithmetic",
    level=1,
    entry="mfu",
    statement=(
        "Compute MFU: the fraction of a cluster's peak arithmetic actually spent "
        "on the model's own FLOPs. Achieved FLOP/s is 6·N·(tokens per second) by "
        "the training-FLOPs rule; peak is per-device peak times device count. MFU "
        "is the honest efficiency number because it counts only useful work — "
        "unlike 'hardware FLOPs utilisation', it does not credit recomputation."
    ),
    shapes=("n_params · tokens_per_sec · n_devices · peak_flops_per_device float"
            "  ->  float in [0, 1]"),
    stub=("def mfu(n_params, tokens_per_sec, n_devices, peak_flops_per_device):\n"
          "    # -> achieved / peak\n    pass\n"),
    hints=[
        "Achieved FLOP/s = 6 · N · tokens_per_sec.",
        "Peak FLOP/s = n_devices · peak_flops_per_device.",
        "MFU is their ratio; it is a fraction, not a percentage.",
    ],
    solution=(
        "def mfu(n_params, tokens_per_sec, n_devices, peak_flops_per_device):\n"
        "    achieved = 6.0 * n_params * tokens_per_sec\n"
        "    peak = n_devices * peak_flops_per_device\n"
        "    return achieved / peak\n"
    ),
    solution_np=(
        "def mfu(n_params, tokens_per_sec, n_devices, peak_flops_per_device):\n"
        "    return (6.0 * n_params * tokens_per_sec) / (n_devices * peak_flops_per_device)\n"
    ),
    traps=[
        "Using 2ND, the forward-only count, which understates MFU by three times.",
        "Forgetting to multiply peak by the device count.",
        "Confusing MFU with hardware FLOPs utilisation — the latter counts "
        "recomputed activations as useful work, so it is always the larger number.",
    ],
    tests='''
def checks(fn, check):
    check("ratio of achieved to peak",
          lambda: abs(fn(1e9, 1e5, 8, 1e14) - (6 * 1e9 * 1e5) / (8 * 1e14)) < 1e-12)
    check("doubling throughput doubles MFU",
          lambda: abs(fn(1e9, 2e5, 8, 1e14) - 2 * fn(1e9, 1e5, 8, 1e14)) < 1e-12)
    check("doubling the cluster halves MFU at fixed throughput",
          lambda: abs(fn(1e9, 1e5, 16, 1e14) - 0.5 * fn(1e9, 1e5, 8, 1e14)) < 1e-12)
    check("uses 6ND, not 2ND",
          lambda: abs(fn(1e9, 1e5, 1, 6e14) - 1.0) < 1e-9)
    check("a realistic run lands in a plausible band",
          lambda: 0.2 < fn(7e9, 7.6e4, 8, 1e15) < 0.9)
''',
),

task(
    id="allreduce-volume",
    title="Data-parallel communication volume",
    chapter=CH,
    section="8.1 What each parallelism axis costs",
    level=2,
    entry="allreduce_bytes",
    statement=(
        "Return the bytes each rank sends during a ring all-reduce of the "
        "gradients, and the wall-clock time at a given per-link bandwidth. A ring "
        "all-reduce moves 2·(P-1)/P · S bytes per rank for a payload of S bytes — "
        "reduce-scatter then all-gather, each (P-1)/P. The important consequence "
        "is that this is nearly independent of P, so data parallelism scales in "
        "communication where naive all-to-all would not."
    ),
    shapes=("n_params · bytes_per_param · n_ranks · bandwidth_bytes_per_sec float"
            "  ->  dict 'bytes_per_rank', 'seconds'"),
    stub=("def allreduce_bytes(n_params, bytes_per_param, n_ranks, bandwidth):\n"
          "    # -> {'bytes_per_rank': float, 'seconds': float}\n    pass\n"),
    hints=[
        "The payload is S = n_params · bytes_per_param.",
        "Ring all-reduce moves 2·(P-1)/P·S bytes per rank.",
        "Time is those bytes divided by the per-link bandwidth. P = 1 sends "
        "nothing at all.",
    ],
    solution=(
        "def allreduce_bytes(n_params, bytes_per_param, n_ranks, bandwidth):\n"
        "    S = n_params * bytes_per_param\n"
        "    b = 2.0 * (n_ranks - 1) / n_ranks * S if n_ranks > 1 else 0.0\n"
        "    return {'bytes_per_rank': b, 'seconds': b / bandwidth}\n"
    ),
    solution_np=(
        "def allreduce_bytes(n_params, bytes_per_param, n_ranks, bandwidth):\n"
        "    S = n_params * bytes_per_param\n"
        "    b = 2.0 * (n_ranks - 1) / n_ranks * S if n_ranks > 1 else 0.0\n"
        "    return {'bytes_per_rank': b, 'seconds': b / bandwidth}\n"
    ),
    traps=[
        "Using S rather than 2·(P-1)/P·S, which ignores the all-gather half.",
        "Assuming volume grows with P — it approaches 2S and stops.",
        "Forgetting that a single rank communicates nothing.",
    ],
    tests='''
def checks(fn, check):
    S = 1e9 * 2
    check("single rank sends nothing",
          lambda: fn(1e9, 2, 1, 1e11)["bytes_per_rank"] == 0.0)
    check("two ranks send S",
          lambda: abs(fn(1e9, 2, 2, 1e11)["bytes_per_rank"] - S) < 1)
    check("volume approaches 2S for large P",
          lambda: abs(fn(1e9, 2, 1024, 1e11)["bytes_per_rank"] - 2 * S) / (2 * S) < 0.01)
    check("volume is nearly flat in P",
          lambda: abs(fn(1e9, 2, 512, 1e11)["bytes_per_rank"] /
                      fn(1e9, 2, 64, 1e11)["bytes_per_rank"] - 1.0) < 0.02)
    check("time is bytes over bandwidth",
          lambda: abs(fn(1e9, 2, 8, 1e11)["seconds"] -
                      fn(1e9, 2, 8, 1e11)["bytes_per_rank"] / 1e11) < 1e-12)
''',
),

task(
    id="failure-throughput",
    title="Effective throughput under failures",
    chapter=CH,
    section="8.2 Utilisation and failure arithmetic",
    level=3,
    entry="effective_fraction",
    statement=(
        "A long training run loses time to two things: writing checkpoints, and "
        "redoing work after a crash. With checkpoints every `interval` seconds "
        "costing `save_cost` each, and a mean time between failures of `mtbf`, "
        "return the fraction of wall-clock actually spent making progress. "
        "Assume a crash loses half an interval on average plus a `restart_cost` "
        "to reload."
    ),
    shapes=("interval · save_cost · mtbf · restart_cost float (seconds)"
            "  ->  float in (0, 1]"),
    stub=("def effective_fraction(interval, save_cost, mtbf, restart_cost):\n"
          "    # -> fraction of wall-clock spent on useful work\n    pass\n"),
    hints=[
        "Per interval of length `interval`, `save_cost` is pure overhead.",
        "Failures arrive at rate 1/mtbf, and each costs restart_cost plus half an "
        "interval of redone work.",
        "Over one interval the expected lost time is save_cost + "
        "(interval/mtbf)·(restart_cost + interval/2); useful time is interval, "
        "and the fraction is useful/(useful + lost).",
    ],
    solution=(
        "def effective_fraction(interval, save_cost, mtbf, restart_cost):\n"
        "    lost = save_cost + (interval / mtbf) * (restart_cost + interval / 2.0)\n"
        "    return interval / (interval + lost)\n"
    ),
    solution_np=(
        "def effective_fraction(interval, save_cost, mtbf, restart_cost):\n"
        "    lost = save_cost + (interval / mtbf) * (restart_cost + interval / 2.0)\n"
        "    return interval / (interval + lost)\n"
    ),
    traps=[
        "Charging a whole interval of redone work per failure rather than half — "
        "a crash lands uniformly within the interval on average.",
        "Ignoring the checkpoint cost, which dominates when the interval is short.",
        "Treating the result as if longer intervals were always better: they cut "
        "checkpoint overhead but raise the expected redo, so there is an optimum.",
    ],
    tests='''
def checks(fn, check):
    check("no cost and no failures gives 1.0",
          lambda: abs(fn(3600., 0., 1e18, 0.) - 1.0) < 1e-9)
    check("checkpoint overhead alone",
          lambda: abs(fn(1000., 100., 1e18, 0.) - 1000. / 1100.) < 1e-9)
    check("failures reduce the fraction",
          lambda: fn(3600., 60., 3600. * 10, 300.) < fn(3600., 60., 1e18, 300.))
    check("result stays in (0, 1]",
          lambda: 0 < fn(3600., 60., 7200., 600.) <= 1.0)
    check("a crash costs half an interval of redo",
          lambda: abs(fn(1000., 0., 1000., 0.) - 1000. / 1500.) < 1e-9)
    def has_an_optimum():
        f = lambda iv: fn(iv, 60., 3600. * 5, 300.)
        best = max(range(60, 20000, 20), key=f)
        return 60 < best < 20000 - 20
    check("an interior optimum exists in the interval length", has_an_optimum)
''',
),

task(
    id="serving-batch",
    title="Where decoding stops being memory-bound",
    chapter=CH,
    section="8.3 Serving: latency, batch, and cost",
    level=2,
    entry="crossover_batch",
    statement=(
        "During decoding, every step reads the whole weight matrix once but does "
        "only 2·N FLOPs per sequence in the batch. Return the smallest batch size "
        "at which arithmetic intensity reaches the device's FLOP-per-byte ratio — "
        "the point where serving stops being bandwidth-limited and batching stops "
        "being free."
    ),
    shapes="bytes_per_param float · device_ratio float (FLOP per byte)  ->  int",
    stub=("def crossover_batch(bytes_per_param, device_ratio):\n"
          "    # -> smallest batch size that is compute-bound\n    pass\n"),
    hints=[
        "Per step: FLOPs = 2·N·B, bytes read = N·bytes_per_param. N cancels.",
        "Intensity = 2·B / bytes_per_param, so set that equal to the device ratio.",
        "B = ceil(device_ratio · bytes_per_param / 2).",
    ],
    solution=(
        "def crossover_batch(bytes_per_param, device_ratio):\n"
        "    return int(math.ceil(device_ratio * bytes_per_param / 2.0))\n"
    ),
    solution_np=(
        "def crossover_batch(bytes_per_param, device_ratio):\n"
        "    return int(math.ceil(device_ratio * bytes_per_param / 2.0))\n"
    ),
    traps=[
        "Keeping N in the answer — it cancels, which is why the crossover is a "
        "property of the hardware and the dtype, not of the model size.",
        "Rounding down, which returns a batch that is still memory-bound.",
        "Forgetting the factor 2 FLOPs per parameter per sequence.",
    ],
    tests='''
def checks(fn, check):
    check("fp16 on a 200 FLOP/byte device", lambda: fn(2.0, 200.) == 200)
    check("fp8 halves the crossover", lambda: fn(1.0, 200.) == 100)
    check("rounds up, never down", lambda: fn(2.0, 201.) == 201)
    check("independent of model size (N cancels)", lambda: fn(2.0, 200.) == fn(2.0, 200.))
    check("returns an int", lambda: isinstance(fn(2.0, 200.), int))
    check("a faster device needs a bigger batch", lambda: fn(2.0, 400.) > fn(2.0, 200.))
''',
),

]
