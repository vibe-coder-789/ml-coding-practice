"""build-gpt2 — one model, assembled in order.

Unlike the other volumes, these steps compose: step 4 builds the block that
step 5 stacks, and step 8 trains the model step 6 defined. Each step is still
checked in isolation, so a wrong answer at step 3 does not hide behind step 7 —
but the reference solution for every later step is written against the earlier
ones, and the final steps actually train the thing and require the loss to fall.

All steps are PyTorch-only: they build nn.Module subclasses and use autograd.
"""
from .schema import task

BOOK = "build-gpt2"
CH = "Build a GPT-2 class model"

# Every step is checked against this reference stack, injected before the
# candidate's own code so a later step can lean on the earlier components
# without requiring the candidate to have solved them first.
REF = '''
import torch.nn as nn

CFG = dict(vocab=65, d_model=64, n_heads=4, n_layers=3, block=32, dropout=0.0)


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block, block)).bool())

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.tril[:L, :L], torch.finfo(att.dtype).min)
        y = torch.softmax(att, -1) @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class _MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, block)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class _GPT(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, block):
        super().__init__()
        self.block_size = block
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, block)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        _init_gpt2(self, n_layers)

    def forward(self, idx):
        B, L = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def _init_gpt2(model, n_layers):
    """GPT-2's scheme: N(0, 0.02), zero biases, residual projections scaled."""
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
    for name, p in model.named_parameters():
        if name.endswith("proj.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
    return model


def _tiny_corpus(n=2048):
    """A deterministic, learnable toy sequence: period-7 repeating ids."""
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, 65, (7,), generator=g)
    return base.repeat(n // 7 + 1)[:n]
'''

TASKS = [

task(
    id="gpt-embeddings",
    title="Step 1 · Token and position embeddings",
    book=BOOK, chapter=CH, section="Step 1 · Embeddings",
    level=1,
    entry="Embeddings",
    statement=(
        "Build the input layer: an nn.Module holding a token embedding of shape "
        "(vocab, d_model) and a learned position embedding of shape (block, "
        "d_model), whose forward adds them. GPT-2 learns positions rather than "
        "fixing them, and they are added — not concatenated — so the residual "
        "stream keeps one width throughout."
    ),
    shapes="__init__(vocab, d_model, block) · forward(idx (B, L) int64) -> (B, L, d_model)",
    stub=("class Embeddings(nn.Module):\n"
          "    def __init__(self, vocab, d_model, block):\n"
          "        super().__init__()\n"
          "        # self.tok = ...  self.pos = ...\n"
          "\n"
          "    def forward(self, idx):\n"
          "        # idx (B, L) -> (B, L, d_model)\n"
          "        pass\n"),
    hints=[
        "nn.Embedding(num_embeddings, embedding_dim) is a lookup table.",
        "Positions are 0..L-1, the same for every sequence in the batch: "
        "torch.arange(L).",
        "Add the two; broadcasting handles the missing batch axis on the "
        "position term.",
    ],
    solution=(
        "class Embeddings(nn.Module):\n"
        "    def __init__(self, vocab, d_model, block):\n"
        "        super().__init__()\n"
        "        self.tok = nn.Embedding(vocab, d_model)\n"
        "        self.pos = nn.Embedding(block, d_model)\n"
        "\n"
        "    def forward(self, idx):\n"
        "        B, L = idx.shape\n"
        "        return self.tok(idx) + self.pos(torch.arange(L, device=idx.device))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Concatenating instead of adding, which doubles the width.",
        "Indexing positions per batch element rather than once — the position "
        "vector is shared.",
        "Sizing the position table by sequence length instead of block size, so "
        "it cannot handle a full-length context later.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    m = fn(65, 64, 32)
    idx = torch.randint(0, 65, (2, 7))
    check("output shape", lambda: shape(m(idx)) == (2, 7, 64))
    check("has a token table of the right size",
          lambda: any(p.shape == (65, 64) for p in m.parameters()))
    check("has a position table of block size",
          lambda: any(p.shape == (32, 64) for p in m.parameters()))
    def positions_differ():
        same = torch.zeros(1, 4, dtype=torch.long)
        out = m(same)[0]
        return not close(out[0], out[1], 1e-6)
    check("the same token at different positions differs", positions_differ)
    def batch_independent():
        a = m(torch.tensor([[1, 2, 3]]))
        b = m(torch.tensor([[1, 2, 3], [4, 5, 6]]))
        return close(a[0], b[0], 1e-5)
    check("batch elements are independent", batch_independent)
    check("gradients reach both tables",
          lambda: (m(idx).sum().backward(),
                   all(p.grad is not None for p in m.parameters()))[-1])
''',
),

task(
    id="gpt-attention",
    title="Step 2 · Causal self-attention",
    book=BOOK, chapter=CH, section="Step 2 · Attention",
    level=3,
    entry="CausalSelfAttention",
    statement=(
        "Build the attention module: one fused qkv projection of width 3·d_model, "
        "split into heads, scaled dot-product with a causal mask, merged back, "
        "then an output projection. Register the mask as a buffer so it moves with "
        "the module and is not treated as a parameter."
    ),
    shapes=("__init__(d_model, n_heads, block) · forward(x (B, L, D)) -> (B, L, D)"
            " · name the projections self.qkv and self.proj — the tests copy your weights by those names"),
    stub=("class CausalSelfAttention(nn.Module):\n"
          "    def __init__(self, d_model, n_heads, block):\n"
          "        super().__init__()\n"
          "        # self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)\n"
          "        # self.proj = ...   register a causal mask buffer\n"
          "\n"
          "    def forward(self, x):\n"
          "        pass\n"),
    hints=[
        "One Linear to 3·d_model, then .chunk(3, dim=-1) gives q, k, v.",
        "Reshape each to (B, L, n_heads, d_head) then transpose(1, 2). Scale by "
        "1/sqrt(d_head), not 1/sqrt(d_model).",
        "Mask with torch.finfo(dtype).min on the strict upper triangle, softmax, "
        "weight v, then transpose back and .contiguous().view() before the output "
        "projection.",
    ],
    solution=(
        "class CausalSelfAttention(nn.Module):\n"
        "    def __init__(self, d_model, n_heads, block):\n"
        "        super().__init__()\n"
        "        self.n_heads, self.d_head = n_heads, d_model // n_heads\n"
        "        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)\n"
        "        self.proj = nn.Linear(d_model, d_model, bias=False)\n"
        "        self.register_buffer('tril', torch.tril(torch.ones(block, block)).bool())\n"
        "\n"
        "    def forward(self, x):\n"
        "        B, L, D = x.shape\n"
        "        q, k, v = self.qkv(x).chunk(3, dim=-1)\n"
        "        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)\n"
        "        att = att.masked_fill(~self.tril[:L, :L], torch.finfo(att.dtype).min)\n"
        "        y = torch.softmax(att, -1) @ v\n"
        "        y = y.transpose(1, 2).contiguous().view(B, L, D)\n"
        "        return self.proj(y)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Scaling by 1/sqrt(d_model) rather than per-head 1/sqrt(d_head).",
        "Slicing the mask to [:L, :L] — forgetting this breaks any sequence "
        "shorter than the block size.",
        "Making the mask a plain attribute instead of a buffer, so .to(device) "
        "leaves it behind.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = fn(64, 4, 32)
    x = torch.randn(2, 7, 64)
    check("output shape", lambda: shape(m(x)) == (2, 7, 64))
    def causal():
        y1 = m(x)
        x2 = x.clone(); x2[:, -1] += 5.0
        return close(y1[:, :-1], m(x2)[:, :-1], 1e-4)
    check("a later token cannot change an earlier output", causal)
    check("handles a sequence shorter than the block",
          lambda: shape(m(torch.randn(1, 3, 64))) == (1, 3, 64))
    check("the mask is a buffer, not a parameter",
          lambda: any(b.dtype == torch.bool or b.numel() == 32 * 32
                      for b in m.buffers()))
    def per_head_scale():
        # compare against a reference built from the same weights
        ref = _CausalSelfAttention(64, 4, 32)
        ref.load_state_dict(m.state_dict(), strict=False)
        return close(m(x), ref(x), 1e-4)
    check("matches the reference implementation given the same weights", per_head_scale)
    check("gradients flow", lambda: (m(x).sum().backward(),
                                     all(p.grad is not None for p in m.parameters()))[-1])
''',
),

task(
    id="gpt-mlp",
    title="Step 3 · The feed-forward block",
    book=BOOK, chapter=CH, section="Step 3 · MLP",
    level=1,
    entry="MLP",
    statement=(
        "Build GPT-2's position-wise feed-forward: project up by 4×, apply GELU, "
        "project back down. The 4× expansion is the convention the parameter count "
        "assumes, and GELU rather than ReLU is what the released weights were "
        "trained with."
    ),
    shapes=("__init__(d_model) · forward(x (..., D)) -> (..., D)"
            " · name the layers self.fc and self.proj — the tests copy your weights by those names"),
    stub=("class MLP(nn.Module):\n"
          "    def __init__(self, d_model):\n"
          "        super().__init__()\n"
          "        # up to 4*d_model, back down\n"
          "\n"
          "    def forward(self, x):\n"
          "        pass\n"),
    hints=[
        "Two nn.Linear layers: d_model -> 4*d_model and 4*d_model -> d_model.",
        "F.gelu between them.",
        "Biases are kept in GPT-2's MLP, unlike the attention projections here.",
    ],
    solution=(
        "class MLP(nn.Module):\n"
        "    def __init__(self, d_model):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(d_model, 4 * d_model)\n"
        "        self.proj = nn.Linear(4 * d_model, d_model)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.proj(F.gelu(self.fc(x)))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Using ReLU, which changes the function and will not match a reference.",
        "Applying the nonlinearity after the down-projection instead of between.",
        "Expanding by something other than 4×, which breaks the parameter budget.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    m = fn(64)
    x = torch.randn(2, 5, 64)
    check("preserves shape", lambda: shape(m(x)) == (2, 5, 64))
    check("hidden width is 4x",
          lambda: any(tuple(p.shape) == (256, 64) for p in m.parameters()))
    check("projects back down",
          lambda: any(tuple(p.shape) == (64, 256) for p in m.parameters()))
    check("is not linear (a nonlinearity is present)",
          lambda: not close(m(2 * x), 2 * m(x) - m(torch.zeros_like(x)), 1e-3))
    def uses_gelu():
        ref = _MLP(64); ref.load_state_dict(m.state_dict(), strict=False)
        return close(m(x), ref(x), 1e-5)
    check("matches a GELU reference given the same weights", uses_gelu)
    check("applies elementwise over positions",
          lambda: close(m(x)[0, 0], m(x[:, :1])[0, 0], 1e-5))
''',
),

task(
    id="gpt-block",
    title="Step 4 · A transformer block",
    book=BOOK, chapter=CH, section="Step 4 · Block",
    level=2,
    entry="Block",
    statement=(
        "Assemble one pre-norm transformer block: x + attn(ln1(x)), then "
        "x + mlp(ln2(x)). Pre-norm — normalising the input to each sublayer rather "
        "than the sum — is what makes deep stacks trainable without a warmup "
        "schedule, because the residual path stays an unmodified identity."
    ),
    shapes=("__init__(d_model, n_heads, block) · forward(x (B, L, D)) -> (B, L, D)"
            " · name the parts self.ln1 / self.attn / self.ln2 / self.mlp — the tests copy weights by name"),
    stub=("class Block(nn.Module):\n"
          "    def __init__(self, d_model, n_heads, block):\n"
          "        super().__init__()\n"
          "        # ln1, attn, ln2, mlp\n"
          "\n"
          "    def forward(self, x):\n"
          "        pass\n"),
    hints=[
        "Two nn.LayerNorm(d_model), one attention, one MLP.",
        "The residual is added to the sublayer's output, and the norm is applied "
        "to the sublayer's input only.",
        "You may use the provided _CausalSelfAttention and _MLP.",
    ],
    solution=(
        "class Block(nn.Module):\n"
        "    def __init__(self, d_model, n_heads, block):\n"
        "        super().__init__()\n"
        "        self.ln1 = nn.LayerNorm(d_model)\n"
        "        self.attn = _CausalSelfAttention(d_model, n_heads, block)\n"
        "        self.ln2 = nn.LayerNorm(d_model)\n"
        "        self.mlp = _MLP(d_model)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = x + self.attn(self.ln1(x))\n"
        "        return x + self.mlp(self.ln2(x))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Post-norm — ln(x + sublayer(x)) — which is the original 2017 design and "
        "needs a warmup to train at depth.",
        "Sharing one LayerNorm between both sublayers.",
        "Dropping the residual on one of the two paths, which usually still trains "
        "but far worse.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = fn(64, 4, 32)
    x = torch.randn(2, 6, 64)
    check("preserves shape", lambda: shape(m(x)) == (2, 6, 64))
    check("is causal",
          lambda: close(m(x)[:, :-1],
                        m(torch.cat([x[:, :-1], x[:, -1:] + 9.0], 1))[:, :-1], 1e-4))
    def is_prenorm():
        ref = _Block(64, 4, 32); ref.load_state_dict(m.state_dict(), strict=False)
        return close(m(x), ref(x), 1e-4)
    check("matches a pre-norm reference given the same weights", is_prenorm)
    def residual_present():
        # zeroing both sublayer output projections must leave x unchanged
        import copy
        z = copy.deepcopy(m)
        with torch.no_grad():
            for n_, p in z.named_parameters():
                if "proj" in n_:
                    p.zero_()
        return close(z(x), x, 1e-4)
    check("the residual path is an identity when sublayers output zero",
          residual_present)
    check("has two separate LayerNorms",
          lambda: sum(1 for mod in m.modules() if isinstance(mod, nn.LayerNorm)) == 2)
''',
),

task(
    id="gpt-model",
    title="Step 5 · The full model",
    book=BOOK, chapter=CH, section="Step 5 · GPT",
    level=2,
    entry="GPT",
    statement=(
        "Stack it into a model: embeddings, n_layers blocks, a final LayerNorm, "
        "and a bias-free language-model head. Tie the head's weight to the token "
        "embedding — GPT-2 does, and on a small model the embedding table is most "
        "of the parameters, so tying roughly halves them."
    ),
    shapes="__init__(vocab, d_model, n_heads, n_layers, block) · forward(idx (B,L)) -> (B, L, vocab)",
    stub=("class GPT(nn.Module):\n"
          "    def __init__(self, vocab, d_model, n_heads, n_layers, block):\n"
          "        super().__init__()\n"
          "        # tok, pos, blocks (nn.ModuleList), ln_f, head — and tie the head\n"
          "\n"
          "    def forward(self, idx):\n"
          "        pass\n"),
    hints=[
        "Hold the blocks in an nn.ModuleList so their parameters register.",
        "Weight tying is an assignment: self.head.weight = self.tok.weight.",
        "The final LayerNorm goes before the head, not after.",
    ],
    solution=(
        "class GPT(nn.Module):\n"
        "    def __init__(self, vocab, d_model, n_heads, n_layers, block):\n"
        "        super().__init__()\n"
        "        self.block_size = block\n"
        "        self.tok = nn.Embedding(vocab, d_model)\n"
        "        self.pos = nn.Embedding(block, d_model)\n"
        "        self.blocks = nn.ModuleList([_Block(d_model, n_heads, block)\n"
        "                                     for _ in range(n_layers)])\n"
        "        self.ln_f = nn.LayerNorm(d_model)\n"
        "        self.head = nn.Linear(d_model, vocab, bias=False)\n"
        "        self.head.weight = self.tok.weight\n"
        "\n"
        "    def forward(self, idx):\n"
        "        B, L = idx.shape\n"
        "        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))\n"
        "        for b in self.blocks:\n"
        "            x = b(x)\n"
        "        return self.head(self.ln_f(x))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Using a plain Python list for the blocks, so their parameters never "
        "reach the optimiser and silently never train.",
        "Copying the embedding weight instead of assigning it, which unties them "
        "after the first update.",
        "Putting the final norm after the head.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = fn(65, 64, 4, 3, 32)
    idx = torch.randint(0, 65, (2, 9))
    check("logits shape", lambda: shape(m(idx)) == (2, 9, 65))
    check("blocks are registered (parameter count is plausible)",
          lambda: sum(p.numel() for p in m.parameters()) > 3 * 4 * 64 * 64)
    check("weights are tied",
          lambda: any(p is q for p in [m.head.weight] for q in [m.tok.weight]))
    check("is causal",
          lambda: close(m(idx)[:, :-1],
                        m(torch.cat([idx[:, :-1], (idx[:, -1:] + 1) % 65], 1))[:, :-1], 1e-4))
    check("every parameter receives a gradient",
          lambda: (m(idx).sum().backward(),
                   all(p.grad is not None for p in m.parameters()))[-1])
    check("handles a full-length context",
          lambda: shape(m(torch.randint(0, 65, (1, 32)))) == (1, 32, 65))
''',
),

task(
    id="gpt-init",
    title="Step 6 · Initialisation",
    book=BOOK, chapter=CH, section="Step 6 · Initialisation",
    level=2,
    entry="init_weights",
    statement=(
        "Apply GPT-2's initialisation in place and return the model: every Linear "
        "and Embedding weight drawn from N(0, 0.02), every Linear bias zeroed, and "
        "every residual output projection — any parameter whose name ends "
        "'proj.weight' — further scaled to std 0.02/sqrt(2·n_layers). "
        "This is not cosmetic. With PyTorch's defaults this model starts at a loss "
        "of about 23; log(vocab) is 4.17, which is what an untrained model that "
        "merely knows nothing should score. The extra 19 nats are the network "
        "shouting confident nonsense, and it takes hundreds of steps to undo."
    ),
    shapes="model nn.Module · n_layers int  ->  the same model, initialised in place",
    stub=("def init_weights(model, n_layers):\n"
          "    # N(0, 0.02) weights, zero biases, scaled residual projections\n"
          "    return model\n"),
    hints=[
        "Iterate model.modules() and branch on isinstance(mod, nn.Linear) and "
        "nn.Embedding.",
        "nn.init.normal_(w, mean=0.0, std=0.02) and nn.init.zeros_(b) do the work "
        "in place.",
        "Then loop model.named_parameters() and re-initialise those ending in "
        "'proj.weight' with std 0.02/sqrt(2*n_layers) — this keeps the variance of "
        "the residual stream from growing with depth.",
    ],
    solution=(
        "def init_weights(model, n_layers):\n"
        "    for mod in model.modules():\n"
        "        if isinstance(mod, nn.Linear):\n"
        "            nn.init.normal_(mod.weight, mean=0.0, std=0.02)\n"
        "            if mod.bias is not None:\n"
        "                nn.init.zeros_(mod.bias)\n"
        "        elif isinstance(mod, nn.Embedding):\n"
        "            nn.init.normal_(mod.weight, mean=0.0, std=0.02)\n"
        "    for name, p in model.named_parameters():\n"
        "        if name.endswith('proj.weight'):\n"
        "            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))\n"
        "    return model\n"
    ),
    frameworks=["torch"],
    traps=[
        "Leaving PyTorch's defaults, which start the loss five times too high.",
        "Initialising with a non-in-place call and discarding the result.",
        "Skipping the residual scaling, which lets the stream's variance grow "
        "linearly with depth — harmless at 3 layers, not at 48.",
    ],
    extra=REF,
    tests="""
def checks(fn, check):
    torch.manual_seed(0)
    V, D, H, NL = 65, 64, 4, 3

    def fresh():
        # rebuild the PyTorch-default model: reset every module, then restore the
        # weight tying the constructor applies (resetting the head would otherwise
        # overwrite the shared embedding with Linear's much smaller init)
        m = _GPT(V, D, H, NL, 32)
        for mod in m.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                mod.reset_parameters()
        # tok and head share one tensor, and modules() reaches head last, so the
        # head's reset just overwrote the embedding. Redraw it the way
        # nn.Embedding does, then re-tie — that is the real constructor default.
        with torch.no_grad():
            m.tok.weight.normal_(0.0, 1.0)
        m.head.weight = m.tok.weight
        return m

    m = fn(fresh(), NL)
    check("returns the model", lambda: isinstance(m, nn.Module))
    check("embedding std is about 0.02",
          lambda: abs(float(m.tok.weight.std()) - 0.02) < 0.004)
    check("linear biases are zeroed",
          lambda: all(float(mod.bias.abs().max()) == 0.0
                      for mod in m.modules()
                      if isinstance(mod, nn.Linear) and mod.bias is not None))
    def residual_scaled():
        want = 0.02 / math.sqrt(2 * NL)
        stds = [float(p.std()) for n, p in m.named_parameters()
                if n.endswith('proj.weight')]
        return len(stds) > 0 and all(abs(s - want) < 0.004 for s in stds)
    check("residual projections use the depth-scaled std", residual_scaled)

    def loss_near_log_v():
        mm = fn(fresh(), NL)
        b = torch.randint(0, V, (4, 16))
        lg = mm(b)
        l = F.cross_entropy(lg[:, :-1].reshape(-1, V), b[:, 1:].reshape(-1))
        return abs(float(l.detach()) - math.log(V)) < 0.3
    check("initial loss lands at log(vocab), not far above it", loss_near_log_v)

    def beats_default():
        d = fresh()
        b = torch.randint(0, V, (4, 16))
        lg = d(b)
        default_loss = float(F.cross_entropy(
            lg[:, :-1].reshape(-1, V), b[:, 1:].reshape(-1)).detach())
        return default_loss > math.log(V) + 2
    check("the PyTorch default really is much worse", beats_default)
"""[1:],
),

task(
    id="gpt-loss",
    title="Step 7 · The shifted loss",
    book=BOOK, chapter=CH, section="Step 7 · Loss",
    level=2,
    entry="lm_loss",
    statement=(
        "Compute the language-modelling loss from logits and the input ids: "
        "position t predicts token t+1, so the logits lose their last position and "
        "the targets lose their first. Off-by-one here is the classic bug — it "
        "still trains, and the model learns to copy its input instead of "
        "predicting."
    ),
    shapes="logits (B, L, V) · idx (B, L) int64  ->  scalar mean cross-entropy",
    stub=("def lm_loss(logits, idx):\n"
          "    # position t predicts token t+1\n    pass\n"),
    hints=[
        "Drop the last position of the logits, and the first of the targets.",
        "Flatten both to (B*(L-1), V) and (B*(L-1),) before F.cross_entropy.",
        "F.cross_entropy takes logits, never probabilities.",
    ],
    solution=(
        "def lm_loss(logits, idx):\n"
        "    pred = logits[:, :-1].reshape(-1, logits.shape[-1])\n"
        "    tgt = idx[:, 1:].reshape(-1)\n"
        "    return F.cross_entropy(pred, tgt)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Not shifting at all, which trains the model to reproduce its input.",
        "Shifting the wrong way, so the model predicts the previous token.",
        "Averaging per sequence and then across the batch, which weights short "
        "sequences more heavily.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, L, V = 2, 6, 65
    logits = torch.randn(B, L, V)
    idx = torch.randint(0, V, (B, L))
    check("returns a scalar", lambda: fn(logits, idx).ndim == 0)
    check("matches the explicit shifted form",
          lambda: close(fn(logits, idx),
                        F.cross_entropy(logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1)), 1e-5))
    check("random logits give about log V",
          lambda: abs(float(fn(logits, idx)) - math.log(V)) < 0.6)
    def is_shifted():
        # perfect next-token logits should give ~0 loss
        perfect = torch.zeros(1, 4, V)
        ids = torch.tensor([[1, 2, 3, 4]])
        for t in range(3):
            perfect[0, t, ids[0, t + 1]] = 50.0
        return float(fn(perfect, ids)) < 1e-3
    check("perfect next-token prediction gives ~0 loss", is_shifted)
    def not_copying():
        # logits that predict the CURRENT token must score badly
        copycat = torch.zeros(1, 4, V)
        ids = torch.tensor([[1, 2, 3, 4]])
        for t in range(4):
            copycat[0, t, ids[0, t]] = 50.0
        return float(fn(copycat, ids)) > 10
    check("predicting the current token is heavily penalised", not_copying)
''',
),

task(
    id="gpt-train-step",
    title="Step 8 · One training step",
    book=BOOK, chapter=CH, section="Step 8 · Optimisation",
    level=2,
    entry="train_step",
    statement=(
        "Perform one optimisation step: forward, loss, zero the gradients, "
        "backward, clip the global gradient norm to 1.0, then step. Return the "
        "loss as a float. Order matters — zeroing after the backward wipes the "
        "gradients you just computed, and clipping after the step does nothing at "
        "all."
    ),
    shapes="model · opt · batch (B, L) int64  ->  float loss",
    stub=("def train_step(model, opt, batch):\n"
          "    # forward, loss, zero, backward, clip to 1.0, step -> float\n    pass\n"),
    hints=[
        "The loss is the shifted cross-entropy from step 7.",
        "opt.zero_grad() before backward; torch.nn.utils.clip_grad_norm_ after it "
        "and before opt.step().",
        "Return float(loss) — returning the tensor keeps the graph alive across "
        "steps and leaks memory.",
    ],
    solution=(
        "def train_step(model, opt, batch):\n"
        "    logits = model(batch)\n"
        "    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),\n"
        "                           batch[:, 1:].reshape(-1))\n"
        "    opt.zero_grad()\n"
        "    loss.backward()\n"
        "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "    opt.step()\n"
        "    return float(loss)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Calling zero_grad after backward, which discards the gradients.",
        "Clipping after opt.step(), which has no effect on the update just taken.",
        "Returning the loss tensor rather than a float, retaining the graph.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = _GPT(65, 32, 4, 2, 32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = torch.randint(0, 65, (2, 16))
    with torch.no_grad():
        _lg = m(batch)
        expected0 = float(F.cross_entropy(_lg[:, :-1].reshape(-1, 65),
                                          batch[:, 1:].reshape(-1)))
    l0 = fn(m, opt, batch)
    check("returns a float", lambda: isinstance(l0, float))
    check("initial loss is about log V", lambda: abs(l0 - math.log(65)) < 1.0)
    check("the loss is the correctly shifted one (t predicts t+1)",
          lambda: abs(l0 - expected0) < 1e-4)
    def parameters_moved():
        before = [p.detach().clone() for p in m.parameters()]
        fn(m, opt, batch)
        return any(not close(a, b, 1e-9) for a, b in zip(before, m.parameters()))
    check("parameters actually change", parameters_moved)
    def gradients_were_cleared():
        # after the step, a fresh backward must not accumulate two steps' worth
        fn(m, opt, batch)
        g1 = [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]
        fn(m, opt, batch)
        g2 = [p.grad for p in m.parameters() if p.grad is not None]
        return not all(close(a, 2 * b, 1e-6) for a, b in zip(g2, g1))
    check("gradients are zeroed, not accumulated", gradients_were_cleared)
    def clipping_applied():
        m2 = _GPT(65, 32, 4, 2, 32)
        o2 = torch.optim.SGD(m2.parameters(), lr=1e6)   # absurd lr
        fn(m2, o2, batch)
        return all(torch.isfinite(p).all() for p in m2.parameters())
    check("clipping keeps an absurd learning rate from producing NaN", clipping_applied)
''',
),

task(
    id="gpt-overfit",
    title="Step 9 · Prove it learns",
    book=BOOK, chapter=CH, section="Step 9 · Training",
    level=3,
    entry="overfit",
    statement=(
        "Train the model on a single batch until it memorises it, and return the "
        "list of losses. Overfitting one batch is the first thing to do with any "
        "new training loop: if the loss will not go to near zero on data the model "
        "has capacity to memorise, the bug is in the code, not the "
        "hyperparameters."
    ),
    shapes="model · batch (B, L) · steps int · lr float  ->  list of float losses",
    stub=("def overfit(model, batch, steps=200, lr=1e-3):\n"
          "    # -> list of losses, one per step\n    pass\n"),
    hints=[
        "Build one AdamW over model.parameters() outside the loop — creating it "
        "inside resets the optimiser state every step.",
        "Each iteration is the training step from step 7, on the same batch.",
        "Collect float(loss) per step and return the list.",
    ],
    solution=(
        "def overfit(model, batch, steps=200, lr=1e-3):\n"
        "    opt = torch.optim.AdamW(model.parameters(), lr=lr)\n"
        "    losses = []\n"
        "    for _ in range(steps):\n"
        "        logits = model(batch)\n"
        "        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),\n"
        "                               batch[:, 1:].reshape(-1))\n"
        "        opt.zero_grad()\n"
        "        loss.backward()\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "        opt.step()\n"
        "        losses.append(float(loss))\n"
        "    return losses\n"
    ),
    frameworks=["torch"],
    traps=[
        "Constructing the optimiser inside the loop, which throws away Adam's "
        "moments every step and barely learns.",
        "Forgetting zero_grad, so gradients accumulate and the effective step "
        "grows without bound.",
        "Calling model.eval() or wrapping in no_grad, which stops learning "
        "entirely.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = _GPT(65, 64, 4, 2, 32)
    batch = _tiny_corpus(64).view(2, 32)
    with torch.no_grad():
        _lg = m(batch)
        expected0 = float(F.cross_entropy(_lg[:, :-1].reshape(-1, 65),
                                          batch[:, 1:].reshape(-1)))
    losses = fn(m, batch, steps=150, lr=3e-3)
    check("returns one loss per step", lambda: len(losses) == 150)
    check("all floats", lambda: all(isinstance(v, float) for v in losses))
    check("starts near log V", lambda: abs(losses[0] - math.log(65)) < 1.2)
    check("the first loss is the correctly shifted one",
          lambda: abs(losses[0] - expected0) < 1e-3)
    check("the loss falls substantially", lambda: losses[-1] < losses[0] * 0.5)
    check("memorises a single batch (final loss is small)", lambda: losses[-1] < 0.3)
    check("no NaN during training", lambda: all(v == v for v in losses))
    check("progress is broadly monotone",
          lambda: sum(losses[:10]) / 10 > sum(losses[-10:]) / 10)
''',
),

task(
    id="gpt-generate",
    title="Step 10 · Generate",
    book=BOOK, chapter=CH, section="Step 10 · Sampling",
    level=3,
    entry="generate",
    statement=(
        "Autoregressively extend a prompt by n tokens. Each step: take the logits "
        "at the last position only, apply temperature, optionally restrict to the "
        "top-k, sample, and append. Crop the context to the model's block size — "
        "the learned position table has no entry beyond it, and indexing past it "
        "raises."
    ),
    shapes="model · idx (B, L0) · n int · temp float · top_k int|None  ->  (B, L0+n)",
    stub=("def generate(model, idx, n, temp=1.0, top_k=None):\n"
          "    # -> the prompt with n sampled tokens appended\n    pass\n"),
    hints=[
        "Wrap the loop in torch.no_grad() — nothing here needs a graph.",
        "Crop with idx[:, -model.block_size:] before each forward.",
        "logits[:, -1] is the only position that matters; divide by temp, apply "
        "top-k, softmax, then torch.multinomial.",
    ],
    solution=(
        "def generate(model, idx, n, temp=1.0, top_k=None):\n"
        "    for _ in range(n):\n"
        "        with torch.no_grad():\n"
        "            logits = model(idx[:, -model.block_size:])[:, -1] / temp\n"
        "        if top_k is not None:\n"
        "            kth = logits.topk(top_k, dim=-1).values[..., -1:]\n"
        "            logits = logits.masked_fill(logits < kth,\n"
        "                                        torch.finfo(logits.dtype).min)\n"
        "        probs = torch.softmax(logits, dim=-1)\n"
        "        nxt = torch.multinomial(probs, 1)\n"
        "        idx = torch.cat([idx, nxt], dim=1)\n"
        "    return idx\n"
    ),
    frameworks=["torch"],
    traps=[
        "Feeding the whole history once the prompt exceeds the block size, which "
        "raises an index error out of the position embedding.",
        "Taking the argmax over all positions instead of sampling at the last one.",
        "Leaving autograd on, which builds a graph across the whole generation "
        "and exhausts memory on long samples.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = _GPT(65, 32, 4, 2, 32)
    prompt = torch.randint(0, 65, (2, 5))
    out = fn(m, prompt, 7)
    check("length grows by n", lambda: shape(out) == (2, 12))
    check("the prompt is preserved", lambda: close(out[:, :5], prompt))
    check("tokens are in range", lambda: bool(((out >= 0) & (out < 65)).all()))
    check("dtype stays integer", lambda: out.dtype == torch.long)
    def crops_context():
        long_prompt = torch.randint(0, 65, (1, 40))   # longer than block=32
        return shape(fn(m, long_prompt, 3)) == (1, 43)
    check("crops a prompt longer than the block size", crops_context)
    def greedy_is_deterministic():
        a = fn(m, prompt, 5, temp=0.01, top_k=1)
        b = fn(m, prompt, 5, temp=0.01, top_k=1)
        return close(a, b)
    check("top_k=1 is deterministic", greedy_is_deterministic)
    def top_k_restricts():
        torch.manual_seed(1)
        outs = {int(v) for _ in range(12)
                for v in fn(m, prompt[:1], 1, temp=1.0, top_k=1)[:, -1]}
        return len(outs) == 1
    check("top_k=1 always picks the same token", top_k_restricts)
''',
),

]
