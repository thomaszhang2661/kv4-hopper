"""vLLM plugin (11-12): run vLLM's `--kv-cache-dtype ultraquant_4bit` on Hopper with the kv4-hopper kernels.

The ULTRAQUANT backend is overridden by a subclass whose store, decode and continuation-prefill seams use our page
layout (kv4/cuda/store_fp4.cu, decode_fp4.cu): same bytes per token (136 at head_dim 128), same 64-token blocks.
Needs the kv4-hopper repo on PYTHONPATH (it imports `kv4`). Set KV4_VLLM=0 to leave vLLM untouched.
KV4_NO_BOUNDARY=1: quantize the boundary layers too (needed to select ULTRAQUANT explicitly on CUDA).
KV4_BOUNDARY (11-12): which layers stay in the native dtype. Unset: vLLM's default (first and last 2 attention
layers); "none": none (= KV4_NO_BOUNDARY=1); otherwise a comma list of layer indices, e.g. "0" (our recipe).
KV4_AUTOSELECT (11-12, default 1): put ULTRAQUANT in the CUDA backend priority list, so `--kv-cache-dtype
ultraquant_4bit` works without `--attention-backend` and the boundary layers fall back to FLASH_ATTN.
KV4_BOUNDARY_FA3 (11-13, default 1 with KV4_VLLM=1): vLLM forces flash_attn_version=2 for every layer of an
UltraQuant / TurboQuant run (engine/arg_utils.py), so the native-precision boundary layers run FA2 (the profile:
268 ms of FA2 for layer 0 at B=8 x 32K prefill, ~2.5x FA3). This puts FLASH_ATTN back on FA3 on SM90 and applies
the fix of vLLM PR #65891: FA3's ahead-of-time scheduler metadata is built for the layer's own KV dtype, not the
global one (the global quantized dtype gave the scheduler_metadata shape crash of #65821).
KV4_SCALE_C: the MXFP4 scale constant (default 0.156, UltraQuant's; OCP rule if <= 0).
"""
import os


def register():
    _boundary()
    if os.environ.get("KV4_KCENTER"):
        _kcenter(os.environ["KV4_KCENTER"])
    if os.environ.get("KV4_UQ_C"):
        # 11-15: override vLLM's own UltraQuant scale constant (format.DEFAULT_CONSTANT_C, read at call time through
        # get_constant_c() by the Triton store and the reference), to measure the stock path at c = 0.20 for the
        # #67176 PR. Independent of KV4_VLLM (use KV4_VLLM=0 for the stock backend).
        from vllm.v1.attention.ops.ultraquant import format as _uqf
        _uqf.DEFAULT_CONSTANT_C = float(os.environ["KV4_UQ_C"])
        print(f"kv4_vllm: UltraQuant DEFAULT_CONSTANT_C = {_uqf.get_constant_c()}", flush=True)
    if os.environ.get("KV4_UQ_STORE"):
        # 11-20: replace vLLM's own UltraQuant store launcher with the one in the given file (the vllm-contrib branch
        # `ultraquant-best-exponent`: per group, the better of two power-of-two scale exponents). The backend imports
        # `ultraquant_store` from the module at call time, so swapping the module attribute is enough.
        import importlib.util
        from vllm.v1.attention.ops.ultraquant import triton_store as _uqs
        _spec = importlib.util.spec_from_file_location("kv4_uq_store", os.environ["KV4_UQ_STORE"])
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _uqs.ultraquant_store = _mod.ultraquant_store
        print(f"kv4_vllm: UltraQuant store from {os.environ['KV4_UQ_STORE']}", flush=True)
    if os.environ.get("KV4_AUTOSELECT", "1") == "1":
        _autoselect()
    if os.environ.get("KV4_NO_BOUNDARY") == "1":
        # vLLM keeps the first and last 2 attention layers of an ultraquant_4bit / turboquant run in the native dtype
        # (engine/arg_utils.py, TurboQuantConfig.get_boundary_skip_layers). On ROCm those layers get another backend;
        # on CUDA ULTRAQUANT is not in the priority list, so it must be selected explicitly, and then the boundary
        # layers fail validation. KV4_NO_BOUNDARY=1 quantizes every layer instead (independent of KV4_VLLM, so the
        # stock backend can be compared under the same setting).
        from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
        TurboQuantConfig.get_boundary_skip_layers = staticmethod(lambda *a, **k: [])
    if os.environ.get("KV4_VLLM", "1") == "0":
        return
    if os.environ.get("KV4_BOUNDARY_FA3", "1") == "1":
        _boundary_fa3()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.ULTRAQUANT, "kv4_vllm.backend.KV4UltraQuantBackend")


def _boundary():
    b = os.environ.get("KV4_BOUNDARY")
    if b is None:
        return
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    # 11-13: "+" also separates (KV4_BOUNDARY=0+39 inside a vllm_reasoning_check dtype spec, where "," splits specs)
    layers = [] if b.strip().lower() in ("", "none") else [x.strip() for x in b.replace("+", ",").split(",") if x.strip()]
    TurboQuantConfig.get_boundary_skip_layers = staticmethod(lambda *a, **k: list(layers))


def _kcenter(mode):
    """11-16, for the vllm#41343 follow-up: a fixed per-layer K center derived from Qwen2's k_proj bias, subtracted
    from every key right after RoPE in Qwen2Attention.forward, for any KV dtype and backend (independent of
    KV4_VLLM). Exact for attention: q . c is the same for every key of a query and drops out of the softmax; only the
    quantization error of K changes. Prior art: rtp-llm PR #1583 subtracts the raw bias after RoPE for per-token FP8
    (mode "bias"); "ropeavg" is c = mean over p in [4, T) of RoPE(b, p) (bench/kbias_vs_mean.py, our 11-14
    measurement; T = KV4_KCENTER_T, default 4096). A prototype for the numbers, not a vLLM patch proposal.
    The center is a non-persistent buffer filled once after load_weights (no Python side effects inside the
    torch.compile'd forward). Layers without a k bias keep a zero center. TP shards: the k slice of the local bias."""
    if os.environ.get("VLLM_DISABLE_COMPILE_CACHE") != "1":
        # 11-17: vLLM's AOT compile cache is keyed by the config and the recorded source files (qwen2.py), not by the
        # forward patched in here, so a cached graph from a plain run (or a centered one) was loaded silently.
        raise RuntimeError("KV4_KCENTER needs VLLM_DISABLE_COMPILE_CACHE=1 (the compile cache ignores this patch)")
    import torch
    from vllm.model_executor.models import qwen2
    A, M = qwen2.Qwen2Attention, qwen2.Qwen2ForCausalLM
    if getattr(A.forward, "_kv4", False):
        return
    T = int(os.environ.get("KV4_KCENTER_T", "4096"))
    init, load = A.__init__, M.load_weights

    def __init__(self, *a, **k):
        init(self, *a, **k)
        self.register_buffer("_kv4_c", torch.zeros(self.kv_size), persistent=False)

    def fill(self):
        bias = getattr(self.qkv_proj, "bias", None)
        if bias is None:
            return None
        b = bias.detach().float()[self.q_size:self.q_size + self.kv_size].view(self.num_kv_heads, self.head_dim)
        re = self.rotary_emb
        if mode == "ropeavg":
            rd = re.rotary_dim
            assert re.is_neox_style and rd == self.head_dim, "ropeavg assumes full neox-style rotary"
            h = rd // 2
            i = torch.arange(h, dtype=torch.float64, device=b.device)
            th = float(re.base) ** (-2 * i / rd)
            pos = torch.arange(4, T, dtype=torch.float64, device=b.device)
            C = torch.cos(pos[:, None] * th).mean(0).float()
            S = torch.sin(pos[:, None] * th).mean(0).float()
            b1, b2 = b[:, :h], b[:, h:]
            b = torch.cat([b1 * C - b2 * S, b2 * C + b1 * S], -1)
        elif mode != "bias":
            raise ValueError(f"KV4_KCENTER={mode!r}: expected bias or ropeavg")
        self._kv4_c.copy_(b.reshape(-1).to(self._kv4_c.dtype))
        return b.norm(dim=-1).mean().item()

    def load_weights(self, *a, **k):
        out = load(self, *a, **k)
        norms = [fill(m) for m in self.modules() if isinstance(m, A)]
        print(f"kv4_vllm: K center ({mode}, T={T}) filled for {sum(n is not None for n in norms)} layers; "
              f"|c| per head layer 0 {norms[0]}, median {(lambda v: v[len(v) // 2] if v else None)(sorted(x for x in norms if x is not None))}",
              flush=True)
        return out

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_norm:
            n = q.shape[0]
            q = self.q_norm(q.view(n, self.num_heads, self.head_dim)).view(n, self.q_size)
            k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim)).view(n, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        k = k - self._kv4_c
        output, _ = self.o_proj(self.attn(q, k, v))
        return output
    forward._kv4 = True
    A.__init__, A.forward, M.load_weights = __init__, forward, load_weights
    print(f"kv4_vllm: Qwen2Attention K center = {mode}", flush=True)


def _autoselect():
    """vLLM 0.33 lists TURBOQUANT but not ULTRAQUANT among the CUDA (non-MLA) backend priorities, so ultraquant_4bit
    layers find no backend unless it is forced for every layer. Appending it lets each layer pick by its own KV dtype:
    ultraquant_4bit layers -> ULTRAQUANT (ours when KV4_VLLM=1), skipped layers -> FLASH_ATTN."""
    from vllm.platforms import cuda
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    if getattr(cuda._get_backend_priorities, "_kv4", False):
        return
    orig = cuda._get_backend_priorities

    def prio(use_mla, *a, **k):
        out = orig(use_mla, *a, **k)
        if not use_mla and AttentionBackendEnum.ULTRAQUANT not in out:
            out = [*out, AttentionBackendEnum.ULTRAQUANT]
        return out
    prio._kv4 = True
    cuda._get_backend_priorities = prio


def _boundary_fa3():
    import torch
    from vllm.v1.attention.backends import fa_utils, flash_attn
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache
    if getattr(flash_attn.get_flash_attn_version, "_kv4", False):
        return
    orig = fa_utils.get_flash_attn_version

    def version(*a, **k):
        v = orig(*a, **k)
        if v != 2 or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return v
        from vllm.config import get_current_vllm_config_or_none
        from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
        cfg = get_current_vllm_config_or_none()
        dt = getattr(getattr(cfg, "cache_config", None), "cache_dtype", None)
        if dt == "ultraquant_4bit" and is_fa_version_supported(3):
            return 3
        return v
    version._kv4 = True
    flash_attn.get_flash_attn_version = version  # the module imported the name

    B = flash_attn.FlashAttentionMetadataBuilder
    sched = B._get_scheduler_metadata

    def get_scheduler_metadata(self, **kw):
        spec = getattr(self, "kv_cache_spec", None)
        cc = self.cache_config
        if spec is not None and int(getattr(spec, "kv_quant_mode", 1)) == 0 and \
                is_quantized_kv_cache(cc.cache_dtype):  # KVQuantMode.NONE (an IntEnum)
            saved = cc.cache_dtype
            cc.cache_dtype = "auto"  # PR #65891: the layer's own (native) dtype
            try:
                return sched(self, **kw)
            finally:
                cc.cache_dtype = saved
        return sched(self, **kw)
    B._get_scheduler_metadata = get_scheduler_metadata
