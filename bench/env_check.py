"""Does the reference-kernel env run on this node? Versions, CUDA, one FlashInfer decode against a torch reference."""
import torch
import triton

print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, flush=True)
print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "", flush=True)
import flashinfer  # noqa: E402

print("flashinfer", flashinfer.__version__, flush=True)
HQ, HKV, D, S = 32, 8, 128, 4096
q = torch.randn(HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = flashinfer.single_decode_with_kv_cache(q, k, v)
kk = k.float().repeat_interleave(HQ // HKV, 1)  # (S, HQ, D)
w = torch.softmax(torch.einsum("hd,shd->hs", q.float(), kk) / D ** 0.5, -1)
ref = torch.einsum("hs,shd->hd", w, v.float().repeat_interleave(HQ // HKV, 1))
print("single_decode bf16 max err vs torch:", (out.float() - ref).abs().max().item(), flush=True)
