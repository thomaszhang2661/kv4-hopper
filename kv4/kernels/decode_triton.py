"""Decode attention in Triton for every KV format in kv4.formats (pass 1: correct and readable).

Contract (same as kv4.kernels.decode_bf16, checked by tests/test_kernels.py):
  q (B, H_q, D) bf16;  k, v (B, H_kv, S, D) bf16 or QTensor;  seq_lens (B,) int32;
  returns (B, H_q, D) float32.  Query head h reads kv head h // (H_q // H_kv).

How it works, in three steps
----------------------------
1. Split-K ("flash decoding"). One program handles one (batch, kv head, chunk of the
   sequence). A single (batch, kv head) would leave most of the GPU's SMs idle at small
   batch sizes, so the sequence is cut into NSPLIT chunks that run in parallel.
2. Inside a program: loop over the chunk in tiles of BLOCK_N keys, with online softmax
   (running max m, running sum l, running output acc). All G = H_q / H_kv query heads
   that share this kv head are processed together, so every K/V tile is read once.
3. A second small kernel merges the NSPLIT partial results (m, l, acc) of each query head.

4-bit formats and the even/odd trick
------------------------------------
A byte holds two codes: element 2i in the low nibble, 2i+1 in the high nibble. Instead of
interleaving the two halves back into one (BLOCK_N, D) tile, the kernel keeps them apart:
  scores = q_even . K_low + q_odd . K_high        (two dots over D/2)
  out_even = P . V_low,  out_odd = P . V_high      (written back to columns 0::2 / 1::2)
which is the same arithmetic and needs no shuffles.

Rotation (Hadamard formats) is done outside the kernel with torch: q is rotated before,
the output after (see kv4.formats for why that is exact). Fusing it is a pass-3 item.

Known simplifications to revisit in pass 3: fixed BLOCK_N / num_warps, scales are loaded
once per element pair (redundant but cached), no fused rotation, q and probabilities are
cast to bf16 for the tensor-core dots.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from kv4.formats import FORMATS, QTensor, rotate

# element/scale kinds, as a compile-time switch inside the kernels
# (tl.constexpr: Triton only lets jit code read module-level constants declared this way;
# plain Python code below uses the integer .value)
BF16, FP8, INT4, INT4A, NVFP4, MXFP4 = (tl.constexpr(i) for i in range(6))


@triton.jit
def _e2m1_raw(code):
    """FP4 E2M1 code (int32, 0..15: sign bit 3, exponent bits 2..1, mantissa bit 0) -> float32, times 2^-14.

    Pass 3 (bit trick instead of the pass-1 branches and exp2): put the 3 magnitude bits at
    bits 11..9 of an fp16 and the sign at bit 15, then read those 16 bits as an fp16.
      exponent field e > 0:  2^(e-15) * (1 + m/2)   = 2^-14 * 2^(e-1) * (1 + m/2)  (the E2M1 value)
      exponent field e == 0: subnormal m * 2^-1 * 2^-14                            (0.5 * m)
    so the result is the E2M1 value scaled by 2^-14. The caller multiplies the 2^14 into the
    block scale it applies anyway, so each element costs only a few integer operations.
    """
    bits = ((code & 7) << 9) | ((code & 8) << 12)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _partial_int4_loads(k_ptr, ks_ptr, v_ptr, vs_ptr, seqlen_ptr, pacc_ptr, S, CHUNK, NSPLIT,
                        H_Q: tl.constexpr, H_KV: tl.constexpr, G: tl.constexpr,
                        D: tl.constexpr, BLOCK_N: tl.constexpr):
    """Ablation only (wrong numbers): same loads as _partial_int4_fold, no dot, no softmax.

    Its time is the floor for this load pattern and this BLOCK_N: bytes, scales, loop, nothing else.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    seq_len = tl.load(seqlen_ptr + pid_b)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)
    HD: tl.constexpr = D // 2
    d2 = tl.arange(0, HD)
    bh = pid_b.to(tl.int64) * H_KV + pid_h
    dbase = bh * S * HD
    sbase = bh * S
    acc = tl.zeros([HD], tl.float32)
    zs = 0.0
    for n0 in range(start, end, BLOCK_N):
        rows = tl.minimum(n0 + tl.arange(0, BLOCK_N), end - 1)
        kb = tl.load(k_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        vb = tl.load(v_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        acc += tl.sum(kb, 0).to(tl.float32) + tl.sum(vb, 0).to(tl.float32)
        zs += tl.sum(tl.load(ks_ptr + sbase + rows).to(tl.float32) + tl.load(vs_ptr + sbase + rows).to(tl.float32), 0)
    qrow = pid_b * H_Q + pid_h * G
    tl.store(pacc_ptr + (qrow * NSPLIT + pid_s) * D + 2 * d2, acc + zs)


@triton.jit
def _dequant_pair(lo, hi, s_ptr, z_ptr, sbase, rows, sidx, ts, NBLK: tl.constexpr, FMT: tl.constexpr):
    """Dequantize the low and high nibbles of a (BLOCK_N, D/2) tile of bytes.

    lo, hi : int32 codes 0..15 of elements 2i and 2i+1.
    rows   : (BLOCK_N,) token indices; sidx: (D/2,) index of the scale block of element 2i
             (2i and 2i+1 always share a block). Scales are (.., S, NBLK) contiguous.
    """
    sp = sbase + rows[:, None] * NBLK + sidx[None, :]
    if FMT == INT4:  # symmetric: code = q + 8, x = q * scale (fp16 scale)
        sc = tl.load(s_ptr + sp).to(tl.float32)
        a = (lo - 8).to(tl.float32) * sc
        b = (hi - 8).to(tl.float32) * sc
    elif FMT == INT4A:  # asymmetric: x = code * scale + zero (fp16 each)
        sc = tl.load(s_ptr + sp).to(tl.float32)
        z = tl.load(z_ptr + sp).to(tl.float32)
        a = lo.to(tl.float32) * sc + z
        b = hi.to(tl.float32) * sc + z
    elif FMT == NVFP4:  # scale: E4M3 (loaded as fp8) times one fp32 tensor scale
        sc = tl.load(s_ptr + sp).to(tl.float32) * (ts * 16384.0)  # 2^14 undoes the scaling in _e2m1_raw
        a = _e2m1_raw(lo) * sc
        b = _e2m1_raw(hi) * sc
    else:  # MXFP4: scale is the power of two 2^(e - 127), e stored as a uint8
        sc = tl.exp2(tl.load(s_ptr + sp).to(tl.float32) - 113.0)  # 2^(e-127) * 2^14
        a = _e2m1_raw(lo) * sc
        b = _e2m1_raw(hi) * sc
    return a, b


@triton.jit
def _partial_full(q_ptr, k_ptr, v_ptr, seqlen_ptr, pm_ptr, pl_ptr, pacc_ptr,
                  sm_scale, k_ts, v_ts, S, CHUNK, NSPLIT,
                  H_Q: tl.constexpr, H_KV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr,
                  D: tl.constexpr, BLOCK_N: tl.constexpr, FMT: tl.constexpr, CONV: tl.constexpr):
    """Partial attention of one (batch, kv head, sequence chunk) for BF16 and FP8 caches."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    seq_len = tl.load(seqlen_ptr + pid_b)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)  # this chunk's keys: start .. end-1 (maybe none)

    QDT: tl.constexpr = tl.float16 if (FMT == FP8 and CONV == 4) else tl.bfloat16  # dot operand type
    g = tl.arange(0, G_PAD)  # query heads of the group, padded up to 16 for tl.dot
    d = tl.arange(0, D)
    gmask = g < G
    qrow = pid_b * H_Q + pid_h * G + g
    q = tl.load(q_ptr + qrow[:, None] * D + d[None, :], mask=gmask[:, None], other=0.0).to(QDT)

    kvbase = (pid_b.to(tl.int64) * H_KV + pid_h) * S * D

    if CONV == 0:  # scale applied per element (pass-1 behaviour)
        ksc = sm_scale
        vsc = 1.0
    else:          # per-tensor scales folded into the scores / the output
        ksc = sm_scale * k_ts
        vsc = v_ts

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)  # running max of the scores
    l_i = tl.zeros([G_PAD], tl.float32)                # running sum of exp(score - m_i)
    acc = tl.zeros([G_PAD, D], tl.float32)             # running sum of exp(score - m_i) * v

    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        rows = tl.minimum(n, end - 1)  # out-of-range rows re-read a valid row; they are masked below

        kt = tl.load(k_ptr + kvbase + rows[:, None] * D + d[None, :])
        if FMT == FP8:
            if CONV == 0:
                kt = (kt.to(tl.float32) * k_ts).to(tl.bfloat16)
            elif CONV == 1:
                kt = kt.to(tl.bfloat16)
            elif CONV == 2:
                kt = kt.to(tl.float32).to(tl.bfloat16)
            elif CONV == 3:
                kt = kt.to(tl.float16).to(tl.bfloat16)
            else:
                kt = kt.to(tl.float16)
        s = tl.dot(q, tl.trans(kt)) * ksc  # (G_PAD, BLOCK_N)
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)                      # rescales what was accumulated so far
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        vt = tl.load(v_ptr + kvbase + rows[:, None] * D + d[None, :])
        if FMT == FP8:
            if CONV == 0:
                vt = (vt.to(tl.float32) * v_ts).to(tl.bfloat16)
            elif CONV == 1:
                vt = vt.to(tl.bfloat16)
            elif CONV == 2:
                vt = vt.to(tl.float32).to(tl.bfloat16)
            elif CONV == 3:
                vt = vt.to(tl.float16).to(tl.bfloat16)
            else:
                vt = vt.to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(QDT), vt)
        m_i = m_new

    pidx = qrow * NSPLIT + pid_s
    tl.store(pm_ptr + pidx, m_i, mask=gmask)
    tl.store(pl_ptr + pidx, l_i, mask=gmask)
    tl.store(pacc_ptr + pidx[:, None] * D + d[None, :], acc * vsc, mask=gmask[:, None])


@triton.jit
def _partial_4bit(q_ptr, k_ptr, ks_ptr, kz_ptr, v_ptr, vs_ptr, vz_ptr, seqlen_ptr,
                  pm_ptr, pl_ptr, pacc_ptr, sm_scale, k_ts, v_ts, S, CHUNK, NSPLIT,
                  H_Q: tl.constexpr, H_KV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr,
                  D: tl.constexpr, BLOCK_N: tl.constexpr, FMT: tl.constexpr,
                  NBLK: tl.constexpr, BLKE: tl.constexpr):
    """Same as _partial_full for packed 4-bit caches (INT4, INT4 asymmetric, NVFP4, MXFP4)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    seq_len = tl.load(seqlen_ptr + pid_b)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)

    HD: tl.constexpr = D // 2
    g = tl.arange(0, G_PAD)
    d2 = tl.arange(0, HD)
    gmask = g < G
    qrow = pid_b * H_Q + pid_h * G + g
    qp = q_ptr + qrow[:, None] * D + 2 * d2[None, :]
    q_even = tl.load(qp, mask=gmask[:, None], other=0.0).to(tl.bfloat16)      # q[0], q[2], ...
    q_odd = tl.load(qp + 1, mask=gmask[:, None], other=0.0).to(tl.bfloat16)   # q[1], q[3], ...

    bh = pid_b.to(tl.int64) * H_KV + pid_h
    dbase = bh * S * HD            # packed bytes: (.., S, D/2)
    sbase = bh * S * NBLK          # scales / zeros: (.., S, NBLK)
    sidx = (2 * d2) // BLKE        # scale block of element 2i (same as of 2i+1)

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc_e = tl.zeros([G_PAD, HD], tl.float32)
    acc_o = tl.zeros([G_PAD, HD], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        rows = tl.minimum(n, end - 1)

        kb = tl.load(k_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        k_lo, k_hi = _dequant_pair(kb & 0xF, kb >> 4, ks_ptr, kz_ptr, sbase, rows, sidx, k_ts, NBLK, FMT)
        s = tl.dot(q_even, tl.trans(k_lo.to(tl.bfloat16)))
        s += tl.dot(q_odd, tl.trans(k_hi.to(tl.bfloat16)))
        s = tl.where(valid[None, :], s * sm_scale, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None]).to(tl.bfloat16)
        l_i = l_i * alpha + tl.sum(p.to(tl.float32), 1)

        vb = tl.load(v_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        v_lo, v_hi = _dequant_pair(vb & 0xF, vb >> 4, vs_ptr, vz_ptr, sbase, rows, sidx, v_ts, NBLK, FMT)
        acc_e = acc_e * alpha[:, None] + tl.dot(p, v_lo.to(tl.bfloat16))
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v_hi.to(tl.bfloat16))
        m_i = m_new

    pidx = qrow * NSPLIT + pid_s
    tl.store(pm_ptr + pidx, m_i, mask=gmask)
    tl.store(pl_ptr + pidx, l_i, mask=gmask)
    op = pacc_ptr + pidx[:, None] * D + 2 * d2[None, :]
    tl.store(op, acc_e, mask=gmask[:, None])       # even columns
    tl.store(op + 1, acc_o, mask=gmask[:, None])   # odd columns


@triton.jit
def _partial_int4_fold(q_ptr, k_ptr, ks_ptr, kz_ptr, v_ptr, vs_ptr, vz_ptr, seqlen_ptr,
                       pm_ptr, pl_ptr, pacc_ptr, sm_scale, S, CHUNK, NSPLIT,
                       H_Q: tl.constexpr, H_KV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr,
                       D: tl.constexpr, BLOCK_N: tl.constexpr, ASYM: tl.constexpr, ABL: tl.constexpr = 0, INJ: tl.constexpr = 0):
    """Pass 3, step 1: affine folding for token-wise INT4 (one fp16 scale, and a zero, per row).

    A row is x[d] = c[d] * sc + z (symmetric INT4: z = -8 * sc). Instead of building x for
    every element, keep the integer codes and apply sc and z to the per-token results, which
    are far fewer than the per-element values. The codes are centred first, c' = c - OFF
    (OFF = 8 symmetric, 7.5 asymmetric; exact in bf16), so x = c' * sc + z' with z' = 0
    (symmetric) or z + 7.5 * sc. Without centring, the large zero term cancels against
    the code sum and bf16 rounding of p * sc is amplified (the first version failed the
    test tolerance this way). Below, c and z mean c' and z':
      K:  q . x = sc * (q . c) + z * sum(q)           -> fix up the (G, BLOCK_N) scores
      V:  sum_n p[n] x[n, d] = sum_n (p[n] sc[n]) c[n, d] + sum_n p[n] z[n]
          -> scale p by sc before the dot, and add one per-head constant to every column
    The per-element work is now just: unpack the nibble, convert to bf16.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    seq_len = tl.load(seqlen_ptr + pid_b)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)

    HD: tl.constexpr = D // 2
    # centre the codes: (c - OFF) is exact in bf16; inject=2 needs OFF = 8 (135.5 is not a bf16 number)
    OFF: tl.constexpr = 7.5 if (ASYM and INJ != 2) else 8.0
    g = tl.arange(0, G_PAD)
    d2 = tl.arange(0, HD)
    gmask = g < G
    qrow = pid_b * H_Q + pid_h * G + g
    qp = q_ptr + qrow[:, None] * D + 2 * d2[None, :]
    q_even = tl.load(qp, mask=gmask[:, None], other=0.0).to(tl.bfloat16)
    q_odd = tl.load(qp + 1, mask=gmask[:, None], other=0.0).to(tl.bfloat16)
    qsum = tl.sum(q_even.to(tl.float32), 1) + tl.sum(q_odd.to(tl.float32), 1)  # (G_PAD,)

    bh = pid_b.to(tl.int64) * H_KV + pid_h
    dbase = bh * S * HD
    sbase = bh * S  # one scale per token

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    zacc = tl.zeros([G_PAD], tl.float32)  # sum_n p[n] * z[n], same for every column
    psum = tl.zeros([G_PAD], tl.float32)  # injection only: sum_n ps[n] (rounded), to remove the 128 + OFF
    acc_e = tl.zeros([G_PAD, HD], tl.float32)
    acc_o = tl.zeros([G_PAD, HD], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        rows = tl.minimum(n, end - 1)

        if ABL == 3:  # ablation: no scale loads, no fix-ups
            ksc = tl.full([BLOCK_N], 1.0, tl.float32)
        else:
            ksc = tl.load(ks_ptr + sbase + rows).to(tl.float32)
        if ASYM:
            kz = tl.load(kz_ptr + sbase + rows).to(tl.float32) + OFF * ksc
        else:
            kz = tl.zeros_like(ksc)

        kb = tl.load(k_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        if INJ:  # mantissa injection: 0x4300 | code is the bf16 number 128 + code, exactly
            klo = ((kb & 0xF) | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
            khi = ((kb >> 4) | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
            if INJ == 2:  # (128 + c) - 136 = c - 8, exact in bf16: no correction needed after the dot
                klo = (klo - 136.0).to(tl.bfloat16)
                khi = (khi - 136.0).to(tl.bfloat16)
        elif ABL == 2:  # ablation: no nibble unpacking (wrong numbers, load + dot only)
            klo = kb.to(tl.float32).to(tl.bfloat16)
            khi = klo
        else:
            klo = ((kb & 0xF).to(tl.float32) - OFF).to(tl.bfloat16)
            khi = ((kb >> 4).to(tl.float32) - OFF).to(tl.bfloat16)
        raw = tl.dot(q_even, tl.trans(klo))
        raw += tl.dot(q_odd, tl.trans(khi))
        if INJ == 1:  # q . (c + 128) -> q . (c - OFF): subtract (128 + OFF) * sum(q), in fp32
            raw = raw - (128.0 + OFF) * qsum[:, None]
        s = (raw * ksc[None, :] + qsum[:, None] * kz[None, :]) * sm_scale
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])  # fp32, zero for masked rows
        l_i = l_i * alpha + tl.sum(p, 1)

        if ABL == 3:
            vsc = tl.full([BLOCK_N], 1.0, tl.float32)
        else:
            vsc = tl.load(vs_ptr + sbase + rows).to(tl.float32)
        if ASYM:
            vz = tl.load(vz_ptr + sbase + rows).to(tl.float32) + OFF * vsc
        else:
            vz = tl.zeros_like(vsc)
        ps = (p * vsc[None, :]).to(tl.bfloat16)
        zacc = zacc * alpha + tl.sum(p * vz[None, :], 1)

        vb = tl.load(v_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        if INJ:
            vlo = ((vb & 0xF) | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
            vhi = ((vb >> 4) | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
            if INJ == 2:
                vlo = (vlo - 136.0).to(tl.bfloat16)
                vhi = (vhi - 136.0).to(tl.bfloat16)
            else:  # the correction uses the same rounded ps as the dot, so its rounding error cancels
                psum = psum * alpha + tl.sum(ps.to(tl.float32), 1)
        elif ABL == 2:
            vlo = vb.to(tl.float32).to(tl.bfloat16)
            vhi = vlo
        else:
            vlo = ((vb & 0xF).to(tl.float32) - OFF).to(tl.bfloat16)
            vhi = ((vb >> 4).to(tl.float32) - OFF).to(tl.bfloat16)
        acc_e = acc_e * alpha[:, None] + tl.dot(ps, vlo)
        acc_o = acc_o * alpha[:, None] + tl.dot(ps, vhi)
        m_i = m_new

    if INJ == 1:
        zacc = zacc - (128.0 + OFF) * psum
    acc_e += zacc[:, None]
    acc_o += zacc[:, None]
    pidx = qrow * NSPLIT + pid_s
    tl.store(pm_ptr + pidx, m_i, mask=gmask)
    tl.store(pl_ptr + pidx, l_i, mask=gmask)
    op = pacc_ptr + pidx[:, None] * D + 2 * d2[None, :]
    tl.store(op, acc_e, mask=gmask[:, None])
    tl.store(op + 1, acc_o, mask=gmask[:, None])


@triton.jit
def _partial_int4_fold_swap(q_ptr, k_ptr, ks_ptr, kz_ptr, v_ptr, vs_ptr, vz_ptr, seqlen_ptr,
                            pm_ptr, pl_ptr, pacc_ptr, sm_scale, S, CHUNK, NSPLIT,
                            H_Q: tl.constexpr, H_KV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr,
                            D: tl.constexpr, BLOCK_N: tl.constexpr, ASYM: tl.constexpr):
    """_partial_int4_fold with the dot operands swapped ("swap A and B"), same math and result.

    Before: scores = q (G_PAD x D) . K^T, so the unpacked K is the dot's B operand and the dot's
    M is G_PAD = 16 (4 real query heads). Now: scores^T = K (BLOCK_N x D) . q^T, so the unpacked K
    is the A operand and M = BLOCK_N = 128 tokens; likewise out^T = V^T . P^T makes the unpacked V
    the A operand (M = D/2 = 64 per half). Why: on Hopper, Triton uses wgmma only when M >= 64,
    and only wgmma's A operand may live in registers; for a register A operand Triton can move the
    layout conversion to before our elementwise unpack (read the bytes straight in the dot
    layout), which is what would remove the write-back of unpacked bf16 to shared memory.
    B (q^T, P^T) stays in shared memory: q^T once, P^T (BLOCK_N x 16 bf16, 4 KB) per tile.
    Everything per token is now along axis 0 (columns of the old layout became rows).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    seq_len = tl.load(seqlen_ptr + pid_b)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)

    HD: tl.constexpr = D // 2
    OFF: tl.constexpr = 7.5 if ASYM else 8.0  # centred codes, exact in bf16 (see _partial_int4_fold)
    g = tl.arange(0, G_PAD)
    d2 = tl.arange(0, HD)
    gmask = g < G
    qrow = pid_b * H_Q + pid_h * G + g
    qp = q_ptr + qrow[None, :] * D + 2 * d2[:, None]  # (HD, G_PAD): q^T, even / odd dims
    q_even_t = tl.load(qp, mask=gmask[None, :], other=0.0).to(tl.bfloat16)
    q_odd_t = tl.load(qp + 1, mask=gmask[None, :], other=0.0).to(tl.bfloat16)
    qsum = tl.sum(q_even_t.to(tl.float32), 0) + tl.sum(q_odd_t.to(tl.float32), 0)  # (G_PAD,)

    bh = pid_b.to(tl.int64) * H_KV + pid_h
    dbase = bh * S * HD
    sbase = bh * S

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    zacc = tl.zeros([G_PAD], tl.float32)
    acc_e = tl.zeros([HD, G_PAD], tl.float32)  # out^T, even dims
    acc_o = tl.zeros([HD, G_PAD], tl.float32)  # out^T, odd dims

    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        rows = tl.minimum(n, end - 1)

        ksc = tl.load(ks_ptr + sbase + rows).to(tl.float32)
        if ASYM:
            kz = tl.load(kz_ptr + sbase + rows).to(tl.float32) + OFF * ksc
        else:
            kz = tl.zeros_like(ksc)
        kb = tl.load(k_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        klo = ((kb & 0xF).to(tl.float32) - OFF).to(tl.bfloat16)  # (BLOCK_N, HD)
        khi = ((kb >> 4).to(tl.float32) - OFF).to(tl.bfloat16)
        raw = tl.dot(klo, q_even_t)                               # (BLOCK_N, G_PAD)
        raw += tl.dot(khi, q_odd_t)
        s = (raw * ksc[:, None] + kz[:, None] * qsum[None, :]) * sm_scale
        s = tl.where(valid[:, None], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[None, :])                            # (BLOCK_N, G_PAD)
        l_i = l_i * alpha + tl.sum(p, 0)

        vsc = tl.load(vs_ptr + sbase + rows).to(tl.float32)
        if ASYM:
            vz = tl.load(vz_ptr + sbase + rows).to(tl.float32) + OFF * vsc
        else:
            vz = tl.zeros_like(vsc)
        ps = (p * vsc[:, None]).to(tl.bfloat16)                   # P^T scaled per token
        zacc = zacc * alpha + tl.sum(p * vz[:, None], 0)

        vb = tl.load(v_ptr + dbase + rows[:, None] * HD + d2[None, :]).to(tl.int32)
        vlo = ((vb & 0xF).to(tl.float32) - OFF).to(tl.bfloat16)  # (BLOCK_N, HD)
        vhi = ((vb >> 4).to(tl.float32) - OFF).to(tl.bfloat16)
        acc_e = acc_e * alpha[None, :] + tl.dot(tl.trans(vlo), ps)  # (HD, G_PAD)
        acc_o = acc_o * alpha[None, :] + tl.dot(tl.trans(vhi), ps)
        m_i = m_new

    acc_e += zacc[None, :]
    acc_o += zacc[None, :]
    pidx = qrow * NSPLIT + pid_s
    tl.store(pm_ptr + pidx, m_i, mask=gmask)
    tl.store(pl_ptr + pidx, l_i, mask=gmask)
    op = pacc_ptr + pidx[None, :] * D + 2 * d2[:, None]
    tl.store(op, acc_e, mask=gmask[None, :])
    tl.store(op + 1, acc_o, mask=gmask[None, :])


@triton.jit
def _combine(pm_ptr, pl_ptr, pacc_ptr, out_ptr, NSPLIT, NSPLIT_P: tl.constexpr, D: tl.constexpr):
    """Merge the NSPLIT partial (m, l, acc) of one query head into the final output.

    Each partial result is relative to its own max m_s, so rescale by exp(m_s - M) with
    M = max of all m_s. A chunk with no keys has m = -inf and l = 0, i.e. weight 0.
    """
    row = tl.program_id(0)  # b * H_Q + h
    sidx = tl.arange(0, NSPLIT_P)
    smask = sidx < NSPLIT
    d = tl.arange(0, D)

    m = tl.load(pm_ptr + row * NSPLIT + sidx, mask=smask, other=float("-inf"))
    l = tl.load(pl_ptr + row * NSPLIT + sidx, mask=smask, other=0.0)
    w = tl.exp(m - tl.max(m, 0))
    acc = tl.load(pacc_ptr + (row * NSPLIT + sidx[:, None]) * D + d[None, :], mask=smask[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(out_ptr + row * D + d, out)


def _kind(fmt) -> int:
    if fmt.element == "bf16":
        return BF16.value
    if fmt.element == "fp8":
        return FP8.value
    if fmt.element == "int4":
        return (INT4A if fmt.asym else INT4).value
    return (NVFP4 if fmt.scale == "e4m3" else MXFP4).value


def _plan_splits(b: int, h_kv: int, s: int, block_n: int, device) -> tuple[int, int]:
    """(chunk, nsplit): cut S so that about 4 programs per SM are available, never below 256 keys each."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    want = max(1, -(-4 * sms // (b * h_kv)))
    nsplit = max(1, min(want, s // 256, 128))
    chunk = -(-s // nsplit)
    chunk = -(-chunk // block_n) * block_n
    return chunk, -(-s // chunk)


LAST_META: dict = {}  # registers / spills of the main kernel of the last launch (for ablation scripts)


def decode_attention(q, k, v, seq_lens, sm_scale, *, fmt_name: str = "bf16", block_n: int = 128, fold: bool = True, fp8_conv: int = 0,
                     num_warps: int | None = None, num_stages: int | None = None, abl: int = 0, inject: int = 0,
                     swap: bool = False):
    fmt = FORMATS[fmt_name]
    kind = _kind(fmt)
    b, h_q, d = q.shape
    kd = k.data if isinstance(k, QTensor) else k
    h_kv, s = kd.shape[1], kd.shape[2]
    g = h_q // h_kv
    g_pad = max(16, triton.next_power_of_2(g))  # tl.dot needs at least 16 rows

    qf = rotate(q.float(), fmt.hadamard).contiguous()  # identity when fmt.hadamard == 0
    seq_lens = seq_lens.to(torch.int32).contiguous()
    chunk, nsplit = _plan_splits(b, h_kv, s, block_n, q.device)
    kw = {}
    if num_warps is not None:
        kw["num_warps"] = num_warps
    if num_stages is not None:
        kw["num_stages"] = num_stages

    pm = torch.empty(b, h_q, nsplit, dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty(b, h_q, nsplit, d, dtype=torch.float32, device=q.device)
    grid = (b, h_kv, nsplit)

    if kind in (BF16.value, FP8.value):
        _m = _partial_full[grid](
            qf, kd.contiguous(), (v.data if isinstance(v, QTensor) else v).contiguous(), seq_lens, pm, pl, pacc,
            sm_scale, getattr(k, "tensor_scale", 1.0), getattr(v, "tensor_scale", 1.0), s, chunk, nsplit,
            H_Q=h_q, H_KV=h_kv, G=g, G_PAD=g_pad, D=d, BLOCK_N=block_n, FMT=kind, CONV=fp8_conv, **kw)
    else:
        def parts(t: QTensor):
            sc = t.scales
            if fmt.scale == "e4m3":
                sc = sc.view(torch.float8_e4m3fn)  # raw bits -> fp8 so the kernel loads real values
            return t.data.contiguous(), sc.contiguous(), (t.zeros if t.zeros is not None else sc).contiguous()
        kdat, ksc, kz = parts(k)
        vdat, vsc, vz = parts(v)
        if fold and kind in (INT4.value, INT4A.value) and ksc.shape[-1] == 1:
            if abl == 1:  # ablation: loads only
                _m = _partial_int4_loads[grid](
                    kdat, ksc, vdat, vsc, seq_lens, pacc, s, chunk, nsplit,
                    H_Q=h_q, H_KV=h_kv, G=g, D=d, BLOCK_N=block_n, **kw)
            elif swap:  # operands swapped: unpacked K, V as the dot's A operand
                assert abl == 0 and not inject, "swap has no ablation or injection variants"
                _m = _partial_int4_fold_swap[grid](
                    qf, kdat, ksc, kz, vdat, vsc, vz, seq_lens, pm, pl, pacc,
                    sm_scale, s, chunk, nsplit,
                    H_Q=h_q, H_KV=h_kv, G=g, G_PAD=g_pad, D=d, BLOCK_N=block_n, ASYM=kind == INT4A.value, **kw)
            else:
                _m = _partial_int4_fold[grid](
                    qf, kdat, ksc, kz, vdat, vsc, vz, seq_lens, pm, pl, pacc,
                    sm_scale, s, chunk, nsplit,
                    H_Q=h_q, H_KV=h_kv, G=g, G_PAD=g_pad, D=d, BLOCK_N=block_n, ASYM=kind == INT4A.value, ABL=abl, INJ=int(inject), **kw)
        else:
            _m = _partial_4bit[grid](
                qf, kdat, ksc, kz, vdat, vsc, vz, seq_lens, pm, pl, pacc,
                sm_scale, k.tensor_scale, v.tensor_scale, s, chunk, nsplit,
                H_Q=h_q, H_KV=h_kv, G=g, G_PAD=g_pad, D=d, BLOCK_N=block_n, FMT=kind,
                NBLK=ksc.shape[-1], BLKE=fmt.block or d, **kw)

    LAST_META.update(regs=getattr(_m, "n_regs", None), spills=getattr(_m, "n_spills", None), kernel=_m)
    out = torch.empty(b, h_q, d, dtype=torch.float32, device=q.device)
    _combine[(b * h_q,)](pm, pl, pacc, out, nsplit, NSPLIT_P=triton.next_power_of_2(nsplit), D=d)
    return rotate(out, fmt.hadamard)  # V was rotated too: undo it (identity without Hadamard)
