# SPDX-License-Identifier: Apache-2.0
"""Isolated DCP2 GPU prototype; not integrated into Step5.
Full replicated Q/K/V heads, 64-token physical pages and 128-token logical
pages. Global sparse regions must be distinct. FP32 partial states are
retained until inter-rank LSE merge. Query and probability arithmetic
match the existing BF16 HD192 path.
"""
import torch
import triton
import triton.language as tl
NEGATIVE_INFINITY=tl.constexpr(float("-inf"))
REGION_ID_MASK=tl.constexpr((1<<24)-1)
REGION_VALID_SHIFT=tl.constexpr(24)

@triton.jit
def _pack_striped_regions(regions, counts, qpos, requests, table, packed,
                         ROWS:tl.constexpr, TOPK:tl.constexpr,
                         QUERIES:tl.constexpr, REQUESTS:tl.constexpr,
                         TABLE_WIDTH:tl.constexpr, PHYSICAL_PAGES:tl.constexpr, GLOBAL_REGION_SIZE:tl.constexpr, RANK:tl.constexpr,
                         BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//TOPK
    slot=i%TOPK
    valid=i<ROWS*TOPK
    query=row%QUERIES
    logical=tl.load(regions+i,mask=valid,other=-1)
    count=tl.load(counts+row,mask=row<ROWS,other=0)
    position=tl.load(qpos+query,mask=row<ROWS,other=-1)
    request=tl.load(requests+query,mask=row<ROWS,other=-1)
    regions_per_page:tl.constexpr=128//GLOBAL_REGION_SIZE
    page=logical//regions_per_page
    live=valid&(slot<count)&(logical>=0)&(page<TABLE_WIDTH)&(position>=0)
    live=live&(request>=0)&(request<REQUESTS)
    physical_page=tl.load(table+request*TABLE_WIDTH+page,mask=live,other=-1)
    global_count=tl.minimum(GLOBAL_REGION_SIZE,tl.maximum(0,position-logical*GLOBAL_REGION_SIZE+1))
    local_count=(global_count+1-RANK)//2
    physical=physical_page*regions_per_page+logical%regions_per_page
    live=live&(physical_page>=0)&(physical_page<PHYSICAL_PAGES)&(physical<1<<24)&(local_count>0)
    meta=physical|(local_count<<24)
    tl.store(packed+i,tl.where(live,meta,-1),mask=valid)
@triton.jit
def _striped_hd192_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    logical_ptr,
    qpos_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    q_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DK: tl.constexpr,
    QK_FP8: tl.constexpr,
    PV_FP8: tl.constexpr,
    P_SCALE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    RANK: tl.constexpr,
    WINDOW: tl.constexpr,
) -> None:
    """HD192 split attention over the selector\'s causal int32 prefill regions."""
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dk = tl.arange(0, DK)
    head_ok = heads < heads_per_group

    qh = (group * heads_per_group + heads) * stride_q_head
    qr0 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (0 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr1 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (1 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr2 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (2 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    if QK_FP8:
        # 3 * BLOCK_H * DK elements quantized once, against the whole selected cache
        # that now never gets widened.
        q0 = (qr0.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q1 = (qr1.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q2 = (qr2.to(tl.float32) * q_scale).to(tl.float8e4nv)
    else:
        q0, q1, q2 = qr0, qr1, qr2

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    kg = group * stride_k_group
    vg = group * stride_v_group

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        # ``slots < window_end`` is not redundant: ``slots_per_split`` is a ceiling
        # division, so it is a multiple of BLOCK_R only by luck. The production
        # width 513 over 16 splits gives 33 slots per split against BLOCK_R=16, and
        # the last tile of every split would otherwise reach 15 slots into the next
        # split's window. Both splits then accumulate those regions and the merge
        # double-counts them.
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = (meta >> REGION_VALID_SHIFT) & 0xFF
        live = slot_ok & (valid > 0)

        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        logical = tl.load(logical_ptr + row * stride_packed_row + slots,
                          mask=slot_ok, other=-1)
        qpos = tl.load(qpos_ptr + request)
        global_positions = logical[:, None] * (2 * region_size) + 2 * offsets_in_region[None, :] + RANK
        visible = live[:, None] & (offsets_in_region[None, :] < valid[:, None])
        visible = visible & (global_positions <= qpos) & (qpos >= 0)
        if WINDOW > 0:
            visible = visible & (global_positions >= qpos - WINDOW + 1)
        token_ok = tl.reshape(visible, (BLOCK_R * region_size,))

        # Load K already transposed as ``[DK, tokens]``. On the fp8 path the tile goes
        # into the dot as fp8; otherwise it is cast to the query's compute dtype, and an
        # fp8 cache is dequantized by that cast. Either way the descale itself rides on
        # softmax_scale (which the caller has already multiplied by k_scale, and divided
        # by q_scale) and on v_scale at the end.
        kb = k_ptr + tokens[None, :] * stride_k_token + kg
        k0 = tl.load(kb + (0 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k1 = tl.load(kb + (1 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k2 = tl.load(kb + (2 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        if not QK_FP8:
            k0 = k0.to(qr0.dtype)
            k1 = k1.to(qr0.dtype)
            k2 = k2.to(qr0.dtype)

        s = tl.dot(q0, k0, out_dtype=tl.float32)
        s = tl.dot(q1, k1, acc=s, out_dtype=tl.float32)
        s = tl.dot(q2, k2, acc=s, out_dtype=tl.float32)
        s = s * softmax_scale
        s = tl.where(token_ok[None, :] & head_ok[:, None], s, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(s, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp2(running_max - shift))

        probabilities = tl.exp2(s - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)

        vb = v_ptr + tokens[:, None] * stride_v_token + vg
        v0 = tl.load(vb + (0 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v1 = tl.load(vb + (1 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v2 = tl.load(vb + (2 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        if PV_FP8:
            pb = (probabilities * P_SCALE).to(tl.float8e4nv)
        else:
            pb = probabilities.to(qr0.dtype)
            v0 = v0.to(qr0.dtype)
            v1 = v1.to(qr0.dtype)
            v2 = v2.to(qr0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(pb, v0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(pb, v1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(pb, v2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    # One reciprocal per head instead of a divide per output element, as in the hd192
    # prefill kernel. P_SCALE rides out here too, so neither it nor q_scale reaches the
    # output, and both -- like v_scale -- are constant per split and so commute with the
    # LSE-weighted merge.
    if PV_FP8:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum * P_SCALE), 0.0) * v_scale
    else:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum), 0.0) * v_scale
    output0 = acc0 * inv[:, None]
    output1 = acc1 * inv[:, None]
    output2 = acc2 * inv[:, None]
    oh = (group * heads_per_group + heads) * stride_out_head
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (0 * DK + dk)[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (1 * DK + dk)[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (2 * DK + dk)[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    if STORE_LSE:
        tl.store(
            lse_ptr
            + split * stride_lse_split
            + request * stride_lse_token
            + group * heads_per_group
            + heads,
            # running_max/running_sum are base-2; the stored lse is natural log, as the
            # general kernel's is, so convert rather than changing the contract.
            tl.where(
                selected,
                (running_max + tl.log2(running_sum)) * 0.6931471805599453,
                NEGATIVE_INFINITY,
            ),
            mask=head_ok,
        )




def partial(query,key,value,regions,counts,qpos,requests,table,*,rank,window=0,splits=1,
            k_scale=1.,v_scale=1.,block_regions=None,global_region_size=128):
    if global_region_size not in (8,128):
        raise ValueError("only global regions8 and128 are implemented")
    if block_regions is None:
        block_regions=8 if global_region_size==8 else 1
    allowed_blocks=(4,8) if global_region_size==8 else (1,2)
    if rank not in (0,1) or window<0 or splits<1 or block_regions not in allowed_blocks:
        raise ValueError("invalid DCP2 prototype launch")
    nq,nh,hd=query.shape
    if nq<1 or key.ndim!=3 or value.ndim!=3:
        raise ValueError("empty query or invalid cache rank")
    groups=key.shape[1]
    if hd!=192 or query.dtype!=torch.bfloat16 or nh%groups:
        raise ValueError("requires BF16 HD192 GQA")
    if query.device.type!="cuda" or any(x.device!=query.device for x in
        [key,value,regions,counts,qpos,requests,table]):
        raise ValueError("all tensors must share the GPU")
    if key.shape!=value.shape or key.shape[2]!=hd or key.shape[0]%64:
        raise ValueError("cache must have complete local64 pages")
    if key.dtype not in (torch.bfloat16,torch.float8_e4m3fn) or value.dtype!=key.dtype:
        raise ValueError("unsupported cache dtype")
    if regions.dtype!=torch.int32 or regions.ndim!=2 or regions.shape[0]!=nq*groups:
        raise ValueError("regions must be contiguous group-major int32 rows")
    if any(x.dtype!=torch.int32 or not x.is_contiguous() for x in
           [regions,counts,qpos,requests,table]):
        raise ValueError("integer metadata must be contiguous int32")
    if counts.shape!=(nq*groups,) or qpos.shape!=(nq,) or requests.shape!=(nq,) or table.ndim!=2:
        raise ValueError("invalid metadata shapes")
    topk=regions.shape[1]
    if topk<1 or table.shape[1]<1 or table.shape[0]<1 or groups<1 or any(x.stride(-1)!=1 for x in [query,key,value]):
        raise ValueError("invalid empty metadata or feature stride")
    packed=torch.empty_like(regions)
    _pack_striped_regions[(triton.cdiv(regions.numel(),256),)](
        regions,counts,qpos,requests,table,packed,
        ROWS=regions.shape[0],TOPK=topk,QUERIES=nq,REQUESTS=table.shape[0],
        TABLE_WIDTH=table.shape[1],PHYSICAL_PAGES=key.shape[0]//64,GLOBAL_REGION_SIZE=global_region_size,RANK=rank,BLOCK=256,num_warps=4)
    output=torch.empty((splits,nq,nh,hd),dtype=torch.float32,device=query.device)
    lse=torch.empty((splits,nq,nh),dtype=torch.float32,device=query.device)
    _striped_hd192_partial_kernel[(nq,groups,splits)](
        query,key,value,packed,counts,regions,qpos,output,lse,
        query.stride(0),query.stride(1),key.stride(0),key.stride(1),
        value.stride(0),value.stride(1),regions.stride(0),
        output.stride(0),output.stride(1),output.stride(2),
        lse.stride(0),lse.stride(1),hd**-.5*k_scale*1.4426950408889634,
        v_scale,1.,nq,topk=topk,heads_per_group=nh//groups,
        region_size=global_region_size//2,slots_per_split=triton.cdiv(topk,splits),
        BLOCK_H=max(16,triton.next_power_of_2(nh//groups)),BLOCK_R=block_regions,
        DK=64,QK_FP8=False,PV_FP8=False,P_SCALE=256.,STORE_LSE=True,
        RANK=rank,WINDOW=window,num_warps=4,num_stages=1)
    return output,lse,packed
@triton.jit
def _merge_split_states_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_LSE: tl.constexpr,
) -> None:
    """Combine per-split attention states into one, weighting by the partial lse.

    Each split produced an output already divided by its own softmax denominator, so the
    combination is a softmax over the partial log-sum-exps rather than a plain sum:
    ``out = sum_s softmax(lse)_s * out_s`` and ``lse = logsumexp_s(lse_s)``. A split that
    covered no live slot reports ``lse = -inf`` and drops out with weight zero; if *every*
    split does, the row selected nothing and must come back as a zero output with
    ``lse = -inf`` rather than ``0/0``.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    split_ok = splits < num_splits
    dim_ok = dims < head_dim

    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.exp(partial_lse - shift), 0.0)
    total = tl.sum(weights)

    states = tl.load(
        partial_out_ptr
        + splits[:, None] * stride_po_split
        + token * stride_po_token
        + head * stride_po_head
        + dims[None, :],
        mask=split_ok[:, None] & dim_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, None], axis=0) / tl.maximum(total, 1.0e-20)

    tl.store(
        out_ptr + token * stride_out_token + head * stride_out_head + dims,
        merged.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )
    if WRITE_LSE:
        tl.store(
            lse_ptr + token * stride_lse_token + head,
            tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
        )



def merge(outputs,lses,*,dtype=torch.float32):
    if outputs.dtype!=torch.float32 or lses.dtype!=torch.float32:
        raise ValueError("DCP2 partial states must remain FP32")
    ns,nq,nh,hd=outputs.shape
    if lses.shape!=(ns,nq,nh):
        raise ValueError("partial state shapes disagree")
    out=torch.empty((nq,nh,hd),dtype=dtype,device=outputs.device)
    lse=torch.empty((nq,nh),dtype=torch.float32,device=outputs.device)
    _merge_split_states_kernel[(nq,nh)](
        outputs,lses,out,lse,outputs.stride(0),outputs.stride(1),outputs.stride(2),
        lses.stride(0),lses.stride(1),out.stride(0),out.stride(1),lse.stride(0),ns,
        head_dim=hd,BLOCK_S=triton.next_power_of_2(ns),BLOCK_D=triton.next_power_of_2(hd),
        WRITE_LSE=True,num_warps=1 if ns<=2 else 4,num_stages=1)
    return out,lse
