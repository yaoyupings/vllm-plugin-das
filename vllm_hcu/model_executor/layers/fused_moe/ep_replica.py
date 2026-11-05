"""Redundant experts for the EP fused MoE decode path (VLLM_HCU_EP_REPLICA=1).

At decode the EP combine (all-reduce over EP ranks) waits for the slowest rank's MoE, and
routing skew makes that rank ~50 us/layer slower than the mean. Here every rank also holds
replicas of hot experts homed elsewhere (VLLM_HCU_EP_REPLICA_PLACEMENT, a JSON file
{"rep_e": [...], "rep_q": [...]}: replica j is global expert rep_e[j] held by EP rank
rep_q[j]). Per step, _rep_sort_kernel (run identically on every rank, so no communication)
splits each replicated expert's routes between home and holder at M-block granularity by a
few damped Jacobi rounds on per-rank block loads, then emits this rank's sorted layout over
its local slots [44 home | replicas] directly, replacing ep_sorting.

Replica weights are fetched once per layer at weight postprocessing (attach(), called from
the fp8 oracle's kernel-format conversion) by broadcasting the rows from their home ranks;
the layer's w13/w2 become views of the home part of the extended [EPR + RPR, ...] tensors,
so no second copy of the home experts is kept. Costs ~0.14 GiB/layer of KV memory at RPR=8.
Only small batches (<= MAX_ROUTES routes, i.e. decode) take the replica path.
"""
import json
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("VLLM_HCU_EP_REPLICA", "0") == "1"
PLACEMENT = os.environ.get("VLLM_HCU_EP_REPLICA_PLACEMENT", "")
ROUNDS = int(os.environ.get("VLLM_HCU_EP_REPLICA_ROUNDS", "2"))
DAMP = float(os.environ.get("VLLM_HCU_EP_REPLICA_DAMP", "0.5"))
MAX_ROUTES = 2048
EP2 = 512


@triton.jit
def _rep_sort_kernel(ids_ptr, w_ptr, emapg_ptr, rep_of_ptr, rep_e_ptr, rep_q_ptr, slot_g_ptr,
                     slot_j_ptr, slot_h_ptr, scr_ptr, sids_ptr, sw_ptr, eids_ptr, nvalid_ptr,
                     lids_ptr, M, E: tl.constexpr, EPR: tl.constexpr, R: tl.constexpr,
                     TOPK: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr, EP2: tl.constexpr,
                     NRP: tl.constexpr, NREP: tl.constexpr, NLOC: tl.constexpr,
                     NLP: tl.constexpr, ROUNDS: tl.constexpr, DAMP: tl.constexpr):
    # One program per local slot x (+1 that marks routes to experts not held here).
    # Writes the slot's sorted rows, its M-block expert ids and the slot-space topk ids.
    x = tl.program_id(0)
    r = tl.arange(0, NT)
    rv = r < M * TOPK
    e = tl.load(ids_ptr + r, mask=rv, other=E)
    if x == NLOC:
        held = tl.load(emapg_ptr + tl.minimum(e, E - 1)) >= 0
        tl.store(lids_ptr + r, tl.zeros((NT,), tl.int32) + NLOC, mask=rv & ~held)
        return
    cnt = tl.histogram(e, EP2)
    xs = tl.arange(0, EP2)
    sc = scr_ptr + x * EP2
    tl.store(sc + xs, cnt)
    blk = tl.where(xs < E, (cnt + BM - 1) // BM, 0)
    rep_of_x = tl.load(rep_of_ptr + xs)
    ranks = tl.arange(0, R)
    # per-rank M-blocks of the unreplicated experts
    base = tl.sum(tl.where((xs[None, :] // EPR == ranks[:, None]) & (rep_of_x[None, :] < 0),
                           blk[None, :], 0), axis=1).to(tl.float32)
    tl.debug_barrier()
    js = tl.arange(0, NRP)
    jm = js < NREP
    re = tl.load(rep_e_ptr + js, mask=jm, other=0)
    rq = tl.load(rep_q_ptr + js, mask=jm, other=-1)
    rh = tl.where(jm, re // EPR, -1)
    rc = tl.load(sc + re, mask=jm, other=0)
    nb = ((rc + BM - 1) // BM).to(tl.float32)
    hm = rh[:, None] == ranks[None, :]
    qm = rq[:, None] == ranks[None, :]
    # xh: blocks of replica j left at home; move toward equalizing home/holder loads
    xh = nb
    for _ in tl.static_range(ROUNDS):
        ld = base + tl.sum(tl.where(hm, xh[:, None], 0.0) + tl.where(qm, (nb - xh)[:, None], 0.0), axis=0)
        lh = tl.sum(tl.where(hm, ld[None, :], 0.0), axis=1) - xh
        lq = tl.sum(tl.where(qm, ld[None, :], 0.0), axis=1) - (nb - xh)
        tgt = tl.minimum(tl.maximum((lq - lh + nb) * 0.5, 0.0), nb)
        xh = xh + DAMP * (tgt - xh)
    kept = tl.minimum(rc, (xh + 0.5).to(tl.int32) * BM)
    ys = tl.arange(0, NLP)
    ym = ys < NLOC
    sg = tl.load(slot_g_ptr + ys, mask=ym, other=E + 1)
    sj = tl.load(slot_j_ptr + ys, mask=ym, other=-1)
    sh = tl.load(slot_h_ptr + ys, mask=ym, other=1)
    scnt = tl.load(sc + sg)
    kj = tl.sum(tl.where(sj[:, None] == js[None, :], kept[None, :], 0), axis=1)
    ycnt = tl.where(sj < 0, scnt, tl.where(sh == 1, kj, scnt - kj))
    ycnt = tl.where(ym, ycnt, 0)
    padded = (ycnt + BM - 1) // BM * BM
    start = tl.sum(tl.where(ys < x, padded, 0), axis=0)
    n = tl.sum(tl.where(ys == x, ycnt, 0), axis=0)
    npad = (n + BM - 1) // BM * BM
    gx = tl.load(slot_g_ptr + x)
    jx = tl.load(slot_j_ptr + x)
    hx = tl.load(slot_h_ptr + x)
    kx = tl.sum(tl.where(js == jx, kept, 0), axis=0)
    is_e = rv & (e == gx)
    o = tl.cumsum(is_e.to(tl.int32), axis=0) - 1
    # home keeps the first kept routes of a replicated expert, the holder the rest
    mine = is_e & ((jx < 0) | ((hx == 1) & (o < kx)) | ((hx == 0) & (o >= kx)))
    pos = start + tl.cumsum(mine.to(tl.int32), axis=0) - 1
    tok = r // TOPK
    tl.store(sids_ptr + pos, tok | ((r - tok * TOPK) << 24), mask=mine)
    tl.store(sw_ptr + pos, tl.load(w_ptr + r, mask=mine, other=0.0), mask=mine)
    tl.store(lids_ptr + r, tl.where(mine, x, NLOC), mask=is_e)
    j = tl.arange(0, BM)
    pm = n + j < npad
    tl.store(sids_ptr + start + n + j, tl.zeros((BM,), tl.int32) + (M | (TOPK << 24)), mask=pm)
    tl.store(sw_ptr + start + n + j, tl.zeros((BM,), tl.float32), mask=pm)
    for b in range(start // BM, (start + npad) // BM):
        tl.store(eids_ptr + b, x)
    if x == 0:
        tl.store(nvalid_ptr, tl.sum(padded, axis=0))


class RepTables:
    """Device tables of one placement, as seen from EP rank `me`."""

    def __init__(self, rep_e, rep_q, num_experts, ep_size, me, device):
        assert len(rep_e) == len(rep_q) and num_experts % ep_size == 0
        E, R = num_experts, ep_size
        EPR = E // R
        assert E < EP2
        nrep = len(rep_e)
        for e, q in zip(rep_e, rep_q):
            assert 0 <= e < E and 0 <= q < R and e // EPR != q, (e, q)
        assert len(set(rep_e)) == nrep, "an expert may have at most one replica"
        self.E, self.R, self.EPR, self.me, self.nrep = E, R, EPR, me, nrep
        self.NRP = max(16, triton.next_power_of_2(nrep))
        rep_of = [-1] * EP2
        for j, e in enumerate(rep_e):
            rep_of[e] = j
        self.rpr = max(sum(q == r for q in rep_q) for r in range(R))
        self.nloc = EPR + self.rpr
        mine = [j for j, q in enumerate(rep_q) if q == me]
        self.mine_j = mine  # replica j held in local slot EPR + k, k = mine.index(j)
        ex = list(range(me * EPR, (me + 1) * EPR)) + [rep_e[j] for j in mine]
        sg = ex + [E + 1] * (self.nloc - len(ex))
        sj = [rep_of[e] if e < E else -1 for e in sg]
        sh = [1] * EPR + [0] * (self.nloc - EPR)
        emapg = [-1] * E
        for i, e in enumerate(ex):
            emapg[e] = i
        t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
        pad = lambda v: t(list(v) + [0] * (self.NRP - nrep))
        self.rep_e_l, self.rep_q_l = list(rep_e), list(rep_q)
        self.rep_of, self.rep_e, self.rep_q = t(rep_of), pad(rep_e), pad(rep_q)
        self.emapg, self.sg, self.sj, self.sh = t(emapg), t(sg), t(sj), t(sh)
        # slot-space expert map: slot i -> i, the "not held" slot nloc -> -1
        self.emap = t(list(range(self.nloc)) + [-1])
        self.mask = (self.emap >= 0).to(torch.int32)
        self.scr = torch.empty((self.nloc + 1) * EP2, dtype=torch.int32, device=device)

    def sort_fn(self, ids, lids):
        """ep_sorting replacement over the local slots; also fills lids (slot-space ids)."""

        def fn(topk_ids, topk_weights, expert_mask, G, E, bm, expert_map=None):
            assert topk_ids is lids
            Mx, topk = ids.shape
            routes = Mx * topk
            bound = triton.cdiv(routes + min(routes, E) * (bm - 1), bm) * bm
            dev = ids.device
            sids = torch.empty((bound,), dtype=torch.int32, device=dev)
            sw = torch.empty((bound,), dtype=torch.float32, device=dev)
            eids = torch.empty((bound // bm,), dtype=torch.int32, device=dev)
            nvalid = torch.empty((1,), dtype=torch.int32, device=dev)
            _rep_sort_kernel[(self.nloc + 1,)](
                ids, topk_weights, self.emapg, self.rep_of, self.rep_e, self.rep_q, self.sg,
                self.sj, self.sh, self.scr, sids, sw, eids, nvalid, lids, Mx, E=self.E,
                EPR=self.EPR, R=self.R, TOPK=topk, BM=bm, NT=triton.next_power_of_2(routes),
                EP2=EP2, NRP=self.NRP, NREP=self.nrep, NLOC=self.nloc,
                NLP=triton.next_power_of_2(self.nloc), ROUNDS=ROUNDS, DAMP=DAMP, num_warps=8)
            return sids, sw, eids, nvalid, bound

        return fn


_placement = None


def load_placement():
    global _placement
    if _placement is None:
        p = json.load(open(PLACEMENT))
        _placement = (list(map(int, p["rep_e"])), list(map(int, p["rep_q"])))
    return _placement


class ReplicaState:
    """Extended weights + tables of one MoE layer on this rank."""

    def __init__(self, w1, w2, w1_scale, w2_scale, tabs):
        self.w1, self.w2, self.w1_scale, self.w2_scale, self.tabs = w1, w2, w1_scale, w2_scale, tabs


def _bcast_rows(t, rows, src, group, ranks):
    """t[rows] from EP rank src to every rank (fp8 sent as bytes)."""
    shape = (len(rows),) + tuple(t.shape[1:])
    if group.rank() == src:
        buf = t[torch.tensor(rows, device=t.device)].contiguous()
    else:
        buf = torch.empty(shape, dtype=t.dtype, device=t.device)
    raw = buf.view(torch.uint8) if buf.element_size() == 1 else buf
    torch.distributed.broadcast(raw, src=ranks[src], group=group)
    return buf


def build(w1, w2, w1_scale, w2_scale, expert_map):
    """Replica state for one layer, or None if this layer can't use replicas.

    Collective over the EP group: every rank must call it for the same layer at the same
    time, with the same outcome of the shape checks (true: they see identical shapes).
    """
    from vllm.distributed import get_ep_group

    epg = get_ep_group()
    group, ranks, R = epg.device_group, epg.ranks, epg.world_size
    me = epg.rank_in_group
    E = expert_map.numel()
    EPR = w1.shape[0]
    if R * EPR != E or w2.shape[0] != EPR or w1_scale.shape[0] != EPR \
            or w2_scale.shape[0] != EPR:
        return None
    home = torch.nonzero(expert_map >= 0).flatten()
    # linear placement assumed: rank me holds experts me*EPR .. me*EPR+EPR-1 in order
    want = torch.arange(me * EPR, (me + 1) * EPR, device=expert_map.device)
    if not (home.numel() == EPR and torch.equal(home, want)
            and torch.equal(expert_map[home].long(), torch.arange(EPR, device=home.device))):
        raise RuntimeError("VLLM_HCU_EP_REPLICA needs the linear expert placement")
    rep_e, rep_q = load_placement()
    tabs = RepTables(rep_e, rep_q, E, R, me, w1.device)
    nloc = tabs.nloc

    def ext(t):
        x = torch.empty((nloc,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
        x[:EPR].copy_(t)
        return x

    xs = [ext(t) for t in (w1, w2, w1_scale, w2_scale)]
    for src in range(R):
        js = [j for j, e in enumerate(rep_e) if e // EPR == src]
        if not js:
            continue
        rows = [rep_e[j] - src * EPR for j in js]
        for t, x in zip((w1, w2, w1_scale, w2_scale), xs):
            buf = _bcast_rows(t, rows, src, group, ranks)
            for i, j in enumerate(js):
                if rep_q[j] == me:
                    x[EPR + tabs.mine_j.index(j)].copy_(buf[i])
            del buf
    # unused slots (rank with fewer replicas) never get routes; leave them as zeros
    for x in xs:
        x[EPR + len(tabs.mine_j):].zero_()
    return ReplicaState(*xs, tabs)


# ReplicaState by the data_ptr of the layer's (rebound) w13 weight
STATES = {}
_announced = False


def attach(layer, w13, w2, w13_scale, w2_scale):
    """Weight postprocessing hook: build this layer's replicas and return (w13, w2) to
    register, views of the home part of the extended weights.

    Runs at load rather than lazily in the profile run: there the freed 44-expert weights
    get interleaved with activations and ~20 GiB of torch segments can't be released,
    which the KV sizing then counts as used. Collective over the EP group (every rank
    postprocesses the same layers in the same order)."""
    global _announced
    em = getattr(layer, "_expert_map", None)
    if not ENABLED or em is None or w13.data_ptr() in STATES:
        return w13, w2
    # release the previous layer's freed weights first so the extended tensors get
    # fresh segments instead of splitting cached ones
    torch.cuda.empty_cache()
    rep = build(w13, w2, w13_scale, w2_scale, em.to(w13.device))
    if rep is None:
        return w13, w2
    STATES[rep.w1.data_ptr()] = rep
    if not _announced:
        _announced = True
        print(f"[ep_replica] on: nloc={rep.tabs.nloc} placement={PLACEMENT}", flush=True)
    return rep.w1[:w13.shape[0]], rep.w2[:w2.shape[0]]
