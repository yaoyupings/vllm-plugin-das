# SPDX-License-Identifier: Apache-2.0
"""One launch for calibrated FP8 QKV and original BF16 indexer/gate projections."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma
# Only this kernel receives the HCU Gluon matrix pipeline.
from triton.backends.hcu.compiler import HIPBackend
from triton._C.libtriton import ir,passes,amd,hcu
if not hasattr(HIPBackend, "_step5_mixed_projection_original_pipeline"):
    HIPBackend._step5_mixed_projection_original_pipeline = HIPBackend.gluon_to_ttgir
_gluon_original = HIPBackend._step5_mixed_projection_original_pipeline

def _gluon_hcu_matrix_pipeline(src,metadata,options):
    if src.get_entry_func_name() != "vllm_step5_mixed_projection":
        return _gluon_original(src,metadata,options)
    mod=_gluon_original(src,metadata,options)
    assert options.arch=="gfx938" and not metadata.get("wasp_enabled",False)
    builder=ir.builder(mod.context)
    mod.set_attr("hcu.sched_barrier_before_mmac",builder.get_bool_attr(False))
    mod.set_attr("hcu.sched_barrier_between_a_b_loads",builder.get_bool_attr(False))
    mod.set_attr("hcu.empty_arrive_after_mmac",builder.get_bool_attr(False))
    pm=ir.pass_manager(mod.context);pm.enable_debug()
    passes.ttgpuir.add_coalesce(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm,False)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    hcu.passes.ttgpuir.add_accelerate_matmul(pm,options.arch,options.matrix_instr_nonkdim,options.kpack,options.mmac_layout_force)
    amd.passes.ttgpuir.add_optimize_epilogue(pm)
    amd.passes.ttgpuir.add_optimize_dot_operands(pm,options.arch)
    hcu.passes.ttgpuir.add_mls_encoding_insertion(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm,False)
    amd.passes.ttgpuir.add_hoist_layout_conversions(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_triton_licm(pm)
    hcu.passes.ttgpuir.add_mls_stream_pipeline(pm,options.num_stages,0,False)
    amd.passes.ttgpuir.add_schedule_loops(pm,options.num_stages)
    amd.passes.ttgpuir.add_pipeline(pm,False,False)
    passes.common.add_canonicalizer(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm,False)
    hcu.passes.ttgpuir.add_mls_lowering_pass(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    amd.passes.ttgpuir.add_reorder_instructions(pm)
    if __import__("triton").knobs.amd.use_buffer_ops:
        amd.passes.ttgpuir.add_canonicalize_pointers(pm)
        passes.common.add_canonicalizer(pm)
        hcu.passes.ttgpuir.add_convert_to_buffer_ops(pm,options.arch,__import__('triton').knobs.amd.use_buffer_atomics,__import__('triton').knobs.amd.buffer_ops_analyze_small_tensor_range,__import__('triton').knobs.amd.emit_buffer_ops_offset_assert,__import__('triton').knobs.amd.buffer_ops_use_range_analysis,True)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod,"isolated_gluon_hcu_matrix_pipeline_v1")
    metadata["experimental_gluon_hcu_pipeline"]="v1"
    return mod
if not getattr(HIPBackend,"_step5_mixed_projection_pipeline_installed",False):
    HIPBackend.gluon_to_ttgir=staticmethod(_gluon_hcu_matrix_pipeline)
    HIPBackend._step5_mixed_projection_pipeline_installed=True

@gluon.jit
def vllm_step5_mixed_projection(AQ,AS,X,WQ,WS,WR,Y,M:gl.constexpr,Q:gl.constexpr,R:gl.constexpr,K:gl.constexpr,
                     BMQ:gl.constexpr,BMR:gl.constexpr,BNQ:gl.constexpr,BNR:gl.constexpr,BKQ:gl.constexpr,BKR:gl.constexpr,
                     GROUP_Q:gl.constexpr,GROUP_R:gl.constexpr,PIPE_Q:gl.constexpr,PIPE_R:gl.constexpr,PURE_PHASE:gl.constexpr,
                     GL_WARPS:gl.constexpr,GL_WARP_M_Q:gl.constexpr,GL_TILE_N_Q:gl.constexpr,GL_WARP_M_R:gl.constexpr,GL_TILE_N_R:gl.constexpr,
                     GL_TRANS_Q:gl.constexpr,GL_TRANS_R:gl.constexpr,GL_R_INSTR:gl.constexpr):
    gl.static_assert(True, "vllm_step5_full_mixed_projection_v1_cache_boundary")
    pid=gl.program_id(0)
    qcols=gl.cdiv(Q,BNQ)
    qprograms=gl.cdiv(M,BMQ)*qcols
    if PURE_PHASE==1 or (PURE_PHASE==0 and pid<qprograms):
        qrows=gl.cdiv(M,BMQ)
        qgroup=pid//(GROUP_Q*qcols)
        qfirst=qgroup*GROUP_Q
        qsize=gl.minimum(qrows-qfirst,GROUP_Q)
        qlocal=pid%(GROUP_Q*qcols)
        pm=qfirst+qlocal%qsize
        pn=qlocal//qsize
        qa_layout:gl.constexpr=gl.BlockedLayout([1,8],[2,32],[GL_WARPS,1],[1,0])
        qb_layout:gl.constexpr=gl.BlockedLayout([8,1],[4,16],[1,GL_WARPS],[0,1])
        qo_layout:gl.constexpr=gl.BlockedLayout([1,8],[4,16],[GL_WARPS,1],[1,0])
        qm_layout:gl.constexpr=AMDMFMALayout(3,[16,16,32],GL_TRANS_Q,[GL_WARP_M_Q,GL_WARPS//GL_WARP_M_Q],tiles_per_warp=[1,GL_TILE_N_Q])
        q_rm=pm*BMQ+gl.arange(0,BMQ,layout=gl.SliceLayout(1,qa_layout))
        q_rn=pn*BNQ+gl.arange(0,BNQ,layout=gl.SliceLayout(0,qb_layout))
        q_ka=gl.arange(0,BKQ,layout=gl.SliceLayout(0,qa_layout))
        q_kb=gl.arange(0,BKQ,layout=gl.SliceLayout(1,qb_layout))
        qacc=gl.full((BMQ,BNQ),0,gl.float32,layout=qm_layout)
        for ki in range(0,gl.cdiv(K,BKQ)):
            q_a=gl.load(AQ+q_rm[:,None]*K+(ki*BKQ+q_ka)[None,:],(q_rm[:,None]<M)&((ki*BKQ+q_ka)[None,:]<K))
            q_b=gl.load(WQ+(ki*BKQ+q_kb)[:,None]+q_rn[None,:]*K,((ki*BKQ+q_kb)[:,None]<K)&(q_rn[None,:]<Q))
            q_a_dot=gl.convert_layout(q_a,gl.DotOperandLayout(0,qm_layout,8))
            q_b_dot=gl.convert_layout(q_b,gl.DotOperandLayout(1,qm_layout,8))
            qacc=mfma(q_a_dot,q_b_dot,qacc)
        q_out=gl.convert_layout(qacc,qo_layout)
        q_orow=pm*BMQ+gl.arange(0,BMQ,layout=gl.SliceLayout(1,qo_layout))
        q_ocol=pn*BNQ+gl.arange(0,BNQ,layout=gl.SliceLayout(0,qo_layout))
        sa=gl.load(AS+q_orow,q_orow<M,0)
        sb=gl.load(WS+q_ocol,q_ocol<Q,0)
        q_out=((q_out*sa[:,None])*sb[None,:]).to(gl.bfloat16)
        gl.store(Y+q_orow[:,None]*(Q+R)+q_ocol[None,:],q_out,(q_orow[:,None]<M)&(q_ocol[None,:]<Q))
    else:
        rid=pid if PURE_PHASE==2 else pid-qprograms
        rcols=gl.cdiv(R,BNR)
        rrows=gl.cdiv(M,BMR)
        rgroup=rid//(GROUP_R*rcols)
        rfirst=rgroup*GROUP_R
        rsize=gl.minimum(rrows-rfirst,GROUP_R)
        rlocal=rid%(GROUP_R*rcols)
        r_pm=rfirst+rlocal%rsize
        r_pn=rlocal//rsize
        ra_layout:gl.constexpr=gl.BlockedLayout([1,8],[2,32],[GL_WARPS,1],[1,0])
        rb_layout:gl.constexpr=gl.BlockedLayout([8,1],[4,16],[1,GL_WARPS],[0,1])
        ro_layout:gl.constexpr=gl.BlockedLayout([1,8],[4,16],[GL_WARPS,1],[1,0])
        rm_layout:gl.constexpr=AMDMFMALayout(3,[GL_R_INSTR,GL_R_INSTR,16 if GL_R_INSTR==16 else 8],GL_TRANS_R,[GL_WARP_M_R,GL_WARPS//GL_WARP_M_R],tiles_per_warp=[1,GL_TILE_N_R])
        r_rm=r_pm*BMR+gl.arange(0,BMR,layout=gl.SliceLayout(1,ra_layout))
        r_rn=r_pn*BNR+gl.arange(0,BNR,layout=gl.SliceLayout(0,rb_layout))
        r_ka=gl.arange(0,BKR,layout=gl.SliceLayout(0,ra_layout))
        r_kb=gl.arange(0,BKR,layout=gl.SliceLayout(1,rb_layout))
        racc=gl.full((BMR,BNR),0,gl.float32,layout=rm_layout)
        for r_ki in range(0,gl.cdiv(K,BKR)):
            r_a=gl.load(X+r_rm[:,None]*K+(r_ki*BKR+r_ka)[None,:],(r_rm[:,None]<M)&((r_ki*BKR+r_ka)[None,:]<K),0)
            r_b=gl.load(WR+(r_ki*BKR+r_kb)[:,None]+r_rn[None,:]*K,((r_ki*BKR+r_kb)[:,None]<K)&(r_rn[None,:]<R),0)
            r_a_dot=gl.convert_layout(r_a,gl.DotOperandLayout(0,rm_layout,4))
            r_b_dot=gl.convert_layout(r_b,gl.DotOperandLayout(1,rm_layout,4))
            racc=mfma(r_a_dot,r_b_dot,racc)
        r_out=gl.convert_layout(racc,ro_layout).to(gl.bfloat16)
        r_orow=r_pm*BMR+gl.arange(0,BMR,layout=gl.SliceLayout(1,ro_layout))
        r_ocol=r_pn*BNR+gl.arange(0,BNR,layout=gl.SliceLayout(0,ro_layout))
        gl.store(Y+r_orow[:,None]*(Q+R)+Q+r_ocol[None,:],r_out,(r_orow[:,None]<M)&(r_ocol[None,:]<R))
