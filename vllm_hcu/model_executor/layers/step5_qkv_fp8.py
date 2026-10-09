# SPDX-License-Identifier: Apache-2.0
"""Online per-output-channel FP8 QKV weights with dynamic per-token activations."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
from vllm_hcu.platforms import envs as henvs
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger("vllm.hcu.step5.qkv_fp8")


@torch.no_grad()
def quantize_qkv_channelwise(
    weight_nn: torch.Tensor,
    scale_mode: str = "amax",
    calibration_inputs: torch.Tensor | None = None,
    calibration_dequantized: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return column-major [K,N] FP8 and FP32 dequantization scales [N,1]."""
    if weight_nn.ndim != 2 or weight_nn.dtype != torch.bfloat16:
        raise ValueError("Online Step5 QKV quantization requires BF16 [K,N] weights")
    if scale_mode not in {"amax", "mse", "output_mse"}:
        raise ValueError("Step5 QKV FP8 scale mode must be amax, mse or output_mse")
    # Quantize each output channel over its full input dimension, not blocks.
    weight_nt = weight_nn.t().float()
    scale = weight_nt.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    if scale_mode == "output_mse":
        if (
            calibration_inputs is None or calibration_dequantized is None
            or calibration_inputs.ndim != 2
            or calibration_inputs.shape != calibration_dequantized.shape
            or calibration_inputs.shape[1] != weight_nn.shape[0]
            or calibration_inputs.shape[0] == 0
            or calibration_inputs.dtype != torch.bfloat16
            or calibration_dequantized.dtype != torch.float32
            or calibration_inputs.device != weight_nn.device
            or calibration_dequantized.device != weight_nn.device
        ):
            raise ValueError("Output-MSE QKV scales require matching BF16/FP32 calibration inputs")
        if not (
            torch.isfinite(calibration_inputs).all()
            and torch.isfinite(calibration_dequantized).all()
        ):
            raise ValueError("QKV calibration inputs must be finite")
        target = calibration_inputs.float() @ weight_nt.t()
    if scale_mode in {"mse", "output_mse"}:
        # Both searches run only at load time. Output-MSE uses the actual
        # per-token activation quantizer on independent document inputs.
        reference_scale = scale
        baseline = (weight_nt / scale).clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn
        ).float() * scale
        if scale_mode == "output_mse":
            best_error = (
                calibration_dequantized @ baseline.t() - target
            ).square().mean(dim=0).unsqueeze(1)
        else:
            best_error = (baseline - weight_nt).square().mean(dim=1, keepdim=True)
        for step in range(41):
            ratio = 0.80 + step * 0.01
            if step == 20:
                continue  # The unchanged amax candidate is already included.
            candidate_scale = reference_scale * ratio
            dequantized = (weight_nt / candidate_scale).clamp(
                -448.0, 448.0
            ).to(torch.float8_e4m3fn).float() * candidate_scale
            if scale_mode == "output_mse":
                error = (
                    calibration_dequantized @ dequantized.t() - target
                ).square().mean(dim=0).unsqueeze(1)
            else:
                error = (dequantized - weight_nt).square().mean(dim=1, keepdim=True)
            improve = error < best_error
            scale = torch.where(improve, candidate_scale, scale)
            best_error = torch.where(improve, error, best_error)
    quantized_nt = (weight_nt / scale).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    ).contiguous()
    return quantized_nt.t(), scale.contiguous()


_CALIBRATION_CACHE: dict[str, dict] = {}


def _load_output_calibration(prefix: str, device: torch.device) -> torch.Tensor:
    path = Path(os.environ.get("VLLM_STEP5_QKV_FP8_CALIBRATION_FILE", ""))
    if not path.is_file():
        raise ValueError("Output-MSE QKV quantization requires a calibration input file")
    key = str(path.resolve())
    data = _CALIBRATION_CACHE.get(key)
    if data is None:
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = Path(get_current_vllm_config().model_config.model)
        model_index = model / "model.safetensors.index.json"
        if (
            data.get("version") != 1
            or data.get("training_dp_ranks") != [0, 1]
            or not model_index.is_file()
            or data.get("model_index_sha256")
            != hashlib.sha256(model_index.read_bytes()).hexdigest()
            or len(data.get("inputs", {})) != 95
        ):
            raise ValueError("Step5 QKV calibration checkpoint or coverage mismatch")
        for name, value in data["inputs"].items():
            if (
                not isinstance(name, str)
                or not isinstance(value, torch.Tensor)
                or value.ndim != 2 or value.shape[1] != 4096
                or value.shape[0] == 0 or value.dtype != torch.bfloat16
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid Step5 QKV calibration group")
        _CALIBRATION_CACHE[key] = data
    if prefix not in data["inputs"]:
        raise ValueError(f"Missing Step5 QKV calibration input group: {prefix}")
    return data["inputs"][prefix].to(device=device).contiguous()


def _reject_quantized_weight_reload(*args, **kwargs) -> None:
    raise RuntimeError(
        "Online Step5 FP8 QKV weights must be loaded from BF16 during model "
        "initialization; restart the service to load a different checkpoint"
    )


class Step5QKVFp8LinearMethod(UnquantizedLinearMethod):
    """Keep the BF16 checkpoint loader, then finalize only this QKV layer."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.scale_mode = os.environ.get("VLLM_STEP5_QKV_FP8_SCALE_MODE", "amax")
        if self.scale_mode not in {"amax", "mse", "output_mse"}:
            raise ValueError("Step5 QKV FP8 scale mode must be amax, mse or output_mse")
        self.kernel = None
        self.use_lightop_quant = False
        small_qkv = os.environ.get("VLLM_STEP5_TP4_SMALL_QKV_FP8", "0")
        if small_qkv not in {"0", "1"}:
            raise ValueError("VLLM_STEP5_TP4_SMALL_QKV_FP8 must be 0 or 1")
        self.tp4_small_qkv_fp8 = small_qkv == "1"


    @torch.no_grad()
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self.kernel is not None:
            # A second finalize without a reload is harmless.
            if layer.weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("Step5 FP8 QKV runtime weight was replaced")
            return
        hidden = layer.input_size
        width = sum(layer.output_partition_sizes)
        if self.tp4_small_qkv_fp8:
            config = get_current_vllm_config()
            parallel = config.parallel_config
            if (hidden != 4096 or width != 3456
                    or parallel.tensor_parallel_size != 4
                    or parallel.data_parallel_size != 2
                    or not parallel.enable_expert_parallel
                    or parallel.decode_context_parallel_size != 1
                    or not henvs.custom_ops_enabled()):
                raise ValueError("Step5 small QKV is qualified only for TP4 DP2 EP8 DCP1 with HCU custom ops")
        original = layer.weight
        if tuple(original.shape) == (hidden, width):
            weight_nn = original
        elif tuple(original.shape) == (width, hidden):
            weight_nn = original.t()
        else:
            raise ValueError(f"Unexpected local Step5 QKV weight shape {original.shape}")
        calibration = None
        dequantized_calibration = None
        if self.scale_mode == "output_mse":
            calibration = _load_output_calibration(self.prefix, weight_nn.device)
            if not henvs.custom_ops_enabled():
                raise ValueError("Output-MSE QKV calibration requires the qualified HCU token quantizer")
            lightop_fp8_runtime.ensure_registered(
                torch.float8_e4m3fn, direct_register_custom_op
            )
            quantized_calibration, calibration_scale = lightop_fp8_runtime.quantize(
                calibration, torch.float8_e4m3fn, direct_register_custom_op
            )
            dequantized_calibration = quantized_calibration.float() * calibration_scale
        weight, scale = quantize_qkv_channelwise(
            weight_nn, self.scale_mode, calibration, dequantized_calibration
        )
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(width, hidden),
            force_kernel=ChannelWiseTorchFP8ScaledMMLinearKernel,
            module_name="Step5 QKV FP8 per-channel / per-token",
        )
        if kernel.fp8_dtype != torch.float8_e4m3fn:
            raise ValueError("Step5 FP8 QKV requires the HCU e4m3fn runtime")
        parameter = nn.Parameter(weight, requires_grad=False)
        parameter.input_dim = 0
        parameter.output_dim = 1
        parameter.weight_loader = _reject_quantized_weight_reload
        layer.weight = parameter
        layer.weight_scale = nn.Parameter(scale, requires_grad=False)
        layer.input_scale = None
        layer.input_scale_ub = None
        kernel.process_weights_after_loading(layer)
        self.kernel = kernel
        self.use_lightop_quant = henvs.custom_ops_enabled()
        if self.use_lightop_quant:
            lightop_fp8_runtime.ensure_registered(
                torch.float8_e4m3fn, direct_register_custom_op
            )
        if self.tp4_small_qkv_fp8:
            if (not self.use_lightop_quant
                    or getattr(type(self.kernel), "_hcu_fp8_backend", None) != "lightop"):
                raise ValueError("Step5 small QKV requires the qualified LightOp FP8 backend")
            arch = torch.cuda.get_device_properties(layer.weight.device).gcnArchName
            if arch.split(":")[0] != "gfx938":
                raise ValueError("Step5 small QKV is qualified only on gfx938")
            from vllm_hcu.ops.step5_small_qkv_fp8 import ensure_registered
            ensure_registered()
            logger.info_once(
                "Step5 TP4 small QKV fusion enabled for M1/4/8/16/32/64; source=%s",
                __file__,
                scope="process",
            )
        logger.info_once(
            "Step5 QKV W8A8 enabled: FP8 per-channel weights, per-token activations; "
            "indexer, gate and output projection retain their original precision.",
            scope="local",
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.kernel is None:
            raise RuntimeError("Step5 FP8 QKV must be finalized after BF16 weight loading")
        if x.shape[0] == 0:
            return torch.empty(
                (*x.shape[:-1], layer.weight.shape[1]), device=x.device,
                dtype=torch.bfloat16,
            )
        if not self.use_lightop_quant:
            return self.kernel.apply_weights(layer, x, bias)
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if self.tp4_small_qkv_fp8 and bias is None:
            # vLLM compiles one symbolic range for Prefill and Decode.  A
            # Python shape branch here specializes to the large tracing input
            # and removes the Decode kernel from that range.  Keep routing
            # inside the opaque op, with the original native path as fallback.
            output = torch.ops.vllm.step5_tp4_small_qkv_fp8(
                x_2d, layer.weight, layer.weight_scale,
            )
            return output.reshape(*original_shape[:-1], layer.weight.shape[1])
        quantized, scale = lightop_fp8_runtime.quantize(
            x_2d, torch.float8_e4m3fn, direct_register_custom_op
        )
        return self.kernel.apply_scaled_mm(
            A=quantized, B=layer.weight, As=scale, Bs=layer.weight_scale,
            out_dtype=torch.bfloat16, bias=bias,
            output_shape=[*original_shape[:-1], layer.weight.shape[1]],
        )


    def apply_prequantized(
        self, layer: nn.Module, quantized: torch.Tensor, scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the existing GEMM without requantizing the fused norm output."""
        if (
            self.kernel is None or not self.use_lightop_quant
            or getattr(type(self.kernel), "_hcu_fp8_backend", None) != "lightop"
        ):
            raise RuntimeError("Step5 prequantized QKV requires finalized LightOp FP8")
        if (
            quantized.ndim != 2 or quantized.dtype != torch.float8_e4m3fn
            or quantized.shape[1] != layer.weight.shape[0]
            or not quantized.is_contiguous()
            or scale.shape != (quantized.shape[0], 1)
            or scale.dtype != torch.float32 or not scale.is_contiguous()
            or quantized.device != layer.weight.device
            or scale.device != quantized.device
        ):
            raise ValueError("Invalid Step5 prequantized QKV activations or scales")
        if quantized.shape[0] == 0:
            return torch.empty(
                (0, layer.weight.shape[1]), device=quantized.device,
                dtype=torch.bfloat16,
            )
        return self.kernel.apply_scaled_mm(
            A=quantized, B=layer.weight, As=scale, Bs=layer.weight_scale,
            out_dtype=torch.bfloat16, bias=bias,
            output_shape=[quantized.shape[0], layer.weight.shape[1]],
        )
