# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project.
"""Step5 three-module MTP draft using the checkpoint's BF16 dense blocks.

Each predictor has its own normalization, embedding/hidden projection, sliding
attention block and output head. The target's FP32 residual semantics are
retained; draft layers stay unquantized, as serialized in the checkpoint.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .step5 import Step5DecoderLayer, _fp32_gemma_rms_norm


def _pad_per_layer_list(config, attr: str, layer_idx: int) -> None:
    """Extend a per-layer config list to cover the MTP layer index.

    Step5Config normally retains the checkpoint lists, including MTP entries.
    This compatibility fallback extends a shortened nonempty list with its final
    value; it is not used when the loaded config already covers the MTP index.
    """
    vals = getattr(config, attr, None)
    if isinstance(vals, list) and vals and len(vals) <= layer_idx:
        setattr(config, attr, vals + [vals[-1]] * (layer_idx + 1 - len(vals)))


class Step5SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = _fp32_gemma_rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        # shared_head.output is BF16 in the checkpoint -- never quantized.
        self.output = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "output"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The decoder layer returns an fp32 residual stream; narrow to the
        # head's dtype before the lm_head matmul (matches the target model's
        # "cast only at lm_head" convention).
        return self.norm(hidden_states).to(self.output.weight.dtype)


class Step5MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self, vllm_config: VllmConfig, prefix: str, layer_idx: int
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.layer_idx = layer_idx
        compute_dtype = vllm_config.model_config.dtype

        self.enorm = _fp32_gemma_rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = _fp32_gemma_rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=compute_dtype,
        )
        self.shared_head = Step5SharedHead(
            config=config, prefix=f"{prefix}.shared_head", quant_config=None
        )
        # Preserve full checkpoint lists. Extend only shortened compatibility
        # configs before constructing this predictor's decoder layer.
        _pad_per_layer_list(config, "layer_types", self.layer_idx)
        _pad_per_layer_list(config, "partial_rotary_factors", self.layer_idx)
        _pad_per_layer_list(config, "rope_theta", self.layer_idx)
        _pad_per_layer_list(config, "use_rope_layers", self.layer_idx)
        # Reuse the target decoder's dense-MLP branch for each MTP layer.
        # The prefix carries this predictor's index (92, 93 or 94), so attention
        # reads its own per-layer configuration.
        self.mtp_block = Step5DecoderLayer(
            vllm_config, prefix=f"{prefix}.mtp_block"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # The token at position 0 is prompt-side; its embedding is not needed.
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0, inputs_embeds
        )
        normed_embeds = self.enorm(inputs_embeds)
        normed_hidden = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([normed_embeds, normed_hidden], dim=-1).to(self.eh_proj.weight.dtype)
        )
        hidden_states = self.mtp_block(
            positions=positions, hidden_states=hidden_states
        )
        return hidden_states


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1) or 1
        self.current_step_idx = 0

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Step5MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}", idx
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        self.current_step_idx = current_step_idx
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int | None = None
    ) -> torch.Tensor:
        current_step_idx = (self.current_step_idx if spec_step_idx is None
                            else spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.output, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile
class Step5MTP(nn.Module, SupportsPP):
    def process_weights_after_loading(self) -> None:
        from .step5 import prepare_model_projections
        prepare_model_projections(self)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.draft_model_config is not None:
            vllm_config = vllm_config.with_hf_config(
                speculative.draft_model_config.hf_config, architectures=["Step5MTP"]
            )
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = Step5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )
        self.use_pp = vllm_config.parallel_config.pipeline_parallel_size > 1

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_mtp_step(self, spec_step_idx: int) -> None:
        self.model.current_step_idx = spec_step_idx % self.model.num_mtp_layers

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | IntermediateTensors:
        if hidden_states is None:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int | None = None
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Mirrors Step5Model.load_weights' stacked-shard handling for the one
        # dense decoder layer, plus the MTP-specific top-level tensors.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        start = self.model.mtp_start_layer_idx
        layer_prefixes = tuple(
            f"model.layers.{idx}."
            for idx in range(start, start + self.model.num_mtp_layers)
        )
        loaded_shards: set[tuple[str, str | int]] = set()

        for name, loaded_weight in weights:
            full_name = name if name.startswith("model.") else f"model.{name}"

            if full_name.startswith("model.embed_tokens."):
                target = full_name
            elif full_name.startswith(layer_prefixes):
                layer_prefix = next(p for p in layer_prefixes if full_name.startswith(p))
                rest = full_name[len(layer_prefix):]
                if rest.startswith("transformer.shared_head."):
                    target = layer_prefix + "shared_head." + rest[len("transformer.shared_head."):]
                elif rest.startswith(
                    ("self_attn.", "mlp.", "input_layernorm.", "post_attention_layernorm.")
                ):
                    target = layer_prefix + "mtp_block." + rest
                elif rest.startswith(("enorm.", "hnorm.", "eh_proj.")):
                    target = full_name
                else:
                    continue
            else:
                # Everything else belongs to the target model.
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in target:
                    continue
                target = target.replace(weight_name, param_name)
                param = params_dict[target]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_shards.add((target, shard_id))
                break
            else:
                if target.endswith(".bias") and target not in params_dict:
                    continue
                param = params_dict[target]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(target)

        optional_suffixes = (".k_scale", ".v_scale", ".q_scale", ".prob_scale")
        missing = [n for n in params_dict
                   if n not in loaded_params and not n.endswith(optional_suffixes)]
        for name in params_dict:
            shards = ("q", "k", "v") if ".qkv_proj." in name else (
                (0, 1) if ".gate_up_proj." in name else ()
            )
            missing.extend(f"{name}[{shard}]" for shard in shards
                           if (name, shard) not in loaded_shards)
        if missing:
            raise ValueError(f"Step5 MTP parameters missing from checkpoint: {missing}")
        return loaded_params
