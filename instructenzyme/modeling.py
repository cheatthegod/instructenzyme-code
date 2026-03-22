from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from llava.model.multimodal_projector.builder import FixedQueryCrossAttentionProjector

IGNORE_INDEX = -100


class InstructEnzymeStage1Model(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        structure_hidden_size: int = 128,
        num_queries: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        pos_encoding: str = "1d",
        use_query_pos: bool = True,
        use_input_pos: bool = True,
        projector_use_post_proj: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.backbone.config.use_cache = False
        hidden_size = getattr(self.backbone.config, "hidden_size", None) or getattr(self.backbone.config, "embed_dim")

        projector_cfg = SimpleNamespace(
            mm_hidden_size=structure_hidden_size,
            hidden_size=hidden_size,
            mm_patch_merge_type="flat",
            mm_projector_num_queries=num_queries,
            mm_projector_num_heads=num_heads,
            mm_projector_num_layers=num_layers,
            mm_projector_ffn_mult=ffn_mult,
            mm_projector_dropout=dropout,
            mm_projector_pos_encoding=pos_encoding,
            mm_projector_use_query_pos=use_query_pos,
            mm_projector_use_input_pos=use_input_pos,
            mm_projector_use_post_proj=projector_use_post_proj,
        )
        self.projector = FixedQueryCrossAttentionProjector(projector_cfg)
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.structure_hidden_size = structure_hidden_size

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.projector.parameters():
            parameter.requires_grad = True

    def gradient_checkpointing_enable(self):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def get_trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]

    def get_input_embeddings(self):
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "wte"):
            return self.backbone.transformer.wte
        if hasattr(self.backbone, "get_input_embeddings"):
            try:
                return self.backbone.get_input_embeddings()
            except NotImplementedError:
                pass
        raise AttributeError("Could not locate token embedding layer on backbone")

    def encode_structure(self, structure_embeddings: torch.Tensor, structure_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if structure_attention_mask is not None:
            key_padding_mask = ~structure_attention_mask.bool()
        return self.projector(structure_embeddings, key_padding_mask=key_padding_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        structure_embeddings: torch.Tensor,
        structure_attention_mask: Optional[torch.Tensor] = None,
    ):
        prompt_embeds = self.encode_structure(structure_embeddings, structure_attention_mask)
        token_embeds = self.get_input_embeddings()(input_ids)

        full_inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        prompt_attention = torch.ones(
            (attention_mask.shape[0], prompt_embeds.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
        prompt_labels = torch.full(
            (labels.shape[0], prompt_embeds.shape[1]),
            IGNORE_INDEX,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat([prompt_labels, labels], dim=1)

        return self.backbone(
            input_ids=None,
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
        )

    def save_projector(self, output_dir: str | Path, extra_state: Optional[dict] = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "projector": self.projector.state_dict(),
            "num_queries": self.num_queries,
            "structure_hidden_size": self.structure_hidden_size,
            "hidden_size": self.hidden_size,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, output_dir / "projector.pt")
