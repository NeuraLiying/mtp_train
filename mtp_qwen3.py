import copy
import inspect
import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Model
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPastMTP(ModelOutput):
    """Model output class for MTP with multiple heads"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_mtp1: torch.FloatTensor = None
    logits_mtp2: torch.FloatTensor = None
    # Added MTP3 support
    logits_mtp3: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    past_key_values_mtp1: Optional[List[torch.FloatTensor]] = None
    past_key_values_mtp2: Optional[List[torch.FloatTensor]] = None
    # Added MTP3 support
    past_key_values_mtp3: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# Structure fix 1/3: Define MTP head as standard nn.Module
class Qwen3MTP(nn.Module):
    """MTP component module"""
    def __init__(self, config: Qwen3Config, layer_idx: int, embed_tokens: nn.Embedding, lm_head: nn.Linear):
        super().__init__()
        # MTP module specific layers
        self.token_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = Qwen3DecoderLayer(config, layer_idx)
        self.final_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared weight fix: Direct reference to main model's shared layers
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward(self, 
                main_hidden_states: torch.Tensor,
                mtp_input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs):
        """Forward pass for MTP head"""
        
        token_embeds = self.embed_tokens(mtp_input_ids)
        norm_token_embeds = self.token_norm(token_embeds)
        norm_main_hidden = self.hidden_norm(main_hidden_states)
        
        concatenated_features = torch.cat([norm_main_hidden, norm_token_embeds], dim=-1)
        projected_hidden = self.input_proj(concatenated_features)
        
        layer_outputs = self.layers(
            projected_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        decoder_hidden_states = layer_outputs[0]

        final_hidden_states = self.final_layernorm(decoder_hidden_states)
        logits = self.lm_head(final_hidden_states)
        
        return logits, final_hidden_states

# Structure fix 2/3: Use correct inheritance and refactor __init__
class Qwen3ForCausalLMWithMTP(Qwen3PreTrainedModel):
    """Qwen3 MTP model based on MiMo architecture"""
    
    def __init__(self, config: Qwen3Config, **kwargs):
        super().__init__(config, **kwargs)
        
        # Explicitly define main model and LM head
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shared weight fix: Pass shared layers to MTP modules during initialization
        self.MTP1 = Qwen3MTP(config, 1, self.model.embed_tokens, self.lm_head)
        self.MTP2 = Qwen3MTP(config, 2, self.model.embed_tokens, self.lm_head)
        # Added MTP3
        self.MTP3 = Qwen3MTP(config, 3, self.model.embed_tokens, self.lm_head)
        
        # Updated to 3 for three MTP heads
        self.num_assistant_tokens = 3
        self.loss_details = {}

        # Important: Call post_init() to complete weight initialization and binding
        self.post_init()

    def state_dict(self, *args, **kwargs):
        """
        Save fix: Override state_dict to remove duplicate shared weights in MTP heads
        to pass Hugging Face's save checks.
        """
        state_dict = super().state_dict(*args, **kwargs)
        # Remove references to embed_tokens and lm_head weights in MTP modules,
        # keep only one copy in the main model
        keys_to_remove = [
            k for k in state_dict
            if k.startswith("MTP1.embed_tokens.") or k.startswith("MTP1.lm_head.") or \
               k.startswith("MTP2.embed_tokens.") or k.startswith("MTP2.lm_head.") or \
               k.startswith("MTP3.embed_tokens.") or k.startswith("MTP3.lm_head.") # Added MTP3
        ]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def freeze_main_model(self):
        """Freeze all parameters of main model (including shared embed and lm_head)"""
        print("Freezing main model, embedding and lm_head parameters...")
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.lm_head.weight.requires_grad = False
        
        print("Parameters frozen. Trainable parameters:")
        total_trainable_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  - {name} ({param.numel()})")
                total_trainable_params += param.numel()
        print(f"  Total trainable parameters: {total_trainable_params:,}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """Forward pass - switches between training and inference modes"""
        if labels is not None:
            return self._forward_training(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )
        else:
            return self._forward_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=return_dict
            )

    def _forward_training(self, input_ids, attention_mask, position_ids, labels):
        """Online training method"""
        with torch.no_grad():
            self.model.eval()
            hidden_seq = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).last_hidden_state.detach()
            self.model.train()

        B, L, H = hidden_seq.shape
        loss_fn = nn.CrossEntropyLoss()
        
        # Modified minimum length check for 3 MTP heads
        if L < 5:
            return CausalLMOutputWithPastMTP(loss=torch.zeros(..., requires_grad=True))

        # Process for 3 MTP heads
        main_hidden_for_mtp = hidden_seq[:, :-4, :]
        batch_size, seq_len, _ = main_hidden_for_mtp.shape
        
        
        _attention_mask_for_mtp = attention_mask[:, :-4] if attention_mask is not None else None
        _4d_attention_mask_for_mtp = None
        if _attention_mask_for_mtp is not None and _attention_mask_for_mtp.any():
            _4d_attention_mask_for_mtp = _prepare_4d_causal_attention_mask(
                _attention_mask_for_mtp, (batch_size, seq_len), main_hidden_for_mtp, 0
            )

        # Create position_ids for MTP modules to avoid pollution from main input position_ids
        
        _position_ids_for_mtp = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # --- MTP1 Training ---
        # Training/inference alignment: Create corresponding position embeddings for each MTP head
        pos_ids_mtp1 = _position_ids_for_mtp
        cos1, sin1 = self.model.rotary_emb(main_hidden_for_mtp, pos_ids_mtp1)
        position_embeddings_mtp1 = (cos1, sin1)

        # Modified slicing
        mtp1_input_ids = input_ids[:, 1:-3]
        mtp1_labels = labels[:, 2:-2]
        
        logits1, _ = self.MTP1(
            main_hidden_states=main_hidden_for_mtp,
            mtp_input_ids=mtp1_input_ids,
            attention_mask=_4d_attention_mask_for_mtp,
            position_ids=pos_ids_mtp1,
            position_embeddings=position_embeddings_mtp1,
        )
        loss1 = torch.tensor(0.0, device=logits1.device, dtype=torch.float32)
        if (mtp1_labels != -100).any():
            loss1 = loss_fn(logits1.to(torch.float32).reshape(-1, self.config.vocab_size), mtp1_labels.reshape(-1))
            self.loss_details["mtp1_loss"] = loss1.item()

        # --- MTP2 Training ---
        # Training/inference alignment: Create shifted position embeddings for MTP2
        pos_ids_mtp2 = _position_ids_for_mtp + 1
        cos2, sin2 = self.model.rotary_emb(main_hidden_for_mtp, pos_ids_mtp2)
        position_embeddings_mtp2 = (cos2, sin2)

        with torch.no_grad():
            draft_token_1_for_mtp2 = torch.argmax(logits1, dim=-1).detach()

        
        mtp2_labels = labels[:, 3:-1]

        logits2, _ = self.MTP2(
            main_hidden_states=main_hidden_for_mtp,
            mtp_input_ids=draft_token_1_for_mtp2,
            attention_mask=_4d_attention_mask_for_mtp,
            position_ids=pos_ids_mtp2, 
            position_embeddings=position_embeddings_mtp2,
        )
        loss2 = torch.tensor(0.0, device=logits2.device, dtype=torch.float32)
        if (mtp2_labels != -100).any():
            loss2 = loss_fn(logits2.to(torch.float32).reshape(-1, self.config.vocab_size), mtp2_labels.reshape(-1))
            self.loss_details["mtp2_loss"] = loss2.item()
        
        # --- MTP3 Training ---
        # Training/inference alignment: Create shifted position embeddings for MTP3
        pos_ids_mtp3 = _position_ids_for_mtp + 2
        cos3, sin3 = self.model.rotary_emb(main_hidden_for_mtp, pos_ids_mtp3)
        position_embeddings_mtp3 = (cos3, sin3)
        
        with torch.no_grad():
            draft_token_2_for_mtp3 = torch.argmax(logits2, dim=-1).detach()
        
        mtp3_labels = labels[:, 4:]

        logits3, _ = self.MTP3(
            main_hidden_states=main_hidden_for_mtp,
            mtp_input_ids=draft_token_2_for_mtp3,
            attention_mask=_4d_attention_mask_for_mtp,
            position_ids=pos_ids_mtp3,
            position_embeddings=position_embeddings_mtp3,
        )
        loss3 = torch.tensor(0.0, device=logits3.device, dtype=torch.float32)
        if (mtp3_labels != -100).any():
            loss3 = loss_fn(logits3.to(torch.float32).reshape(-1, self.config.vocab_size), mtp3_labels.reshape(-1))
            self.loss_details["mtp3_loss"] = loss3.item()

        total_loss = loss1 + loss2 + loss3

        self.loss_details["total_loss"] = total_loss.item()
        return CausalLMOutputWithPastMTP(loss=total_loss, logits_mtp1=logits1, logits_mtp2=logits2, logits_mtp3=logits3)

    def _forward_inference(self, input_ids, attention_mask, position_ids, past_key_values,
                           inputs_embeds, use_cache, return_dict, **kwargs):
        """Inference method"""
        main_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache, output_hidden_states=True, return_dict=True,
        )
        main_hidden_last = main_outputs.last_hidden_state[:, -1:, :]
        main_logits = self.lm_head(main_hidden_last)
        
        _position_ids = position_ids
        if _position_ids is None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            _position_ids = sequence_lengths.unsqueeze(-1)
        else:
            _position_ids = _position_ids[:, -1:]
            
        cos, sin = self.model.rotary_emb(main_hidden_last, _position_ids)
        position_embeddings = (cos, sin)
        
        token_curr = input_ids[:, -1:]
        
        logits_mtp1, _ = self.MTP1(
            main_hidden_states=main_hidden_last, mtp_input_ids=token_curr,
            attention_mask=None, position_ids=_position_ids, position_embeddings=position_embeddings
        )
        
        draft_token_1 = torch.argmax(logits_mtp1, dim=-1)
        _position_ids_mtp2 = _position_ids + 1
        cos_mtp2, sin_mtp2 = self.model.rotary_emb(main_hidden_last, _position_ids_mtp2)
        position_embeddings_mtp2 = (cos_mtp2, sin_mtp2)
        
        logits_mtp2, _ = self.MTP2(
            main_hidden_states=main_hidden_last, mtp_input_ids=draft_token_1,
            attention_mask=None, position_ids=_position_ids_mtp2, position_embeddings=position_embeddings_mtp2
        )
        
        # Added MTP3 inference
        draft_token_2 = torch.argmax(logits_mtp2, dim=-1)
        _position_ids_mtp3 = _position_ids + 2
        cos_mtp3, sin_mtp3 = self.model.rotary_emb(main_hidden_last, _position_ids_mtp3)
        position_embeddings_mtp3 = (cos_mtp3, sin_mtp3)

        logits_mtp3, _ = self.MTP3(
            main_hidden_states=main_hidden_last, mtp_input_ids=draft_token_2,
            attention_mask=None, position_ids=_position_ids_mtp3, position_embeddings=position_embeddings_mtp3
        )
        
        return CausalLMOutputWithPastMTP(
            logits=main_logits, logits_mtp1=logits_mtp1, logits_mtp2=logits_mtp2, logits_mtp3=logits_mtp3,
            past_key_values=main_outputs.past_key_values, hidden_states=main_outputs.hidden_states,
        )