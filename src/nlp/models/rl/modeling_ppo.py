# coding=utf8
# @Time    : 2023/5/13 17:44
# @Author  : tk
# @FileName: ppo_modeling

from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.utils import ModelOutput
from .utils import CausalLMOutputWithValue, Seq2SeqLMOutputWithValue, hf_get_decoder_blocks, hf_get_decoder_final_norm, \
    hf_get_lm_head, hf_get_hidden_size, hf_get_num_hidden_layers, CausalPrefixLMOutputWithValue, make_head
from ..chatglm import ChatGLMForConditionalGeneration, TransformerChatGlmLMHeadModel
from ..transformer import TransformerForCausalLM,TransformerForSeq2SeqLM

class AutoModelForCausalLMWithValueHead(TransformerForCausalLM):
    def __init__(self, *args,hidden_size=None,up_sampling_score=False, **kwargs):
        super(AutoModelForCausalLMWithValueHead, self).__init__(*args, **kwargs)
        # base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith(
        #     '_') else self.base_model_prefix
        # self.transformer_bone = getattr(self.model, base_model_prefix, None)
        # assert self.transformer_bone is not None
        self.score = make_head( hidden_size or hf_get_hidden_size(self.config), self.config.num_labels,up_sampling_score=up_sampling_score)




    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **inputs):
        return_dict = inputs.get('return_dict', False)
        if not return_dict:
            inputs.update({"return_dict": True})
        inputs["output_hidden_states"] = True
        outputs = self.model(*args, **inputs)
        value = self.score(outputs.hidden_states[-1]).squeeze(-1)
        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs
        return CausalLMOutputWithValue(**outputs, value=value)

class AutoModelForSeq2SeqLMWithValueHead(TransformerForSeq2SeqLM):
    """An `AutoModel` class wrapper for `transformers` sequence-to-sequence
    models that have a language modeling head and a value head
    """

    def __init__(self, *args,hidden_size=None,up_sampling_score=False, **kwargs):
        super(AutoModelForSeq2SeqLMWithValueHead, self).__init__(*args, **kwargs)
        self.score = make_head( hidden_size or hf_get_hidden_size(self.config), self.config.num_labels,up_sampling_score=up_sampling_score)

    def forward(self, *args, **inputs) -> Seq2SeqLMOutputWithValue:
        return_dict = inputs.get('return_dict', False)
        if not return_dict:
            inputs.update({"return_dict": True})
        inputs["output_hidden_states"] = True
        outputs = self.model(**inputs)
        last_hidden_state = outputs.decoder_hidden_states[-1]
        value = self.score(last_hidden_state).squeeze(-1)
        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs
        return Seq2SeqLMOutputWithValue(**outputs, value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.model.generate(*args, **kwargs)

class AutoModelForCausalPrefixLMWithValueHead(TransformerForCausalLM):
    def __init__(self, *args,hidden_size=None,up_sampling_score=False, **kwargs):
        super(AutoModelForCausalPrefixLMWithValueHead, self).__init__(*args, **kwargs)
        # base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith(
        #     '_') else self.base_model_prefix
        # self.transformer_bone = getattr(self.model, base_model_prefix, None)
        # assert self.transformer_bone is not None
        self.score = make_head(hidden_size or hf_get_hidden_size(self.config), self.config.num_labels,
                               up_sampling_score=up_sampling_score)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **inputs):
        return_dict = inputs.get('return_dict', False)
        if not return_dict:
            inputs.update({"return_dict": True})
        inputs["output_hidden_states"] = True
        outputs = self.model(*args, **inputs)
        value = self.score(outputs.hidden_states[-1]).squeeze(-1)
        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs
        return CausalPrefixLMOutputWithValue(**outputs, value=value)

class ChatglmModelForCausalPrefixLMWithValueHead(TransformerChatGlmLMHeadModel):
    def __init__(self, *args,hidden_size=None,up_sampling_score=False, **kwargs):
        super(ChatglmModelForCausalPrefixLMWithValueHead, self).__init__(*args, **kwargs)
        # base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith(
        #     '_') else self.base_model_prefix
        # self.transformer_bone = getattr(self.model, base_model_prefix, None)
        # assert self.transformer_bone is not None
        self.score = make_head(hidden_size or hf_get_hidden_size(self.config), self.config.num_labels,
                               up_sampling_score=up_sampling_score)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **inputs):
        return_dict = inputs.get('return_dict', False)
        if not return_dict:
            inputs.update({"return_dict": True})
        inputs["output_hidden_states"] = True
        outputs = self.model(*args, **inputs)
        value = self.score(outputs.hidden_states[-1].permute(1,0,2)).squeeze(-1)
        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs
        return CausalPrefixLMOutputWithValue(**outputs, value=value)

# class AutoModelForCausalLMWithHydraValueHead(AutoModelForCausalLMWithValueHead):
#     def __init__(self, *args, num_layers_unfrozen=-1, **kwargs):
#         super(AutoModelForCausalLMWithHydraValueHead, self).__init__(*args, **kwargs)
#         self.num_layers_unfrozen = num_layers_unfrozen
#
#         if self.num_layers_unfrozen > 0:
#             config = self.model.config
#             branch_class = hf_get_branch_class(config)
#             self.frozen_head = branch_class(
#                 self.base_model,
#                 num_layers_unfrozen=self.num_layers_unfrozen,
#             ).eval()
#
#     def forward_hydra(self, *args, **inputs) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
#         return_dict = inputs.get("return_dict", True)
#         inputs["return_dict"] = True
#         inputs["output_hidden_states"] = True
#         outputs = self.forward(**inputs)
#         # Select the hidden state before the first branching layer
#         input_hidden_state = outputs.hidden_states[-(self.num_layers_unfrozen + 1)]
#         output_shape = outputs.hidden_states[-1].size()
#         inputs.pop("input_ids", None)  # Ignore `input_ids` for branch head
#         inputs.pop("inputs_embeds", None)  # Ignore `inputs_embeds` for branch head
#         hydra_outputs = self.frozen_head(input_hidden_state, output_shape, **inputs)
#
#         if not return_dict:
#             return hydra_outputs.logits
#         return hydra_outputs
#
#
#
# class AutoModelForSeq2SeqLMWithHydraValueHead(AutoModelForSeq2SeqLMWithValueHead):
#     _supported_modules = ["v_head", "frozen_head"]
#     _supported_args = ["num_layers_unfrozen"]
#
#     def __init__(self, *args, num_layers_unfrozen=-1, **kwargs):
#         super(AutoModelForSeq2SeqLMWithHydraValueHead, self).__init__(*args, **kwargs)
#         self.num_layers_unfrozen = num_layers_unfrozen
#         self.num_layers_unfrozen = num_layers_unfrozen
#         if self.num_layers_unfrozen > 0:
#             branch_class = T5Branch  # TODO: Add support for other model branches
#             self.frozen_head = branch_class(
#                 self.base_model,
#                 num_layers_unfrozen=self.num_layers_unfrozen,
#             ).eval()
#
#     def forward_hydra(self, *args, **inputs) -> Seq2SeqLMOutputWithValue:
#         return_dict = inputs.get("return_dict", True)
#         inputs["output_hidden_states"] = True
#         inputs["return_dict"] = True
#
#         outputs = self.forward(**inputs)
#         # Select the hidden state before the first branching layer
#         input_hidden_state = outputs.decoder_hidden_states[-(self.num_layers_unfrozen + 1)]
#         decoder_attention_mask = inputs.get("decoder_attention_mask", None)
#         attention_mask = inputs.get("attention_mask", None)
#         hydra_outputs = self.frozen_head(
#             hidden_states=input_hidden_state,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=outputs.encoder_last_hidden_state,
#             encoder_attention_mask=attention_mask,
#             use_cache=False,
#             output_attentions=False,
#             output_hidden_states=True,
#             return_dict=return_dict,
#         )
#         if not return_dict:
#             return hydra_outputs.logits
#         return hydra_outputs

# class ModelBranch(transformers.PreTrainedModel):
#     """Implements the frozen upper trunk of the pretrained reference model used
#     when computing the PPO KL-divergence penalty.
#     """
#
#     def __init__(
#         self,
#         base_model: transformers.PreTrainedModel,
#         *,
#         num_layers_unfrozen: int,
#     ):
#         """
#         Args:
#             base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
#             num_layers_unfrozen (int): The number of trainable layers
#         """
#         super().__init__(base_model.config)
#
#         # The branch is defined by the last `num_layers_unfrozen` layers of the pretrained model
#         decoder_blocks = copy.deepcopy(hf_get_decoder_blocks(base_model))
#         self.decoder_blocks = nn.ModuleList(list(decoder_blocks)[-num_layers_unfrozen:])
#         self.final_norm = copy.deepcopy(hf_get_decoder_final_norm(base_model))
#         self.lm_head = copy.deepcopy(hf_get_lm_head(base_model))
#
#         self.hidden_size = hidden_size = kwargs.get('hidden_size',None) or hf_get_hidden_size(self.config)
#         self.model_parallel = False
#         self.device_map = None
#         self.last_device = None
#         self.gradient_checkpointing = False
#
#         # Freeze the entire branch
#         for parameter in self.parameters():
#             parameter.requires_grad_(False)
#
#
# class GPTModelBranch(ModelBranch):
#     def forward(  # noqa: max-complexity
#         self,
#         hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
#         output_shape: torch.Tensor,  # output_size given by main trunk
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = False,
#     ) -> Union[Tuple, CausalLMOutputWithValue]:
#         """Reference:
#         https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/gpt2/modeling_gpt2.py#L743  # noqa: E501
#         """
#         batch_size, seq_length = hidden_states.shape[:2]
#
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         device = hidden_states.device
#
#         if past_key_values is None:
#             past_length = 0
#             past_key_values = tuple([None] * len(self.decoder_blocks))
#         else:
#             past_length = past_key_values[0][0].size(-2)
#
#         if position_ids is None:
#             position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length)
#
#         if attention_mask is not None:
#             if batch_size <= 0:
#                 raise ValueError("batch_size has to be defined and > 0")
#             attention_mask = attention_mask.view(batch_size, -1)
#             attention_mask = attention_mask[:, None, None, :]
#             attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
#             attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
#
#         if self.config.add_cross_attention and encoder_hidden_states is not None:
#             (
#                 encoder_batch_size,
#                 encoder_sequence_length,
#                 _,
#             ) = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_attention_mask = None
#
#         head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))
#
#         presents = () if use_cache else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
#         all_hidden_states = () if output_hidden_states else None
#         for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
#             if self.model_parallel:
#                 torch.cuda.set_device(hidden_states.device)
#                 if layer_past is not None:
#                     layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
#                 if attention_mask is not None:
#                     attention_mask = attention_mask.to(hidden_states.device)
#                 if isinstance(head_mask, torch.Tensor):
#                     head_mask = head_mask.to(hidden_states.device)
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)
#
#             kwargs = dict(
#                 layer_past=layer_past,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 head_mask=head_mask[i],
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#             )
#
#             # Assumes we are never training the branch
#             block_params = inspect.getfullargspec(block.forward).args
#             if "encoder_hidden_states" not in block_params:
#                 kwargs.pop("encoder_hidden_states")
#                 kwargs.pop("encoder_attention_mask")
#             # Remove position_ids for GPT2Block
#             if "position_ids" not in block_params:
#                 kwargs.pop("position_ids")
#
#             outputs = block(hidden_states, **kwargs)
#
#             hidden_states = outputs[0]
#             if use_cache is True:
#                 presents = presents + (outputs[1],)
#
#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
#                 if self.config.add_cross_attention:
#                     all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
#
#             if self.model_parallel:
#                 for k, v in self.device_map.items():
#                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
#                         hidden_states = hidden_states.to("cuda:" + str(k + 1))
#
#         hidden_states = self.final_norm(hidden_states)
#
#         hidden_states = hidden_states.view(output_shape)
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#
#         if self.model_parallel:
#             torch.cuda.set_device(self.transformer.first_device)
#             hidden_states = hidden_states.to(self.lm_head.weight.device)
#
#         lm_logits = self.lm_head(hidden_states)
#
#         if not return_dict:
#             outputs = (lm_logits,) + (None,) + (None,)
#             return outputs
#
#         return CausalLMOutputWithValue(
#             logits=lm_logits,
#             past_key_values=presents,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )
#
#
# class OPTModelBranch(ModelBranch):
#     def forward(  # noqa: max-complexity
#         self,
#         hidden_states: torch.Tensor,
#         output_shape: torch.Tensor,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = False,
#     ) -> Union[Tuple, CausalLMOutputWithValue]:
#         """Reference:
#         https://github.com/huggingface/transformers/blob/bdb84e2bada3658f99c6a81c963ec562f8485151/src/transformers/models/opt/modeling_opt.py#L840  # noqa: E501
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
#
#         if attention_mask is None:
#             attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
#
#         input_shape = hidden_states.size()[:-1]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             # `modeling_opt._make_causal_mask` @ transformers==4.27.1 doesn't have the `device` argument
#             if "device" in inspect.getfullargspec(modeling_opt._make_causal_mask).args:
#                 kwargs = dict(device=hidden_states.device)
#             else:
#                 kwargs = {}
#
#             combined_attention_mask = modeling_opt._make_causal_mask(
#                 input_shape,
#                 hidden_states.dtype,
#                 past_key_values_length=past_key_values_length,
#                 **kwargs,
#             ).to(hidden_states.device)
#
#         if attention_mask is not None:
#             expanded_attn_mask = modeling_opt._expand_mask(
#                 attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
#             ).to(hidden_states.device)
#             combined_attention_mask = (
#                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )
#         attention_mask = combined_attention_mask
#
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
#
#         for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
#             if attn_mask is not None:
#                 if attn_mask.size()[0] != (len(self.decoder_blocks)):
#                     raise ValueError(
#                         f"The `{mask_name}` should be specified for {len(self.decoder_blocks)} layers, but it is for"
#                         f" {head_mask.size()[0]}."
#                     )
#
#         for idx, decoder_layer in enumerate(self.decoder_blocks):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#
#             past_key_value = past_key_values[idx] if past_key_values is not None else None
#
#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 past_key_value=past_key_value,
#                 attention_mask=attention_mask,
#                 layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#             )
#
#             hidden_states = layer_outputs[0]
#
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
#
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
#
#         if self.final_norm is not None:
#             hidden_states = self.final_norm(hidden_states)
#
#         # TODO: Add output projection support
#         # https://github.com/huggingface/transformers/blob/699e90437f984d69ad3c9b891dd2e9d0fc2cffe4/src/transformers/models/opt/modeling_opt.py#L499  # noqa: E501
#         # if self.project_out is not None:
#         #     hidden_states = self.project_out(hidden_states)
#
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
#
#         next_cache = next_decoder_cache if use_cache else None
#
#         lm_logits = self.lm_head(hidden_states).contiguous()
#
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     lm_logits,
#                     hidden_states,
#                     next_cache,
#                     all_hidden_states,
#                     all_self_attns,
#                 ]
#                 if v is not None
#             )
#
#         return CausalLMOutputWithValue(
#             logits=lm_logits,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
#
#
# class BloomModelBranch(ModelBranch):
#     def forward(  # noqa: max-complexity
#         self,
#         hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
#         output_shape: torch.Tensor,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = False,
#     ) -> Union[Tuple, CausalLMOutputWithValue]:
#         """Reference:
#         https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/bloom/modeling_bloom.py#L623  # noqa: E501
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         batch_size, seq_length = hidden_states.shape[:2]
#
#         if past_key_values is None:
#             past_key_values = tuple([None] * len(self.decoder_blocks))
#
#         head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))
#
#         presents = () if use_cache else None
#         all_self_attentions = () if output_attentions else None
#         all_hidden_states = () if output_hidden_states else None
#
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
#         if past_key_values[0] is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
#         if attention_mask is None:
#             attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
#         else:
#             attention_mask = attention_mask.to(hidden_states.device)
#
#         alibi = modeling_bloom.build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)
#
#         combined_attention_mask = None
#         device = attention_mask.device
#         input_shape = (batch_size, seq_length)
#         _, src_length = input_shape
#
#         if src_length > 1:
#             combined_attention_mask = modeling_bloom._make_causal_mask(
#                 input_shape,
#                 device=device,
#                 past_key_values_length=past_key_values_length,
#             )
#
#         expanded_attn_mask = modeling_bloom._expand_mask(attention_mask, tgt_length=src_length)
#         combined_attention_mask = (
#             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
#         )
#         causal_mask = combined_attention_mask
#
#         for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)
#
#             outputs = block(
#                 hidden_states,
#                 layer_past=layer_past,
#                 attention_mask=causal_mask,
#                 head_mask=head_mask[i],
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 alibi=alibi,
#             )
#
#             hidden_states = outputs[0]
#             if use_cache is True:
#                 presents = presents + (outputs[1],)
#
#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
#
#         hidden_states = self.final_norm(hidden_states)
#
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#
#         lm_logits = self.lm_head(hidden_states)
#
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     lm_logits,
#                     hidden_states,
#                     presents,
#                     all_hidden_states,
#                     all_self_attentions,
#                 ]
#                 if v is not None
#             )
#
#         return CausalLMOutputWithValue(
#             logits=lm_logits,
#             past_key_values=presents,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )
#
#
# class LlamaModelBranch(ModelBranch):
#     def _make_causal_mask(self, input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
#         """
#         Make causal mask used for bi-directional self-attention.
#         """
#         bsz, tgt_len = input_ids_shape
#         mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
#         mask_cond = torch.arange(mask.size(-1))
#         mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#         mask = mask.to(dtype)
#
#         if past_key_values_length > 0:
#             mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
#         return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
#
#     def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
#         """
#         Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
#         """
#         bsz, src_len = mask.size()
#         tgt_len = tgt_len if tgt_len is not None else src_len
#
#         expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
#
#         inverted_mask = 1.0 - expanded_mask
#
#         return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
#
#     def _prepare_decoder_attention_mask(self, attention_mask, input_shape, hidden_states, past_key_values_length):
#         # create causal mask
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = self._make_causal_mask(
#                 input_shape, hidden_states.dtype, past_key_values_length=past_key_values_length
#             ).to(hidden_states.device)
#
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             expanded_attn_mask = self._expand_mask(attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]).to(
#                 hidden_states.device
#             )
#             combined_attention_mask = (
#                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )
#         return combined_attention_mask
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         output_shape: torch.Tensor,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = False,
#     ) -> Union[Tuple, CausalLMOutputWithValue]:
#         """Reference:
#         https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L491
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         batch_size, seq_length = hidden_states.shape[:2]
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
#
#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
#
#         if position_ids is None:
#             device = hidden_states.device if hidden_states is not None else encoder_hidden_states.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()
#
#         # embed positions
#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
#         )
#
#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
#
#         for idx, decoder_layer in enumerate(self.decoder_blocks):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#
#             past_key_value = past_key_values[idx] if past_key_values is not None else None
#
#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
#
#             hidden_states = layer_outputs[0]
#
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
#
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
#
#         hidden_states = self.final_norm(hidden_states)
#         hidden_states = hidden_states.view(output_shape)
#         lm_logits = self.lm_head(hidden_states)
#
#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
#
#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             outputs = (lm_logits,) + (None,) + (None,)
#             return outputs
#
#         return CausalLMOutputWithValue(
#             logits=lm_logits,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
#
#
# # Seq2Seq architectures
# class T5Branch(ModelBranch):
#     """Decoder only T5 branch"""
#
#     def __init__(
#         self,
#         base_model: transformers.PreTrainedModel,
#         *,
#         num_layers_unfrozen: int,
#     ):
#         super().__init__(base_model, num_layers_unfrozen=num_layers_unfrozen)
#         self.dropout = hf_get_decoder(base_model).dropout
#         self.is_decoder = True
#
#     def forward(  # noqa: max-complexity
#         self,
#         hidden_states: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.LongTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, Seq2SeqLMOutputWithValue]:
#         """Reference:
#         https://github.com/huggingface/transformers/blob/bc21aaca789f1a366c05e8b5e111632944886393/src/transformers/models/t5/modeling_t5.py#L899  # noqa: E501
#         """
#         batch_size, seq_length = hidden_states.shape[:2]
#         input_shape = (batch_size, seq_length)
#
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if attention_mask is None:
#             attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
#         if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
#             encoder_seq_length = encoder_hidden_states.shape[1]
#             encoder_attention_mask = torch.ones(
#                 batch_size, encoder_seq_length, device=hidden_states.device, dtype=torch.long
#             )
#
#         extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
#
#         if self.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_extended_attention_mask = None
#
#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         position_bias = None
#         encoder_decoder_position_bias = None
#
#         for _, layer_module in enumerate(self.decoder_blocks):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)
#
#             layer_outputs = layer_module(
#                 hidden_states,
#                 attention_mask=extended_attention_mask,
#                 position_bias=position_bias,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_extended_attention_mask,
#                 encoder_decoder_position_bias=encoder_decoder_position_bias,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#             )
#
#             if use_cache is False:
#                 layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
#
#             hidden_states, present_key_value_state = layer_outputs[:2]
#
#             position_bias = layer_outputs[2]
#             if self.is_decoder and encoder_hidden_states is not None:
#                 encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
#
#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[3],)
#
#         hidden_states = self.final_norm(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#
#         # Add last layer
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#
#         sequence_output = hidden_states
#
#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586  # noqa: E501
#             sequence_output = sequence_output * (self.config.d_model**-0.5)
#
#         lm_logits = self.lm_head(sequence_output)
#
#         if not return_dict:
#             return (lm_logits,)
#
#         return Seq2SeqLMOutputWithValue(
#             logits=lm_logits,
#             decoder_hidden_states=all_hidden_states,
#             decoder_attentions=all_attentions,
#         )
#
#
# # Branch class utils
#
#
# def hf_get_branch_class(
#     config: transformers.PretrainedConfig,
# ) -> "ModelBranch":
#     """Returns the model branch class for the given config."""
#     gpt_branch_supported_archs = [
#         "GPTJForCausalLM",
#         "GPT2LMHeadModel",
#         "GPTNeoForCausalLM",
#         "GPTNeoXForCausalLM",
#     ]
#     opt_branch_supported_archs = ["OPTForCausalLM"]
#     bloom_branch_supported_archs = ["BloomModel", "BloomForCausalLM"]
#     llama_branch_supported_archs = ["LlamaModel", "LlamaForCausalLM"]
#     arch = config.architectures[0]
#     if arch in gpt_branch_supported_archs:
#         return GPTModelBranch
#     elif arch in opt_branch_supported_archs:
#         return OPTModelBranch
#     elif arch in bloom_branch_supported_archs:
#         return BloomModelBranch
#     elif arch in llama_branch_supported_archs:
#         return LlamaModelBranch
#     else:
#         all_supported_archs = sum(
#             [
#                 gpt_branch_supported_archs,
#                 opt_branch_supported_archs,
#                 bloom_branch_supported_archs,
#                 llama_branch_supported_archs,
#             ],
#             [],
#         )
#         raise ValueError(
#             f"Unsupported architecture: `{arch}`. The following architectures are "
#             f"available for model branching:\n{all_supported_archs}"
#         )
