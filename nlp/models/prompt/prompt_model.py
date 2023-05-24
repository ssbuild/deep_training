# -*- coding: utf-8 -*-
# @Time:  14:50
# @Author: tk
# @File：prompt_model

import inspect
import os
import warnings
from contextlib import contextmanager
import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from .configuration import PromptLearningConfig, PromptType, PromptBaseArguments, PROMPT_TYPE_TO_CONFIG_MAPPING, \
    WEIGHTS_NAME, TaskType
from .save_and_load import get_prompt_model_state_dict, set_prompt_model_state_dict
from .utils import _prepare_prompt_learning_config
from ...layers.prompt.prefix_tuning import PrefixEncoder
from ...layers.prompt.p_tuning import PromptEncoder
from ...layers.prompt.prompt_tuning import PromptEmbedding
from ...layers.prompt.utils import _set_trainable, _set_adapter, \
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING, shift_tokens_right


def get_prompt_model(model, prompt_config):
    """
    Returns a Prompt model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        prompt_config ([`PromptConfig`]): Configuration object containing the parameters of the Prompt model.
    """

    model_config = model.config.to_dict()
    prompt_config = _prepare_prompt_learning_config(prompt_config, model_config)
    return MODEL_TYPE_TO_PROMPT_MODEL_MAPPING[prompt_config.task_type](model, prompt_config)


class PromptModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Prompt methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Prompt.
        prompt_config ([`PromptLearningConfig`]): The configuration of the Prompt model.


    **Attributes**:
        - **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer model used for Prompt.
        - **prompt_config** ([`PromptLearningConfig`]) -- The configuration of the Prompt model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Prompt if
        using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Prompt if
        using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model, prompt_config: PromptLearningConfig, adapter_name="default"):
        super().__init__()

        self.base_model = model
        self.config = self.get_transformer_model().config
        self.modules_to_save = None
        self.prompt_config = {}
        self.active_adapter = adapter_name
        self.prompt_type = prompt_config.prompt_type
        self.base_model_torch_dtype = getattr(self.get_transformer_model(), "dtype", None)

        self.add_adapter(adapter_name, prompt_config)

    def get_transformer_model(self):
        return self.base_model if isinstance(self.base_model,PreTrainedModel) else self.base_model.model

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, prompt_config in self.prompt_config.items():
            # save only the trainable weights
            output_state_dict = get_prompt_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if prompt_config.base_model_name_or_path is None:
                prompt_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(prompt_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = prompt_config.inference_mode
            prompt_config.inference_mode = True
            prompt_config.save_pretrained(output_dir)
            prompt_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, pretrained_model_name_or_path, prompt_config: PromptLearningConfig = None,
                        adapter_name="default", is_trainable=False, **kwargs):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """


        # load the config
        if prompt_config is None:
            prompt_config = PROMPT_TYPE_TO_CONFIG_MAPPING[
                PromptBaseArguments.from_pretrained(pretrained_model_name_or_path,
                                                    subfolder=kwargs.get("subfolder", None)).prompt_type
            ].from_pretrained(pretrained_model_name_or_path, subfolder=kwargs.get("subfolder", None))


        if isinstance(prompt_config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            prompt_config.inference_mode = not is_trainable

        if prompt_config.task_type not in MODEL_TYPE_TO_PROMPT_MODEL_MAPPING.keys():
            model = cls(model, prompt_config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PROMPT_MODEL_MAPPING[prompt_config.task_type](model, prompt_config, adapter_name)
        model.load_adapter(pretrained_model_name_or_path, adapter_name, **kwargs)
        return model



    def load_weight(self, pretrained_model_name_or_path, adapter_name="default", is_trainable=False, **kwargs):
        self.load_adapter(pretrained_model_name_or_path, adapter_name,is_trainable=is_trainable, **kwargs)


    def _setup_prompt_encoder(self, adapter_name):
        config = self.prompt_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.prompt_type == PromptType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.prompt_type == PromptType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.prompt_type == PromptType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def get_prompt_embedding_to_save(self, adapter_name):
        """
        Returns the prompt embedding to save when saving the model.
        """
        prompt_tokens = self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(self.device)
        if self.prompt_config[adapter_name].prompt_type == PromptType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.prompt_config[adapter_name].num_virtual_tokens]
        prompt_embeddings = self.prompt_encoder[adapter_name](prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size):
        """
        Returns the virtual prompts to use for Prompt.
        """
        prompt_config = self.active_prompt_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = self.prompt_tokens[self.active_adapter].unsqueeze(0).expand(batch_size, -1).to(self.device)
        if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : prompt_config.num_virtual_tokens]
            if prompt_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                prompt_config.num_virtual_tokens,
                prompt_config.num_layers * 2,
                prompt_config.num_attention_heads,
                prompt_config.token_dim // prompt_config.num_attention_heads,
            )
            if prompt_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                prompt_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if prompt_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens)
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            try:
                return getattr(self.base_model, name)
            except AttributeError:
                return getattr(self.base_model.model, name)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)



    #becouse implementation new forward , so need define compute_loss in this class
    def compute_loss(self,*args,**kwargs):
        return self.forward(*args,**kwargs)


    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        try:
            if isinstance(self.prompt_config, PromptLearningConfig):
                old_forward = self.forward
                self.forward = self.base_model.forward
            else:
                self.base_model.disable_adapter_layers()
            yield
        finally:
            if isinstance(self.prompt_config, PromptLearningConfig):
                self.forward = old_forward
            else:
                self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model.model
        # return self.base_model if isinstance(self.active_prompt_config, PromptLearningConfig) else self.base_model.model

    def add_adapter(self, adapter_name, prompt_config):
        if prompt_config.prompt_type != self.prompt_type:
            raise ValueError(
                f"Cannot combine adapters with different prompt types. "
                f"Found {self.prompt_type} and {prompt_config.prompt_type}."
            )
        self.prompt_config[adapter_name] = prompt_config
        self._setup_prompt_encoder(adapter_name)
        self.set_additional_trainable_modules(prompt_config, adapter_name)

    def set_additional_trainable_modules(self, prompt_config, adapter_name):
        if getattr(prompt_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(prompt_config.modules_to_save)
            else:
                self.modules_to_save.update(prompt_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        if adapter_name not in self.prompt_config:
            # load the config
            prompt_config = PROMPT_TYPE_TO_CONFIG_MAPPING[
                PromptBaseArguments.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).prompt_type
            ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            if isinstance(prompt_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                prompt_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, prompt_config)


        # load weights if any
        path = os.path.join(model_id, kwargs["subfolder"]) if kwargs.get("subfolder", None) is not None else model_id

        if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
        else:
            raise ValueError(
                f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
            )
        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        set_prompt_model_state_dict(self, adapters_weights, adapter_name=adapter_name)




    def set_adapter(self, adapter_name):
        """
        Sets the active adapter.
        """
        if adapter_name not in self.prompt_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not isinstance(self.prompt_config[adapter_name], PromptLearningConfig):
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def active_prompt_config(self):
        return self.prompt_config[self.active_adapter]


class PromptModelForSequenceClassification(PromptModel):
    """
    Prompt model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        prompt_config ([`PromptLearningConfig`]): Prompt config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        # >>> from transformers import AutoModelForSequenceClassification
        # >>> from peft import PromptModelForSequenceClassification, get_peft_config
        #
        # >>> config = {
        # ...     "prompt_type": "PREFIX_TUNING",
        # ...     "task_type": "SEQ_CLS",
        # ...     "inference_mode": False,
        # ...     "num_virtual_tokens": 20,
        # ...     "token_dim": 768,
        # ...     "num_transformer_submodules": 1,
        # ...     "num_attention_heads": 12,
        # ...     "num_layers": 12,
        # ...     "encoder_hidden_size": 768,
        # ...     "prefix_projection": False,
        # ...     "postprocess_past_key_value_function": None,
        # ... }
        #
        # >>> prompt_config = get_peft_config(config)
        # >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        # >>> peft_model = PromptModelForSequenceClassification(model, prompt_config)
        # >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, prompt_config: PromptLearningConfig, adapter_name="default"):
        super().__init__(model, prompt_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        prompt_config = self.active_prompt_config
        if not isinstance(prompt_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, prompt_config.num_virtual_tokens).to(self.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PromptModelForCausalLM(PromptModel):
    """
    Prompt model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        prompt_config ([`PromptLearningConfig`]): Prompt config.


    Example:

        ```py
        # >>> from transformers import AutoModelForCausalLM
        # >>> from peft import PromptModelForCausalLM, get_peft_config
        #
        # >>> config = {
        # ...     "prompt_type": "PREFIX_TUNING",
        # ...     "task_type": "CAUSAL_LM",
        # ...     "inference_mode": False,
        # ...     "num_virtual_tokens": 20,
        # ...     "token_dim": 1280,
        # ...     "num_transformer_submodules": 1,
        # ...     "num_attention_heads": 20,
        # ...     "num_layers": 36,
        # ...     "encoder_hidden_size": 1280,
        # ...     "prefix_projection": False,
        # ...     "postprocess_past_key_value_function": None,
        # ... }
        #
        # >>> prompt_config = get_peft_config(config)
        # >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        # >>> peft_model = PromptModelForCausalLM(model, prompt_config)
        # >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model, prompt_config: PromptLearningConfig, adapter_name="default"):
        super().__init__(model, prompt_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.get_transformer_model().prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        prompt_config = self.active_prompt_config
        if not isinstance(prompt_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, prompt_config.num_virtual_tokens), -100).to(self.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        prompt_config = self.active_prompt_config
        self.get_transformer_model().prepare_inputs_for_generation = self.prepare_inputs_for_generation
        generate_fn = getattr(self.base_model,"generate",self.base_model.model.generate)
        try:
            if not isinstance(prompt_config, PromptLearningConfig):
                outputs = generate_fn(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Prompt model generation")
                # For gpt2 models, we construct postion_ids on the fly by using attention mask, and position ids need to match input_shape.
                # for prefix tuning, input shape is determined using `input_ids`. Thus we should not expand 'attention_mask' here
                # for prompt tuning input_ids is not passed but a concatenated input_embeds is passed. Thus attention_mask needs to be of same size of num_virtual_tokens + input_ids
                if kwargs.get("attention_mask", None) is not None and prompt_config.prompt_type in [
                    PromptType.PROMPT_TUNING,
                    PromptType.P_TUNING,
                ]:
                    # concat prompt attention mask
                    prefix_attention_mask = torch.ones(
                        kwargs["input_ids"].shape[0], prompt_config.num_virtual_tokens
                    ).to(kwargs["input_ids"].device)
                    kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                outputs = generate_fn(**kwargs)
        except Exception:
            self.get_transformer_model().prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.get_transformer_model().prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        prompt_config = self.active_prompt_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(prompt_config, PromptLearningConfig):
            if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], prompt_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs["past_key_values"] is None and prompt_config.prompt_type == PromptType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])

                if self.base_model_torch_dtype is not None:
                    # handle the case for Bloom where it outputs tuple of tuples
                    if isinstance(past_key_values[0], tuple):
                        past_key_values = tuple(
                            tuple(
                                past_key_value.to(self.base_model_torch_dtype)
                                for past_key_value in past_key_value_tuple
                            )
                            for past_key_value_tuple in past_key_values
                        )
                    else:
                        past_key_values = tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                        )

                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs


class PromptModelForSeq2SeqLM(PromptModel):
    """
    Prompt model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        prompt_config ([`PromptLearningConfig`]): Prompt config.


    Example:

        ```py
        # >>> from transformers import AutoModelForSeq2SeqLM
        # >>> from peft import PromptModelForSeq2SeqLM, get_peft_config
        #
        # >>> config = {
        # ...     "prompt_type": "LORA",
        # ...     "task_type": "SEQ_2_SEQ_LM",
        # ...     "inference_mode": False,
        # ...     "r": 8,
        # ...     "target_modules": ["q", "v"],
        # ...     "lora_alpha": 32,
        # ...     "lora_dropout": 0.1,
        # ...     "merge_weights": False,
        # ...     "fan_in_fan_out": False,
        # ...     "enable_lora": None,
        # ...     "bias": "none",
        # ... }
        #
        # >>> prompt_config = get_peft_config(config)
        # >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        # >>> peft_model = PromptModelForSeq2SeqLM(model, prompt_config)
        # >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(self, model, prompt_config: PromptLearningConfig, adapter_name="default"):
        super().__init__(model, prompt_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.get_transformer_model().prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.get_transformer_model()._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        prompt_config = self.active_prompt_config
        if not isinstance(prompt_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(self.device)
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(self.device)
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if prompt_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif prompt_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, prompt_config.num_virtual_tokens), -100).to(self.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : prompt_config.num_virtual_tokens], inputs_embeds), dim=1)
            if prompt_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif prompt_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, prompt_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        prompt_config = self.active_prompt_config
        self.get_transformer_model().prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.get_transformer_model()._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        generate_fn = getattr(self.base_model, "generate", self.base_model.model.generate)
        try:
            if not isinstance(prompt_config, PromptLearningConfig):
                outputs = generate_fn(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Prompt model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
                    outputs = generate_fn(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.get_transformer_model().prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.get_transformer_model()._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.get_transformer_model().prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.get_transformer_model()._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        prompt_config = self.active_prompt_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None and prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            if self.base_model_torch_dtype is not None:
                # handle the case for Bloom where it outputs tuple of tuples
                if isinstance(past_key_values[0], tuple):
                    past_key_values = tuple(
                        tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_value_tuple
                        )
                        for past_key_value_tuple in past_key_values
                    )
                else:
                    past_key_values = tuple(
                        past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                    )
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs


class PromptModelForTokenClassification(PromptModel):
    """
    Prompt model for token classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        prompt_config ([`PromptLearningConfig`]): Prompt config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        # >>> from transformers import AutoModelForSequenceClassification
        # >>> from peft import PromptModelForTokenClassification, get_peft_config
        #
        # >>> config = {
        # ...     "prompt_type": "PREFIX_TUNING",
        # ...     "task_type": "TOKEN_CLS",
        # ...     "inference_mode": False,
        # ...     "num_virtual_tokens": 20,
        # ...     "token_dim": 768,
        # ...     "num_transformer_submodules": 1,
        # ...     "num_attention_heads": 12,
        # ...     "num_layers": 12,
        # ...     "encoder_hidden_size": 768,
        # ...     "prefix_projection": False,
        # ...     "postprocess_past_key_value_function": None,
        # ... }
        #
        # >>> prompt_config = get_peft_config(config)
        # >>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
        # >>> peft_model = PromptModelForTokenClassification(model, prompt_config)
        # >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, prompt_config: PromptLearningConfig = None, adapter_name="default"):
        super().__init__(model, prompt_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        prompt_config = self.active_prompt_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(prompt_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if prompt_config.prompt_type == PromptType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, prompt_config.num_virtual_tokens).to(self.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.get_transformer_model().get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


MODEL_TYPE_TO_PROMPT_MODEL_MAPPING = {
    "seq_cls": PromptModelForSequenceClassification,
    "seq_2_seq_lm": PromptModelForSeq2SeqLM,
    "causal_lm": PromptModelForCausalLM,
    "token_cls": PromptModelForTokenClassification,
}