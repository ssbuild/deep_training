# @Time    : 2023/5/2 13:49
# @Author  : tk
# @FileName: utils



def _prepare_prompt_learning_config(prompt_config, model_config):
    if prompt_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `prompt_config`")
        prompt_config.num_layers = num_layers

    if prompt_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `prompt_config`")
        prompt_config.token_dim = token_dim

    if prompt_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `prompt_config`")
        prompt_config.num_attention_heads = num_attention_heads

    if getattr(prompt_config, "encoder_hidden_size", None) is None:
        setattr(prompt_config, "encoder_hidden_size", token_dim)

    return prompt_config