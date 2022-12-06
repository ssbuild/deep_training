# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31

from transformers import AutoTokenizer,AutoConfig,CONFIG_MAPPING


__all__ = [
    'load_tokenizer',
    'load_configure'
]

def load_tokenizer(tokenizer_name,
                   model_name_or_path=None,
                   cache_dir="",
                   do_lower_case=True,
                   use_fast_tokenizer=True,
                   model_revision="main",
                   use_auth_token=None,**kwargs):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "do_lower_case": do_lower_case,
        "use_fast": use_fast_tokenizer,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer

def load_configure(config_name,
                   model_name_or_path=None,
                   cache_dir="",
                   model_revision="main",
                   use_auth_token=None,
                   model_type=None,
                   config_overrides=None,
                   bos_token_id=None,
                   pad_token_id=None,
                   eos_token_id=None,
                   sep_token_id=None,
                   task_specific_params=None,
                   **kwargs):
    config_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        "bos_token_id": bos_token_id,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "sep_token_id": sep_token_id,
        "task_specific_params": task_specific_params,
        **kwargs
    }
    if config_name:
        config = AutoConfig.from_pretrained(config_name, **config_kwargs)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    elif model_type:
        config = CONFIG_MAPPING[model_type]()
        if config_overrides is not None:
            config.update_from_string(config_overrides)
    else:
        raise ValueError(
            "You are instantiating a new config_gpt2 from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --config_name."
        )
    return config




# class DataCommonModule(LightningDataModule):
#     def __init__(self,*args,**kwargs):
#         super().__init__()
#
#
#     def prepare_data(self):
#         pass
#
#
#     # def setup(self, stage: str):
#     #     pass
#
#     # def train_dataloader(self):
#     #     try:
#     #         length = len(self.dataset["train"])
#     #     except:
#     #         length = None
#     #     collate_fn = self.dataReaderHelper.collate_fn
#     #     if length is None:
#     #         return DataLoader(torch_IterableDataset(self.dataset["train"].shuffle(1024).repeat(-1)), batch_size=self.train_batch_size,collate_fn=collate_fn)
#     #     return DataLoader(torch_Dataset(self.dataset["train"].shuffle(buffer_size=-1)), batch_size=self.train_batch_size,collate_fn=collate_fn)
#
#     # def val_dataloader(self):
#     #     if self.dataset["validation"] is None:
#     #         return super(GLUEDataModule, self).val_dataloader()
#     #
#     #     collate_fn = self.data_helper.collate_fn
#     #     return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size,collate_fn=collate_fn)
#     #
#     # def test_dataloader(self):
#     #     if self.dataset["test"] is None:
#     #         return super(GLUEDataModule, self).test_dataloader()
#     #
#     #     collate_fn = self.data_helper.collate_fn
#     #     return DataLoader(self.dataset["test"], batch_size=self.test_batch_size,collate_fn=collate_fn)

