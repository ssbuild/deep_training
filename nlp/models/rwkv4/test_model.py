# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/5 17:06
import torch

from modeling_rwkv import RwkvConfig,RwkvModel,set_model_profile

set_model_profile(1024)

config = RwkvConfig(vocab_size=1024,ctx_len=128,n_layers=2)
model = RwkvModel(config)

print(model)
out = model.forward(torch.tensor([[187, 510, 100, 310, 247]]), state=None,return_dict=True)   # use 20B_tokenizer.json
print(out)
# print(out.detach().cpu().numpy())                   # get logits
# out, state = model.forward([187, 510],  state=None)
# out, state = model.forward([1563], state=state)           # RNN has state (use deepcopy if you want to clone it)
# out, state = model.forward([310, 247], state=state)
# print(out.detach().cpu().numpy())