# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/11 10:52
import re
from collections import OrderedDict

import torch

x = torch.load("pytorch_model.bin")

x = x["state_dict"]

weights_dict_new = OrderedDict()
for k,v in x.items():
    weights_dict_new[re.sub('_TransformerLightningModule__backbone.', 'transformer_base.', k)] = v

torch.save(weights_dict_new,"pytorch_model_new.bin")