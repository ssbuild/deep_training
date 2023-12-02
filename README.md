## transformer is all you need.
- deep training framework based on transformers

## install and download

- pip install -U deep_training
- 源码安装
```text
pip uninstall deep_training
pip install -U git+https://github.com/ssbuild/deep_training.git
```

- 源码重装
```text
pip install -U git+https://github.com/ssbuild/deep_training.git --no-deps --force-reinstall
```

  
## update
- <strong>2023-12-12</strong>
   - 0.2.10 update qwen model for 1.8b 7b 14b 72b


- <strong>2023-11-13</strong>
  - 0.2.9 release
  - 0.2.9.post0 support chatglm3-6b-32k

- <strong>2023-10-22</strong>
  - 0.2.7
    - support clip 完整训练 https://github.com/ssbuild/clip_finetuning 
    - support asr seq2seq 完整训练 https://github.com/ssbuild/asr_seq2seq_finetuning
    - support asr ctc 完整训练 https://github.com/ssbuild/asr_ctc_finetuning
    - support object detection 完整训练 https://github.com/ssbuild/detection_finetuning
    - support semantic segmentation 完整训练 https://github.com/ssbuild/semantic_segmentation
    - support chatglm3  完整训练 https://github.com/ssbuild/chatglm3_finetuning
  - 0.2.7.post1
    - support skywork 完整训练 https://github.com/ssbuild/skywork_finetuning
  - 0.2.7.post2
    - support bluelm 完整训练 https://github.com/ssbuild/bluelm_finetuning
  - 0.2.7.post3
    - support yi 完整训练 https://github.com/ssbuild/yi_finetuning
  - 0.2.7.post4
    - fix dataclass serialization in deepspeed
    
- <strong>2023-10-16</strong>
  - 0.2.6 support muti-model
    - visualglm 完整训练 https://github.com/ssbuild/visualglm_finetuning  
    - qwen-vl 完整训练  https://github.com/ssbuild/qwen_vl_finetuning 

- <strong>2023-10-07</strong>
  - 0.2.5 
    - support colossalai 训练 ，策略 ddp ,gemini,gemini_auto，zero2,zero2_cpu,3d
  - 0.2.5.post2 
    - support accelerator 训练 , fix some bug in accelerator and hf trainer
  - 0.2.5.post4 
    - fix trainer some bug

- <strong>2023-09-26</strong>
  - 0.2.4 
    - support transformers trainer and qwen-7b 新版 和 qwen-14b ， 旧版不再支持，旧版可以安装 deep_training <= 0.2.3 
  - 0.2.4.post3 
    - support ia3 finetuning

- <strong>2023-09-21</strong>
  - 0.2.3 
    - support dpo 完整训练 [dpo_finetuning](https://github.com/ssbuild/dpo_finetuning)

- <strong>2023-09-06</strong>
  - 0.2.2 
    - 调整baichuan模块命名 adjust baichuan v2 完整训练 [baichuan2_finetuning](https://github.com/ssbuild/baichuan2_finetuning)
  - 0.2.2.post0 
    - fix baichuan ptv2
  - 0.2.2.post1 
    - fix rwkv4 a bug
  - 0.2.2.post4 
    - fix llama and baichuan mask bug

- <strong>2023-09-02</strong>
  - 0.2.1 
    - fix llama model

- <strong>2023-08-23</strong>
  - 0.2.0 
    - release lora内部调整
  - 0.2.0.post1 
    - add xverse-13b chat  and fix muti lora
  
- <strong>2023-08-16</strong>
  - 0.1.21 
    - release 增加 5种 rope scale 方法 ， fix chatglm2-6b-32k 推理 rope_ratio
  - 0.1.21.post1 
    - fix moss rope
  
- <strong>2023-08-09</strong>
  - 0.1.17 
    - update qwen model
  - 0.1.17.post0 
    - update qwen config

- <strong>2023-08-08</strong>
  - 0.1.15.rc2 
    - support XVERSE-13B  完整训练 [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)

  
- <strong>2023-08-05</strong>
  - 0.1.13
    - support qwen(千问)  完整训练 [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
  - 0.1.13.post2 
    - fix quantization bug
  - 0.1.14 
    - release fix qwen stream

- <strong>2023-07-18</strong>
  - 0.1.12
    - support InternLm(书生)  完整训练 [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
    - support baichuan v2 完整训练 [baichuan2_finetuning](https://github.com/ssbuild/baichuan2_finetuning)
    - fix adalora some bugs
    - support rwkv world training

  
- <strong>2023-07-04</strong>
  - 0.1.11 rc1 
    - support baichuan model  完整训练 [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
    - support chatglm2 model  完整训练 [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
  - 0.1.11  
    - fix baichuan and chatglm2 some bugs 
    - support conv2d for lora 
    - support arrow parquet dataset
    
- <strong>2023-06-06</strong>

  
- <strong>2023-06-06</strong>
  - 0.1.10 
    - release add qlora and support more optimizer and scheduler
    - support lora prompt for deepspeed training
    - support rwkv4  完整训练 [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
  - 0.1.10.post0 
     - fix package setup for cpp and cu code for rwkv4
  - 0.1.10.post1 
     - fix infer for rwkv4


- <strong>2023-05-24</strong>
  - 0.1.8  
    - fix load weight in prompt_tuning,p_tuning,prefix_tuning,adaption_prompt

- <strong>2023-05-19</strong>
  - 0.1.7 
    - fix 0.1.5 rl bugs
  - 0.1.7.post1 
    - fix chatglm-6b-int4,chatglm-6b-int4 p-tuning-v2 training , fix ilql lightning import
    - fix load weight in prompt_tuning,p_tuning,prefix_tuning,adaption_prompt
  
- <strong>2023-05-10</strong>
  - 0.1.5 
    - fix lora v2 modules_to_save 自定义额外训练模块
    - support reward ppo  llm 完整训练 [rlhf_llm](https://github.com/ssbuild/rlhf_llm)
    - support reward ppo  chatglm 完整训练 [rlhf_chatglm](https://github.com/ssbuild/rlhf_chatglm)
    - support reward ppo  chatyuan 完整训练 [rlhf_chatyuan](https://github.com/ssbuild/rlhf_chatyuan)
  - 0.1.5.post2 release
    - fix prompt modules_to_save 自定义额外训练模块
    - support ilql 离线模式训练 ilql  完整训练 [rlhf_llm](https://github.com/ssbuild/rlhf_llm)
  - 0.1.5.post4 release
    - fix opt model hidden_size for ppo ilql 
    - fix ppotrainer ilqltrainer deepspeed save weight
    - import AdmaW from transformers or but torch firstly
  
- <strong>2023-05-02</strong>
  - 0.1.4 
    - support prompt_tuning,p_tuning,prefix_tuning,adaption_prompt

- <strong>2023-04-21</strong>
  - 0.1.3rc0 
    - support moss chat模型 完整训练参考 [moss_finetuning](https://github.com/ssbuild/moss_finetuning)
    - moss 量化int4 int8推理
  - 0.1.3.post0 
    - 新版本基于lightning, pytorch-lightning 更名 lightning,分离numpy-io模块




- <strong>2023-04-11</strong>
  - 0.1.2 
    - 重构lora v2, 增加adalora
  - 0.1.2.post0 
    - fix lova v1,lova v2 load_in_8bit
- <strong>2023-04-07</strong>
  - deep_training 0.1.1 
    - update chatglm config 
- <strong>2023-04-02</strong>
  - release 0.1.0 and lightning >= 2
- <strong>2023-03-15</strong>
  - 0.0.18
    - support ChatGLM模型(稳定版>=0.0.18.post7) 完整训练参考 [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
  - fix deepspeed进程数据平衡 
  - 0.0.18.post9 
    - 增加流式输出接口stream_chat接口
  - 0.0.20 ChatGLM lora 
    - 加载权重继续训练 ， 修改数据数据编码 ，权重自适应
  - 0.0.21.post0 
    - fix ChatGLM deepspeed stage 3 权重加载
- <strong>2023-03-09</strong>
  - 增加LLaMA 模型(并行版) 完整训练参考 [llama_finetuning](https://github.com/ssbuild/llama_finetuning)
- <strong>2023-03-08</strong>
  - 增加LLaMA 模型(非模型并行版) 完整训练参考 [poetry_training](https://github.com/ssbuild/poetry_training)
- <strong>2023-03-02</strong>
  - 增加loRA 训练 , lion,lamb优化器 , 完整训练参考 [chatyuan_finetuning](https://github.com/ssbuild/chatyuan_finetuning)
- <strong>2023-02-15</strong>
  - 增加诗歌PaLM预训练模型 
- <strong>2023-02-13</strong>
  - 增加中文语法纠错模型gector, seq2seq语法纠错模型 
- <strong>2023-02-09</strong>
  - 增加诗歌t5decoder预训练, 诗歌laMDA预训练模型 , t5encoder 预训练模型
- <strong>2023-02-07</strong>
  - 增加层次分解位置编码选项，让transformer可以处理超长文本
- <strong>2023-01-24</strong>
  - 增加诗歌gpt2预训练,诗歌t5预训练，诗歌unilm预训练
- <strong>2023-01-20</strong>
  - 增加对抗训练 FGM, FGSM_Local,FreeAT, PGD, FGSM,FreeAT_Local, 其中FreeAT推荐使用FreeAT_Local,FGSM 推荐使用 FGSM_Local
- <strong>2023-01-19</strong>
  - 增加promptbertcse监督和非监督模型
- <strong>2023-01-16</strong>
  - 增加diffcse 监督和非监督模型
- <strong>2023-01-13</strong>
  - 增加ESimcse 模型
- <strong>2023-01-11</strong>
  - 增加TSDAE句向量模型
- <strong>2023-01-09</strong>
  - 增加infonce监督和非监督,simcse监督和非监督,SPN4RE关系模型抽取
- <strong>2023-01-06</strong>
  - 增加onerel关系模型抽取，prgc关系模型抽取，pure实体模型提取
- <strong>2022-12-24</strong>
  - 增加unilm模型蒸馏和事件抽取模型
- <strong>2022-12-16</strong>
  - crf_cascad crf级联抽取实体
  - span ner 可重叠多标签，非重叠多标签两种实现方式抽取实体
  - mhs_ner 多头选择实体抽取模型
  - w2ner 实体抽取模型
  - tplinkerplus 实体抽取
  - tpliner 关系抽取模型
  - tplinkerplus 关系抽取模型
  - mhslinker 多头选择关系抽取模型

- <strong>2022-11-17</strong>: 
  - simcse-unilm 系列
  - simcse-bert-wwm 系列 
  - tnews circle loss
  - afqmc siamese net similar
- <strong>2022-11-15</strong>: 
  - unilm autotitle seq2seq autotitle
  - 普通分类,指针提取命名实体,crf提取命名实体
  - prefixtuning 分类 , prefixtuning 分类 , prefixtuning 指针提取命名实体 , prefixtuning crf 提取命名实体
- <strong>2022-11-12</strong>: 
  - gplinker (全局指针提取)
  - casrel (A Novel Cascade Binary Tagging Framework for Relational Triple Extraction 参考 https://github.com/weizhepei/CasRel)
  - spliner (指针提取关系 sigmoid pointer or simple pointer)
- <strong>2022-11-11</strong>: 
  - cluener_pointer 中文命名实体提取 和 cluener crf 中文命名实体提取
  - tnews 中文分类
- <strong>2022-11-06</strong>: 
  - mlm,gpt2,t5等模型预训练任务



## tasks
- <strong>预训练</strong>:
  - <strong> 数据参考 </strong> [THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
  - <strong>mlm预训练</strong>例子 bert roberta等一些列中文预训练 
  - <strong>lm预训练</strong>例子 gpt2等一些列中文预训练 
  - <strong>seq2seq 预训练</strong>例子 t5 small等一些列中文预训练 &nbsp;&nbsp;
  - <strong>unilm 预训练</strong>例子 unilm bert roberta 等一些列中文预训练 &nbsp;&nbsp
- <strong>中文分类</strong>:
  - 例子 <strong>tnews 中文分类</strong>
- <strong>命名实体提取</strong>: 
  - <strong>参考数据</strong>  cluner
  - <strong>cluener 全局指针提取</strong>
  - <strong>cluener crf提取</strong>
  - <strong>cluener crf prompt提取</strong>
  - <strong>cluener mhs ner多头选择提取</strong>
  - <strong>cluener span指针提取</strong>
  - <strong>cluener crf 级联提取</strong>
  - <strong>cluener tplinkerplus 提取</strong>
  - <strong>pure 提取</strong>
  - <strong>cluener w2ner 提取</strong>
- <strong>关系提取</strong>
  - <strong>参考数据</strong>  [duie和法研杯第一阶段数据](https://github.com/ssbuild/cail2022-info-extract)
  - <strong>gplinker 关系提取</strong>
  - <strong>casrel 关系提取</strong>
  - <strong>spliner 关系提取</strong>
  - <strong>mhslinker 关系提取</strong>
  - <strong>tplinker 关系提取</strong>
  - <strong>tplinkerplus 关系提取</strong>
  - <strong>onerel 关系抽取</strong>
  - <strong>prgc 关系提取</strong>
  - <strong>spn4re 关系提取</strong>
- <strong>事件提取</strong>
  - <strong>参考数据</strong> duee事件抽取 [DuEE v1.0数据集](https://aistudio.baidu.com/aistudio/competition/detail/46/0/datasets)
  - <strong>gplinker 事件提取</strong>
- <strong> prompt 系列</strong>: 
  - 例子 <strong>prefixprompt tnews中文分类</strong>
  - 例子 <strong>prefixtuning tnews 中文分类</strong>
  - 例子 <strong>prefixtuning cluener 命名实体全局指针提取</strong>
  - 例子 <strong>prefixtuning cluener 命名实体crf提取</strong>
  - 例子 <strong>prompt mlm 自行构建数据模板集，训练参考 pretrain/mlm_pretrain</strong>
  - 例子 <strong>prompt lm  自行构建数据模板集，训练参考 pretrain/seq2seq_pretrain ,  pretrain/lm_pretrain</strong>
- <strong> simcse 系列</strong>: 
  - <strong>simcse-unilm 系列</strong>  例子 unilm+simce  &nbsp;&nbsp; 
  参考数据&nbsp;&nbsp; [THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
  - <strong>simcse-bert-wwm 系列</strong> 例子 mlm+simcse &nbsp;&nbsp;
  参考数据&nbsp;&nbsp; [THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong> sentense embeding</strong>: 
  - <strong>circle loss </strong> 例子 tnews circle loss
  - <strong>siamese net </strong> 例子 afqmc siamese net similar



## optimizer
```text
   lamb,adma,adamw_hf,adam,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,
   adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,
   paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,
   lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp
```

## scheduler
```text
  linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,
  constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau
```

  
## works
Create a model factory, lightweight and efficient training program and make it easier, training model easier to get started.



## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [auto_finetuning](https://github.com/ssbuild/auto_finetuning)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)
## 
    纯粹而干净的代码

## 协议
本仓库的代码依照 Apache-2.0 协议开源


## discuss
QQ group：185144988


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/deep_training&type=Date)](https://star-history.com/#ssbuild/deep_training&Date)


