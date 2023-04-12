
## transformer is all you need.
- 基于pytorch-lightning 和 transformers实现的上下游训练框架
- 安装 pip install -U deep_training

## 完整训练模型
  - [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
  - [tf-task-example](https://github.com/ssbuild/tf-task-example)
  - [poetry_training](https://github.com/ssbuild/poetry_training)


## 优化器
  adamw,adam,lamb,lion

## download 
  https://pypi.org/project/deep-training/#history

## 开发计划
  - keras 模块封装

## 更新
- <strong>2023年04月11</strong>
  - 0.1.2 重构lora v2, 增加adalora
  - 0.1.2@post0 fix lova v1,lova v2 load_in_8bit
- <strong>2023年04月07</strong>
  - deep_training 0.1.1 同步更新chatglm 词表配置信息 
- <strong>2023年04月02</strong>
  - release 0.1.0 and pytorch_lightning >= 2
- <strong>2023年03月15</strong>
  - 增加ChatGLM模型(稳定版>=0.0.18@post7) 完整训练参考 [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
  - 0.0.18@post8 解决deepspeed进程数据平衡 
  - 0.0.18@post9 增加流式输出接口stream_chat接口
  - 0.0.20 ChatGLM lora 加载权重继续训练 ， 修改数据数据编码 ，权重自适应
  - 0.0.21@post0 fix ChatGLM deepspeed stage 3 权重加载
- <strong>2023年03月09</strong>
  - 增加LLaMA 模型(并行版) 完整训练参考 [llama_finetuning](https://github.com/ssbuild/llama_finetuning)
- <strong>2023年03月08</strong>
  - 增加LLaMA 模型(非模型并行版) 完整训练参考 [poetry_training](https://github.com/ssbuild/poetry_training)
- <strong>2023年03月02</strong>
  - 增加loRA 训练 , lion,lamb优化器 , 完整训练参考 [chatyuan_finetuning](https://github.com/ssbuild/chatyuan_finetuning)
- <strong>2023年02月15</strong>
  - 增加诗歌PaLM预训练模型 
- <strong>2023年02月13</strong>
  - 增加中文语法纠错模型gector, seq2seq语法纠错模型 
- <strong>2023年02月09</strong>
  - 增加诗歌t5decoder预训练, 诗歌laMDA预训练模型 , t5encoder 预训练模型
- <strong>2023年02月07</strong>
  - 增加层次分解位置编码选项，让transformer可以处理超长文本
- <strong>2023年01月24</strong>
  - 增加诗歌gpt2预训练,诗歌t5预训练，诗歌unilm预训练
- <strong>2023年01月20</strong>
  - 增加对抗训练 FGM, FGSM_Local,FreeAT, PGD, FGSM,FreeAT_Local, 其中FreeAT推荐使用FreeAT_Local,FGSM 推荐使用 FGSM_Local
- <strong>2023年01月19</strong>
  - 增加promptbertcse监督和非监督模型
- <strong>2023年01月16</strong>
  - 增加diffcse 监督和非监督模型
- <strong>2023年01月13</strong>
  - 增加ESimcse 模型
- <strong>2023年01月11</strong>
  - 增加TSDAE句向量模型
- <strong>2023年01月09</strong>
  - 增加infonce监督和非监督,simcse监督和非监督,SPN4RE关系模型抽取
- <strong>2023年01月06</strong>
  - 增加onerel关系模型抽取，prgc关系模型抽取，pure实体模型提取
- <strong>2022年12月24</strong>
  - 增加unilm模型蒸馏和事件抽取模型
- <strong>2022年12月16</strong>
  - crf_cascad crf级联抽取实体
  - span ner 可重叠多标签，非重叠多标签两种实现方式抽取实体
  - mhs_ner 多头选择实体抽取模型
  - w2ner 实体抽取模型
  - tplinkerplus 实体抽取
  - tpliner 关系抽取模型
  - tplinkerplus 关系抽取模型
  - mhslinker 多头选择关系抽取模型

- <strong>2022年11月17</strong>: 
  - simcse-unilm 系列
  - simcse-bert-wwm 系列 
  - tnews circle loss
  - afqmc siamese net similar
- <strong>2022年11月15</strong>: 
  - unilm autotitle seq2seq autotitle
  - 普通分类,指针提取命名实体,crf提取命名实体
  - prefixtuning 分类 , prefixtuning 分类 , prefixtuning 指针提取命名实体 , prefixtuning crf 提取命名实体
- <strong>2022年11月12</strong>: 
  - gplinker (全局指针提取)
  - casrel (A Novel Cascade Binary Tagging Framework for Relational Triple Extraction 参考 https://github.com/weizhepei/CasRel)
  - spliner (指针提取关系 sigmoid pointer or simple pointer)
- <strong>2022年11月11</strong>: 
  - cluener_pointer 中文命名实体提取 和 cluener crf 中文命名实体提取
  - tnews 中文分类
- <strong>2022年11月06</strong>: 
  - mlm,gpt2,t5等模型预训练任务



## 支持任务
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

  
## 愿景
创建一个模型工厂, 轻量且高效的训练程序，让训练模型更容易,更轻松上手。

## 交流
QQ交流群：185144988
