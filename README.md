
## transformer is all you need.
- 基于pytorch-lightning 和 transformers实现的上下游训练框架
- 当前正在重构中...，接口是不稳定的...

## 更新
- <strong>2022年11月17</strong>: 
  - simcse
- <strong>2022年11月15</strong>: 
  - unilm autotitle seq2seq autotitle
  - 普通分类,指针提取命名实体,crf提取命名实体
  - prefixtuning 分类 , prefixtuning 分类 , prefixtuning 指针提取命名实体 , prefixtuning crf 提取命名实体
- <strong>2022年11月12</strong>: 
  - gplinker (全局指针提取)
  - hphtlinker (半指针半标注提取 half pointer and half tages)
  - spliner (全指针提取 sigmoid pointer or simple pointer)
- <strong>2022年11月11</strong>: 
  - cluener_pointer 中文命名实体提取 和 cluener crf 中文命名实体提取
  - tnews 中文分类
- <strong>2022年11月06</strong>: 
  - mlm,gpt2,t5等模型预训练任务



## 支持任务
- <strong>mlm 预训练</strong>:
  - 例子 <strong>mlm bert roberta 等一些列中文预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>lm 预训练</strong>:
  - 例子 <strong>gpt2 等预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>seq2seq 预训练</strong>:
  - 例子 <strong>t5 small 等预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>unilm 预训练</strong>: 
  - 例子 <strong>unilm bert roberta 等一些列中文预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>中文分类</strong>:
  - 例子 <strong>tnews 中文分类</strong>
- <strong>命名实体提取</strong>: 
  - 例子 <strong>cluener 全局指针提取</strong>
  - 例子 <strong>cluener crf提取</strong>
- <strong>关系提取</strong>
  - 例子 <strong>gplinker 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
  - 例子 <strong>hphtlinker 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
  - 例子 <strong>spliner 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
- <strong> prompt 系列</strong>: 
  - 例子 <strong>prefixprompt tnews中文分类</strong>
  - 例子 <strong>prefixtuning tnews 中文分类</strong>
  - 例子 <strong>prefixtuning cluener 命名实体全局指针提取</strong>
  - 例子 <strong>prefixtuning cluener 命名实体crf提取</strong>
  - 例子 <strong>prompt mlm 自行构建数据集参考 pretrain/mlm_pretrain</strong>
  - 例子 <strong>prompt lm  自行构建数据集参考 pretrain/seq2seq_pretrain</strong>

  
## 愿景
创建一个模型工厂, 轻量且高效的训练程序，让训练模型更容易,更轻松上手。

## 交流
QQ交流群：185144988
