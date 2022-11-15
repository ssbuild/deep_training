
## transformer is all you need.
- 基于pytorch-lightning 和 transformers实现的上下游训练框架
- 当前正在重构中...，接口是不稳定的...

## 支持任务
- <strong>mlm 预训练</strong>:
  - 例子 <strong>mlm 预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[清华NLP组提供的THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>lm 预训练</strong>:
  - 例子 <strong>gpt2 预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[清华NLP组提供的THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>unilm 预训练</strong>: 
  - 例子 <strong>unilm 预训练</strong> &nbsp;&nbsp;参考数据&nbsp;&nbsp;[清华NLP组提供的THUCNews新闻文本分类数据集的子集](https://pan.baidu.com/s/1eS-QZpWbWfKtdQE4uvzBrA?pwd=1234)
- <strong>中文分类</strong>:
  - 例子 <strong>tnews 中文分类</strong>
- <strong>命名实体提取</strong>: 
  - 例子 <strong>cluener 全局指针提取</strong>
  - 例子 <strong>cluener crf提取</strong>
- <strong>关系提取</strong>
  - 例子 <strong>gplinker 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
  - 例子 <strong>hphtlinker 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
  - 例子 <strong>spliner 关系提取</strong>: &nbsp;&nbsp;参考数据&nbsp;&nbsp;[法研杯2022信息抽取数据](https://github.com/ssbuild/cail2022-info-extract)
- <strong>p-tuning v2 系列</strong>: 
  - 例子 <strong>p-tuning v2 tnews 中文分类</strong>
  - 例子 <strong>p-tuning v2 cluener 命名实体全局指针提取</strong>
  - 例子 <strong>p-tuning v2 cluener 命名实体crf提取</strong>
- <strong>prompt 系列</strong>: 
  - 例子 正在调试ing</strong>

## 更新
- <strong>2022年11月15</strong>: 增加unilm autotitle,p-tuning v2 tnews , p-tuning v2 pointer , p-tuning v2 crf
- <strong>2022年11月12</strong>: 增加关系任务 gplinker (全局指针提取), hphtlinker(半指针半标注提取 ,half pointer and half tages ),spliner (全指针提取 sigmoid pointer or simple pointer)
- <strong>2022年11月11</strong>: 增加cluener_pointer 中文命名实体提取 , cluener crf ,tnews 中文分类
- <strong>2022年11月06</strong>: 增加gpt2,mlm预训练

## 愿景
创建一个模型工厂 , 让训练模型更容易，训练代码不超过100行，轻松上手。

## 交流
QQ交流群：185144988
