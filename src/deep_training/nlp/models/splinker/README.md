## 关系抽取splinker

针对任务中多条、交叠SPO这一抽取目标，对标准的 'BIO' 标注进行了扩展。
对于每个 token，根据其在实体span中的位置（包括B、I、O三种），我们为其打上三类标签，并且根据其所参与构建的predicate种类，将 B 标签进一步区分。给定 schema 集合，对于 N 种不同 predicate，以及头实体/尾实体两种情况，我们设计对应的共 2*N 种 B 标签，再合并 I 和 O 标签，故每个 token 一共有 (2*N+2) 个标签，如下图所示。

<div align="center">
<img src="images/tagging_strategy.png" width="500" height="400" alt="标注策略" align=center />
</div>