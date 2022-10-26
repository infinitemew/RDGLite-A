# RDGLite-A

云南大学学报2022论文[融合属性嵌入与关系注意力的跨语言实体对齐](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CAPJ&dbname=CAPJLAST&filename=YNDZ20220927001&uniplatform=NZKPT&v=ILIcPzEcsTyBEXay-HElQvOlrDc4hZ9MiULTEKRCfwh9-zfxYASOXKTDOBGmJH0h)的源码。

初始数据集来自[RDGCN](https://github.com/StephanieWyt/RDGCN)和[GCN-Align](https://github.com/1049451037/GCN-Align) .

## 环境

* Python = 3.6
* Tensorflow = 1.14
* Scipy = 1.5.0
* Numpy = 1.19.2

> 由于GPU显存的不足，我们运行在 Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz CPU上。

## 数据集

代码使用DBP15K数据集，该数据集包含以下三个跨语言的子数据集：
- fr-en
- ja-en
- zh-en

> 代码默认的数据集路径为 data/DBP15K/

您可以在[这里](https://pan.baidu.com/s/1yeOXx5LlUca8J4gmWGwlEQ?pwd=7ral)直接下载数据集，此外您也可以按照下述内容自行构建。

以下的文件您可以在[这里](https://github.com/1049451037/GCN-Align/tree/master/data)找到。

* ent_ids_1: KG (ZH)中的实体ID和实体名；
* ent_ids_2: KG (EN)中的实体ID和实体名；
* ref_ent_ids: 预先对齐的实体对；
* triples_1: KG (ZH)中的关系三元组；
* triples_2: KG (EN)中的关系三元组;

以下的文件您可以在[这里](https://github.com/StephanieWyt/RDGCN)找到。

* zh_vectorList.json: 由Glove初始化的实体嵌入；

> 考虑到国内的网络访问谷歌网盘的困难，我们在这里也提供了[百度网盘](https://pan.baidu.com/s/1RxUy6m2rBLuTcpfkVslzqA?pwd=p46r)下载。

以下的文件您可以运行getAttrEmbedding.py来生成，考虑到该代码的运行需要消耗大量的内存，我们也在[这里](https://pan.baidu.com/s/1I-7KuzMwk6bjQEJ5qIrNPQ?pwd=5bqe)提供单独下载。

```
python getAttrEmbedding.py --lang zh_en
```

* zh_ae_adj_sparse.json: 用在zh-en数据集中的实体-属性邻接矩阵；
* zh_attr_vector.json: 由PyTorch的nn.Embedding初始化的属性嵌入；

> 上述文件以DBP15K(zh-en)数据集为例。

## 运行

以DBP15K(zh-en)数据集为例。

```
python main.py --lang zh_en
```

> 您可以在Config.py中修改超参数。
