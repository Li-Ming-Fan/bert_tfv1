# bert_tfv1


BERT, tensorflow, v1. Classification. Sequential labelling.


## Pretrained model ckpt

Pretrained model of Chinese language:

```
Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
```

模型的[下载链接](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)可以在github上google的开源代码里找到。

对下载的压缩文件进行解压，得到五个文件，
- bert_model.ckpt开头的文件是负责模型变量载入的
- vocab.txt是训练时中文文本采用的字典
- bert_config.json是BERT在训练时，可选调整的一些参数



## Acknowledgement

- [google-research/bert](https://github.com/google-research/bert)

- [renxingkai/BERT_Chinese_Classification](https://github.com/renxingkai/BERT_Chinese_Classification)

- [FuYanzhe2/Name-Entity-Recognition](https://github.com/FuYanzhe2/Name-Entity-Recognition)

- [guillaumegenthial/tf_metrics](https://github.com/guillaumegenthial/tf_metrics)

