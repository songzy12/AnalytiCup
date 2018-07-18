## Workflow

### 07-02

简单跑通了整个流程，

数据方面只使用了 es 相关的部分，

特征方面只是用了 word2vec 的dot 一个特征，

模型方面使用 LightGBM，

logloss 为 0.62227，排名为 150.

### 07-03

然后多加了几个特征之后效果不升反降？

* word2vec_minkowski_1 
* word2vec_minkowski_2
* ratio
* partial_ratio 
* token_sort_ratio 
* token_set_ratio 
* jaccard

logloss 为 0.64559

### 07-04

简单调了一下 LightGBM 的参数：

logloss 为 0.56064，排名为 122.

### 07-07

添加了 keras-quora-question-pairs 的神经网络模型。

logloss 为 0.71064.

###07-08

然后把 LightGBM 里加上那些英文的翻译数据好了。

logloss 为 0.52247，排名为 120.

###07-09

加上那些英文的翻译数据训练了一下 quora.

logloss 为 0.63559.

## 07-18

* ? ! 处理一下

## TODO

* log to file

* translation

* more features

  

* neural networks