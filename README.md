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

###07-18

* ? ! 处理一下

logloss 为 0.53042

### 07-19

重新调了几个参数：

logloss 为 0.53228

### 07-20

* 加了两个 feature: 
  * word mover's distance 
  * edit distance.
* lgb 模型参数: df[:20000]
* valid's binary_logloss: 0.679708

log loss: 0.66980

所以这个 valid 还是很准的。

###07-27

所以当时 loss 最低的是什么设定来着？

* max_features 5500 -> 3000: 0.679708 -> 0.68103
* 'min_child_samples': 5,  'reg_alpha': 0.5,  'reg_lambda': 0.5:  0.670675

这些都不是关键，关键是什么呢？

* df[:20000] -> df[:1200]: 0.670675 -> 0.5902

所以说这个翻译的数据不能直接用的。

* df_train = df_es_train: 0.368931

我的天简直效果爆炸。我们来提交一发吧。

emmm 为什么提交之后只有 0.54815

###07-28

还是先把各 feature 的比重打印出来吧。

<https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py>

```
[('edit_distance', 13075), ('jaccard', 13667), ('word2vec_minkowski_2', 14023), ('ratio', 17136), ('token_sort_ratio', 18022), ('word2vec_minkowski_1', 19013), ('token_set_ratio', 21037), ('partial_ratio', 22008), ('word2vec_dot', 23677), ('wmd', 25582)]
```



LightGBM [Warning] No further splits with positive gain, best gain: -inf

'min_child_samples': 2,  # Minimum number of data need in a child(min_data_in_leaf)

#### preprocess

* .lower()：全部字符大写转小写
* string.punctuation：全部标点前后加空格（不仅仅是倒写的 ?!）
* tokenize 后就没有标点信息 <https://keras.io/preprocessing/text/>，所以加一个 feature



修了两个疑似 bug: wmd, dot



把 word2vec 的部分拿出来是为了不想算两遍

加了新的 feature: 5w1h



Model Report
bst1.best_iteration:  2100
binary_logloss: 0.27282968232044175

logloss: 0.48075

### 07-29

Seq2Seq 训好了。然而并没有太多心思把它们整在一起。

## TODO

* 西班牙语的神经网络模型
* 西班牙语到英语的  translation



* 西班牙语模型的 ensemble
* 还要保证这个模型切换数据集能直接跑才行
