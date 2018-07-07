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

###07-04

简单调了一下 LightGBM 的参数：

logloss 为 0.56064，排名为 122.

### 07-07

添加了 keras-quora-question-pairs 的神经网络模型。

## References

<https://zhuanlan.zhihu.com/p/35093355>

<https://github.com/HouJP/kaggle-quora-question-pairs>

##Jupyter Notebook 

```
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```



<https://blog.csdn.net/lawme/article/details/51034543>

- **Shift-Enter** : 运行本单元，选中下个单元
- **Ctrl-Enter** : 运行本单元
- **Alt-Enter** : 运行本单元，在其下插入新单元



- **Y** : 单元转入代码状态
- **M** :单元转入markdown状态
- **R** : 单元转入raw状态



- **A** : 在上方插入新单元
- **B** : 在下方插入新单元



- **X** : 剪切选中的单元
- **C** : 复制选中的单元
- **Shift-V** : 粘贴到上方单元
- **V** : 粘贴到下方单元



- **Z** : 恢复删除的最后一个单元
- **D,D** : 删除选中的单元
- **Shift-M** : 合并选中的单元