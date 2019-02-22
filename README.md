# recommendation-based-on-SparkMLLib

## Spark MLLib 简介

Spark MLlib(Machine Learnig lib) 是Spark对常用的机器学习算法的实现库，同时包括相关的测试和数据生成器。Spark的设计初衷就是为了支持一些迭代的Job, 这正好符合很多机器学习算法的特点。

Spark MLlib目前支持4种常见的机器学习问题: 分类、回归、聚类和协同过滤。Spark MLlib基于RDD，天生就可以与Spark SQL、GraphX、Spark Streaming无缝集成，以RDD为基石，4个子框架可联手构建大数据计算中心！

下图是MLlib算法库的核心内容：

![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/mllib.png)

## 协同过滤推荐算法

协同过滤算法（Collaborative Filtering：CF）是很常用的一种算法，在很多电商网站上都有用到。CF算法包括基于用户的CF（User-based CF）和基于物品的CF（Item-based CF）。

### （一）、基于用户（User CF）的协同过滤算法

#### 原理：

* 构建用户对物品的打分矩阵

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/rating.png)

* 根据余弦相似度公式计算**用户**相似度矩阵

	<a href="https://github.com/MrQuJL/product-recommendation-system" target="_blank">余弦相似度计算公式</a>：

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/similarity.png)

	<a href="https://github.com/MrQuJL/product-recommendation-system" target="_blank">用户相似度矩阵：</a>

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/usersimilarity.png)

* 找出与指定用户相似度最高的前N个用户

* 找出这N个用户评价过的商品，去掉被推荐的用户评价过的商品，则是推荐结果

#### 代码实现：

```js

```

### （二）、基于物品（Item CF）的协同过滤算法

#### 原理：

* 构建用户对物品的打分矩阵

* 根据余弦相似度公式计算**物品**相似度矩阵

* 对于当前用户评价高的物品，找出与之相似度最高的N个物品

* 将这N个物品推荐给用户

#### 代码实现：

```js

```

### （三）、基于 ALS 的协同过滤算法

#### 简介：

ALS 是交替最小二乘 （alternating least squares）的简称。

ALS算法是2008年以来，用的比较多的协同过滤算法。它已经集成到Spark的Mllib库中，使用起来比较方便。

从协同过滤的分类来说，ALS算法属于User-Item CF，也叫做混合CF。它同时考虑了User和Item两个方面。

用户和商品的关系，可以抽象为如下的三元组：<User,Item,Rating>。其中，Rating是用户对商品的评分，表征用户对该商品的喜好程度。

假设我们有一批用户数据，其中包含m个User和n个Item，则我们定义Rating矩阵，其中的元素表示第u个User对第i个Item的评分。

在实际使用中，由于n和m的数量都十分巨大，因此R矩阵的规模很容易就会突破1亿项。这时候，传统的矩阵分解方法对于这么大的数据量已经是很难处理了。

另一方面，一个用户也不可能给所有商品评分，因此，R矩阵注定是个稀疏矩阵。矩阵中所缺失的评分，又叫做missing item。

#### ALS算法举例说明：

1. 下面的矩阵R表示：观众对电影的喜好，即：打分的情况。注意：实际情况下，这个矩阵可能非非常庞大，并且是一个稀疏矩阵。

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/r.png)


2. 这时，我们可以把这个大的稀疏矩阵R，拆分成两个小一点的矩阵：U和V。通过U和V来近似表示R，如下图：

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/ruv.png)


* U 矩阵代表：用户的特征，包括三个维度：性格，文化程度，兴趣爱好

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/u.png)

* V 矩阵代表：电影的特征，也包括三个维度：性格，文化程度，兴趣爱好

	![image](https://github.com/MrQuJL/recommendation-based-on-SparkMLLib/raw/master/imgs/v.png)

3. 这样，U和V的乘积，近似表示R。

4. 但是，这样的表示是存在误差的，因为对于一个U矩阵来说，我们并不可能说（性格，文化程度，兴趣爱好）这三个属性就代表着一个人对一部电影评价全部的属性，比如还有地域等因素。这个误差，我们用RMSE（均方根误差）表示。

#### 代码实现：

```js

```


