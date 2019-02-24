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

	```scala
	import org.apache.log4j.Logger
	import org.apache.log4j.Level
	import org.apache.spark.SparkConf
	import org.apache.spark.SparkContext
	import org.apache.spark.rdd.RDD
	import org.apache.spark.mllib.linalg.distributed.MatrixEntry
	import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
	import org.apache.spark.mllib.linalg.distributed.RowMatrix

	object UserBasedCF {
	  def main(args: Array[String]): Unit = {
		Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
		Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
		
		// 创建一个SparkContext
		val conf = new SparkConf().setAppName("test").setMaster("local")
		val sc = new SparkContext(conf)
		
		// 读入数据
		val data = sc.textFile("hdfs://qujianlei:9000/data/ratingdata.txt")
		
		// 解析出评分矩阵的每一行
		val parseData:RDD[MatrixEntry] = data.map(_.split(",")
			match {case Array(user,item,rate) => 
			MatrixEntry(user.toLong,item.toLong,rate.toDouble)})
		// 构建关联矩阵
		val ratings = new CoordinateMatrix(parseData)
		
		// 转置矩阵以计算列(用户)的相似性
		val matrix:RowMatrix = ratings.transpose().toRowMatrix()
		
		// 计算得到用户的相似度矩阵
		val similarities = matrix.columnSimilarities()
		println("输出用户相似度矩阵")
		similarities.entries.collect().map(x=>{
		  println(x.i + "--->" + x.j + "--->" + x.value)
		})
		println("-----------------------------------------")
		
		// 得到某个用户对所有物品的评分
		val ratingOfUser1 = ratings.entries.filter(_.i == 1).
		  map(x=>(x.j,x.value)).
		  sortBy(_._1).
		  map(_._1).
		  collect().
		  toList.
		  toArray
		println("用户1对所有物品的评分")  
		for (s <- ratingOfUser1) println(s)
		println("-----------------------------------------")
		
		// 得到用户1相对于其他用户的相似性
		val similarityOfUser1 = similarities.entries.filter(_.i == 1).
		  sortBy(_.value, false).
		  map(_.value).
		  collect
		println("用户1相对于其他用户的相似性")
		for (s <- similarityOfUser1) println(s)
		
		// 需求：为用户1推荐2个商品
		// 思路：找到与用户1相似性最高的两个用户，将这两个用户评过分的物品，用户1没有评过分的物品推荐给用户1
		val similarityTopUser = similarities.entries.filter(_.i == 1).
		  sortBy(_.value, false).
		  map(x=>(x.j, x.value)).
		  collect.
		  take(2)
		println("与用户1最相似的两个用户如下：")
		for (s <- similarityTopUser) {
		  // 找到这两个用户评过分的商品，与用户1没有评过分的物品
		  val userId = s._1
		  val ratingOfTemp = ratings.entries.filter(_.i == userId).
			map(x=>(x.j,x.value)).
			sortBy(_._1).
			map(_._1).
			collect().
			toList.
			toArray
		  println("用户" + userId + "对物品的评分:")
		  for (s <- ratingOfTemp) println(s)
		  
		  // 用户1与当前用户求差集
		  val dis = ratingOfTemp diff ratingOfUser1
		  println("用户" + userId + "要推荐给用户1的商品id为")
		  dis.foreach(println)
		}
		
		sc.stop()
	  }
	}
	```

### （二）、基于物品（Item CF）的协同过滤算法

#### 原理：

* 构建用户对物品的打分矩阵

* 根据余弦相似度公式计算**物品**相似度矩阵

* 对于当前用户评价高的物品，找出与之相似度最高的N个物品

* 将这N个物品推荐给用户

#### 代码实现：

	```scala
	import org.apache.spark.SparkConf
	import org.apache.spark.SparkContext
	import org.apache.spark.mllib.linalg.distributed.MatrixEntry
	import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
	import org.apache.log4j.Logger
	import org.apache.log4j.Level
	import org.apache.spark.rdd.RDD
	import org.apache.spark.mllib.linalg.distributed.RowMatrix
	import org.apache.spark.mllib.linalg.distributed.IndexedRow
	import org.apache.spark.mllib.linalg.SparseVector
	/*
	 * 建立物品的相似度，来进行推荐
	 */
	object ItemBasedCF {
	  def main(args: Array[String]): Unit = {
		Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
		Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

		//读入数据
		val conf = new SparkConf().setAppName("UserBaseModel").setMaster("local")
		val sc = new SparkContext(conf)
		val data = sc.textFile("hdfs://qujianlei:9000/data/ratingdata.txt")

		/*MatrixEntry代表一个分布式矩阵中的每一行(Entry)
		 * 这里的每一项都是一个(i: Long, j: Long, value: Double) 指示行列值的元组tuple。
		 * 其中i是行坐标，j是列坐标，value是值。*/
		val parseData: RDD[MatrixEntry] =
		  data.map(_.split(",") match { case Array(user, item, rate) => MatrixEntry(user.toLong, item.toLong, rate.toDouble) })

		//CoordinateMatrix是Spark MLLib中专门保存user_item_rating这种数据样本的
		val ratings = new CoordinateMatrix(parseData)

		/* 由于CoordinateMatrix没有columnSimilarities方法，所以我们需要将其转换成RowMatrix矩阵，调用他的columnSimilarities计算其相似性
		 * RowMatrix的方法columnSimilarities是计算，列与列的相似度，现在是user_item_rating，与基于用户的CF不同的是，这里不需要进行矩阵的转置，直接就是物品的相似*/
		val matrix: RowMatrix = ratings.toRowMatrix()

		//需求：为某一个用户推荐商品。基本的逻辑是：首先得到某个用户评价过（买过）的商品，然后计算其他商品与该商品的相似度，并排序；从高到低，把不在用户评价过
		//商品里的其他商品推荐给用户。
		//例如：为用户2推荐商品

		//第一步：得到用户2评价过（买过）的商品  take(5)表示取出所有的5个用户  2:表示第二个用户
		//解释：SparseVector：稀疏矩阵
		val user2pred = matrix.rows.take(5)(2)
		val prefs: SparseVector = user2pred.asInstanceOf[SparseVector]
		val uitems = prefs.indices //得到了用户2评价过（买过）的商品的ID   
		val ipi = (uitems zip prefs.values) //得到了用户2评价过（买过）的商品的ID和评分，即：(物品ID,评分)   
	//    for (s <- ipi) println(s)
	//    println("*******************")


		//计算物品的相似性，并输出
		val similarities = matrix.columnSimilarities()
		val indexdsimilar = similarities.toIndexedRowMatrix().rows.map {
		  case IndexedRow(idx, vector) => (idx.toInt, vector)
		}
	//    indexdsimilar.foreach(println)
	//    println("*******************")
		
		//ij表示：其他用户购买的商品与用户2购买的该商品的相似度
		val ij = sc.parallelize(ipi).join(indexdsimilar).flatMap {
		  case (i, (pi, vector: SparseVector)) => (vector.indices zip vector.values)
		}

		//ij1表示：其他用户购买过，但不在用户2购买的商品的列表中的商品和评分
		val ij1 = ij.filter { case (item, pref) => !uitems.contains(item) }
		//ij1.foreach(println)
		//println("*******************")

		//将这些商品的评分求和，并降序排列，并推荐前两个物品
		val ij2 = ij1.reduceByKey(_ + _).sortBy(_._2, false).take(2)
		println("********* 推荐的结果是 ***********")
		ij2.foreach(println)
	  }
	}
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

	```scala
	import org.apache.spark.mllib.recommendation.ALS
	import org.apache.log4j.Logger
	import org.apache.log4j.Level
	import org.apache.spark.SparkConf
	import org.apache.spark.SparkContext
	import org.apache.spark.mllib.recommendation.Rating
	import scala.io.Source
	import org.apache.spark.rdd.RDD
	import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

	object ALSDemo {
	  def main(args: Array[String]): Unit = {
		Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
		Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

		//读入数据，并转换为RDD[Rating]，得到评分数据
		val conf = new SparkConf().setAppName("UserBaseModel").setMaster("local")
		val sc = new SparkContext(conf)
		val productRatings = loadRatingData("hdfs://qujianlei:9000/ratingdata.txt")
		val prodcutRatingsRDD:RDD[Rating] = sc.parallelize(productRatings)
		
		//输出一些信息
		  val numRatings = prodcutRatingsRDD.count
	//    val numUsers = prodcutRatingsRDD.map(x=>x.user).distinct().count
	//    val numProducts = prodcutRatingsRDD.map(x=>x.product).distinct().count
	//    println("评分数：" + numRatings +"\t 用户总数：" + numUsers +"\t 物品总数："+ numProducts)
	 
		/*查看ALS训练模型的API
			ALS.train(ratings, rank, iterations, lambda)
					参数说明：ratings：评分矩阵
						   rank：小矩阵中，特征向量的个数。推荐的经验值：建议： 10~200之间
								 rank越大，表示：拆分越准确
								 rank越小，表示：速度越快
								 
						   iterations:运行时的迭代（循环）次数，经验值：10左右
						   lambda：控制拟合的正则化过程，值越大，表示正则化过程越厉害；如果这个值越小，越准确 ，使用0.01
		*/    
		//val model = ALS.train(prodcutRatingsRDD, 50, 10, 0.01)
		val model = ALS.train(prodcutRatingsRDD, 10, 5, 0.5)
		val rmse = computeRMSE(model,prodcutRatingsRDD,numRatings)
		println("误差：" + rmse)
		
		
		//使用该模型，来进行推荐
		//需求: 给用户1推荐2个商品                                        用户ID   几个商品
		val recomm = model.recommendProducts(1, 2)
		recomm.foreach(r=>{ 
		  println("用户：" + r.user.toString() +"\t 物品："+r.product.toString()+"\t 评分:"+r.rating.toString())
		})    
		
		sc.stop()
	  }
	  
		//计算RMSE ： 均方根误差
	  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
		val predictions: RDD[Rating] = model.predict((data.map(x => (x.user, x.product))))
		val predictionsAndRating = predictions.map {
		  x => ((x.user, x.product), x.rating)
		}.join(data.map(x => ((x.user, x.product), x.rating))).values

		math.sqrt(predictionsAndRating.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)

	  }
	  
	  //加载数据
	  def loadRatingData(path:String):Seq[Rating] = {
		val lines = Source.fromFile(path).getLines()
		
		//过滤掉评分是0的数据
		val ratings = lines.map(line=>{
			val fields = line.split(",")
			//返回Rating的对象 : 用户ID、物品ID、评分数据
			Rating(fields(0).toInt,fields(1).toInt,fields(2).toDouble)
		}).filter(x => x.rating > 0.0)
		
		//转换成  Seq[Rating]
		if(ratings.isEmpty){
		  sys.error("Error ....")
		}else{
		  //返回  Seq[Rating]
		  ratings.toSeq
		}
		
	  }
	}
	```
