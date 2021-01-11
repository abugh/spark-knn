import org.apache.spark.sql.{Dataset, DataFrame, functions => F}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum, Vector => BreezeVec}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import breeze.linalg.functions.{cosineDistance => BreezeCosineDistance}
import org.apache.spark.sql.types.DataTypes


case class Embedding(id: String, vector: Seq[Double])

case class EmbeddingMatchPair(id: String, id_candidate: String, score: Double, rank: Int)


"""
  Interface function
  @param embeddingMapMain: Main embedding map. We will find k candidates for each item in embeddingMapMain.
  @param embeddingMapCandidates: Candidate embedding map. Candidates are from this embedding map.
  @param sameEmbedding: indicates whether the 2 embedings are the same embedding. If the same , we need to filter out the item itself in candidates.
"""
case class SparkEmbeddingKNN(embeddingMapMain: Dataset[Embedding], embeddingMapCandidates: Dataset[Embedding], sameEmbedding: Boolean = true) {
  def partitionEmbedding(partitionNum: Int)(embedding: Dataset[Embedding]): DataFrame= {
    embedding
      .withColumn("split_index", F.abs(F.hash($"id")) % partitionNum)
      //split the embedding dataset to $partitionNum pieces
      .groupBy($"split_index")
      .agg(F.collect_list(F.struct("id", "vector")).as("embeddings"))
  }

  def toBreezeVec(array: Seq[Double]): BreezeVec[Double] = {
    DenseVector(array.toArray)
  }

  def cosineDistance(v1: BreezeVec[Double], v2: BreezeVec[Double]): Double = {
    BreezeCosineDistance(v1, v2)
  }

  def cosineDistance(v1: Seq[Double], v2: Seq[Double]): Double = {
    BreezeCosineDistance(toBreezeVec(v1), toBreezeVec(v2))
  }

  def cosineSimilarity(v1: BreezeVec[Double], v2: BreezeVec[Double]): Double = {
    1 - cosineDistance(v1, v2)
  }

  def cosineSimilarity(v1: Seq[Double], v2: Seq[Double]): Double = {
    1 - cosineDistance(v1, v2)
  }

  val cosineSimilaritySeqDoubleUDF: UserDefinedFunction = udf { (v1: Seq[Double], v2: Seq[Double]) =>
    cosineSimilarity(v1, v2)
  }

  def getSimilarity(embeddingMap: Map[String, BreezeVec[Double]], embeddingMapCandidate: Map[String, BreezeVec[Double]], scoreLowBound: Double): Map[String, Seq[(String, Double)]] = {
    val embeddingIds = embeddingMap.map(_._1).toArray
    val embeddingCandidateIds = embeddingMapCandidate.map(_._1).toArray
    embeddingIds.map { id =>
      (id, embeddingCandidateIds.map{ id_candidate =>
        (id_candidate, cosineSimilarity(embeddingMap(id), embeddingMapCandidate(id_candidate)))
      }.seq.filter(_._2 > scoreLowBound))
    }.seq.toMap
  }

  def rankByVideo(similaritySeq: Seq[(String, Double)], maxRank: Int): Seq[(String, Double)] = {
    similaritySeq
      .sortBy(_._2)
      .take(maxRank)
  }

  def defaultKnnFunction(embeddingMap: Map[String, BreezeVec[Double]], embeddingMapCandidate: Map[String, BreezeVec[Double]], topK: Int, scoreLowBound: Double): Seq[(String, String, Double)] = {
    val model        = getSimilarity(embeddingMap, embeddingMapCandidate, scoreLowBound)
    val embeddingIds = embeddingMap.map(_._1).toArray
    embeddingIds.par.flatMap { id =>
      val similars = rankByVideo(model(id), topK)
      similars.map { x =>
        (id, x._1, x._2)// id, id_candidate, score
      }
    }.seq
  }

  val defaultKnnUdf = F.udf {
    (embedding: Seq[Row], embeddingCandidate: Seq[Row], topK: Int, scoreLowBound: Double) =>
      val embeddingMap = embedding
        .map(em => {
          val uid = em.getAs[String](0)
          val vec = toBreezeVec(em.getAs[Seq[Double]](1))
          (uid, vec)
        })
        .toMap
      val embeddingMapCandidate = embeddingCandidate
        .map(em => {
          val uid = em.getAs[String](0)
          val vec = toBreezeVec(em.getAs[Seq[Double]](1))
          (uid, vec)
        })
        .toMap
      defaultKnnFunction(embeddingMap, embeddingMapCandidate, topK, scoreLowBound)
        .toSet
        .toSeq
  }
  """
    Interface function
    @param topK: the number of candidates return for each item in embeddingMain
    @param partitionNum: number of partitions to split the whole embedding dataset
    @param udfType: type of udf
    @param explicitUdf: you can input your own udf to fasten the process. Implementation can refer to defaultKnnUdf.
    @param scoreLowBound: embedding pairs under scoreLowBound will be dismissed. The score is Cosine Similarity.
  """
  def getKnnByPartitionUDF(topK: Int, partitionNum: Int = 200, udfType: String = "exlicit", explicitUdf: UserDefinedFunction = defaultKnnUdf, scoreLowBound: Double = 0.5): Dataset[EmbeddingMatchPair] = {
    val partitionedMain = embeddingMapMain.transform(partitionEmbedding(partitionNum))
    val partitionedCandidates = embeddingMapCandidates.transform(partitionEmbedding(partitionNum))

    val knnUdf = udfType match {
      case "explicit" => explicitUdf
      case _  => defaultKnnUdf
    }

    partitionedMain
      .crossJoin(
        partitionedCandidates
          .withColumnRenamed("embeddings", "embeddings_candidate")
          .withColumnRenamed("split_index", "split_index_candidate")
      )
      //drop the duplicated pair of pieces
      .withColumn("split_key",
        F.when(($"split_index" < $"split_index_candidate") || F.lit(!sameEmbedding), F.concat_ws("_", $"split_index", $"split_index_candidate"))
          .otherwise(F.concat_ws("_", $"split_index_candidate", $"split_index")))
      .dropDuplicates("split_key")
      .withColumn("knns",
        knnUdf($"embeddings",
          $"embeddings_candidate",
          F.lit(topK + 1),// + 1 means a placeholder for the same id
          F.lit(scoreLowBound)))
      .select(F.explode($"knns").as("knn"))
      .select($"knn._1".as("id"),
        $"knn._2".as("id_candidate"),
        $"knn._3".as("score"))
      .filter(F.lit(!sameEmbedding) || $"id" =!= $"id_candidate")
      .dropDuplicates("id", "id_candidate")
      .withColumn("rank", F.row_number().over(Window.partitionBy($"id").orderBy($"score".desc)))
      .filter($"rank" <= topK)
      .as[EmbeddingMatchPair]
  }
}