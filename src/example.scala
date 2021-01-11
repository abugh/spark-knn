import scala.util.Random

object example {
  val embeddingSize = 300000
  val vectorSize = 10
  val topK = 20
  val partitionNum = 200
  // generate randomEmbedding, if you have your own embedding dataset, just use it instead.
  val randomEmbedding = (0 to embeddingSize - 1).map{i => (i.toString, (0 to vectorSize - 1).map{j => Random.nextDouble()})}.seq.toDF("id", "vector")

  def main(args: Array[String]): Unit = {
    val obj = SparkEmbeddingKNN(randomEmbedding.as[Embedding], randomEmbedding.as[Embedding], true)
    val matchDF = obj.getKnnByPartitionUDF(topK, partitionNum)
    matchDF.show()
  }
}
