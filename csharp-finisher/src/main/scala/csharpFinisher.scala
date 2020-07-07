import java.nio.file.Paths

import org.apache.spark.ml.feature.{HashingTF, IDFModel, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, desc, lit, udf}

  object csharpFinisher {


    def calcNorm(vectorA: SparseVector): Double = {
      var norm = 0.0
      println(s"vector size: ${vectorA.size} ${vectorA.indices.length} ${vectorA.values.length}")
      for( i <- 0 to vectorA.indices.length-1) {
        norm += vectorA.values(i) * vectorA.indices(i)
      }

      (math.sqrt(norm))

    }


    def cosineSimilarity(vectorA: SparseVector, vectorB:SparseVector,normASqrt:Double,normBSqrt:Double) :(Double) = {
      var dotProduct = 0.0
      for (i <-  vectorA.indices){ dotProduct += vectorA(i) * vectorB(i) }
      val div = (normASqrt * normBSqrt)
      if( div == 0 ) (0)
      else (dotProduct / div)
    }


    def main(args: Array[String]): Unit = {

      println( "hey yo")

      val spark = SparkSession
              .builder()
              .config("spark.master", "local")
              .getOrCreate();

            import spark.implicits._

            val intermediateDir = args(0)

            val searchTermString = args.drop(1).mkString(" ")

            //Load .Net objects
            val tokenizer = Tokenizer.load(Paths.get(intermediateDir,"temp.tokenizer").toString)
            val hashingTF = HashingTF.load(Paths.get(intermediateDir,"temp.hashingTF").toString)
            val idfModel = IDFModel.load(Paths.get(intermediateDir,"temp.idfModel").toString)

            //Load model data
            val modelData = spark.read.parquet(Paths.get(intermediateDir,"temp.parquet").toString)

            val calcNormDF = udf[Double,SparseVector](calcNorm)

            val normalizedModelData = modelData.withColumn("norm",calcNormDF(col("features")))
            normalizedModelData.show(5)
            //Load search term
            val searchTerm = Seq(("1", searchTermString)).toDF("_id", "Content")
            val words = tokenizer.transform(searchTerm)
            val feature = hashingTF.transform(words)

            val search = idfModel
              .transform(feature)
              .withColumnRenamed("features", "features2")
              .withColumn("norm2", calcNormDF(col("features2")))


            val results = search.crossJoin(normalizedModelData)
            val calcCosineUDF =  udf[Double,SparseVector,SparseVector,Double,Double](cosineSimilarity)
            val similarResults = results
              .withColumn("similarity", calcCosineUDF(col("features"), col("features2"), col("norm"), col("norm2")))
              .select("path", "similarity")
              .filter("similarity > 0.0")
              .orderBy(desc("similarity"))
              .limit(10)
              .withColumn("Search Term", lit(searchTermString))

            similarResults.show(100, 100000)
    }
  }




//
//tokenizer.Save("/Users/ed/spark-temp/temp.tokenizer")
//hashingTF.Save("/Users/ed/spark-temp/temp.hashingTF")
//filtered.Write.Mode("overwrite").Parquet("/Users/ed/spark-temp/temp.parquet")
//// Directory.Delete("/Users/ed/spark-temp/temp.idfModel");
//idfModel.Save("/Users/ed/spark-temp/temp.idfModel")