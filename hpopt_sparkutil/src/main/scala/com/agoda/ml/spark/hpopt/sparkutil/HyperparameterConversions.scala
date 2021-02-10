package com.agoda.ml.spark.hpopt.sparkutil

import com.agoda.ml.spark.hpopt.hyperparam.{HyperparameterEvalPoint, HyperparameterSpace}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._

/** Functions to read/write hyperparameters from/to spark DataFrames */
object HyperparameterConversions {

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Seq[(String, Array[Double])], orderColumn: String)
              (implicit spark: SparkSession): DataFrame = {

    metrics.foreach(m => require(m._2.length == hyperparameters.length, "All metric value arrays must be same length as hyperparameters"))

    val hyperparameterTypes = hyperparameters.head.hyperparameters.map(_.hyperparameterType)

    val structFieldJsons = hyperparameterTypes.zipWithIndex.map{
      case (ht, i) => ("name" -> s"${ht.paramName}__$i") ~ ("type", ht.dataType) ~ ("nullable" -> false)
    } ++ metrics.map(metric => ("name" -> metric._1) ~ ("type" -> "double") ~ ("nullable" -> false)) ++
      Array(("name" -> orderColumn) ~ ("type" -> "integer") ~ ("nullable" -> false))
    val schema = DataType.fromJson(compact(render(
      ("type" -> "struct") ~ ("fields" -> structFieldJsons.toList)
    ))).asInstanceOf[StructType]

    val rows = (hyperparameters zip metrics.map(_._2).transpose).zipWithIndex.toSeq
      .map{case ((hp, metricValues), i) => Row.fromSeq(hp.hyperparameters.map(_.value).toSeq ++ metricValues :+ i)}

    spark.createDataFrame(rows.asJava, schema).repartition(1).orderBy(orderColumn)
  }

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Seq[(String, Array[Double])])
              (implicit spark: SparkSession): DataFrame = createDF(hyperparameters, metrics, "order")

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Array[Double],
               metricColumnName: String = "metric", orderColumn: String = "order")
              (implicit spark: SparkSession): DataFrame =
    createDF(hyperparameters, Seq((metricColumnName, metrics)), orderColumn)

  def readDF(hpSpace: HyperparameterSpace, df: DataFrame, metricColumns: Seq[String], orderColumn: String)
  : (Array[HyperparameterEvalPoint], Map[String, Array[Double]]) = {
    val ordered =
      if (df.columns.contains(orderColumn)) df.orderBy(orderColumn)
      else df.orderBy(metricColumns.map(col(_).desc): _*)
    val evalPoints = ordered.selectExpr(hpSpace.hpNamesNumbered: _*).collect
      .map(row => hpSpace.createEvalPoint(row.toSeq))
    val metricValues: Array[Array[Double]] = ordered.selectExpr(metricColumns: _*).collect
      .map(_.toSeq.map{case d: Double => d}.toArray)
    def transpose(array: Array[Array[Double]]): Array[Array[Double]] = {
      val d = array.head.length
      array.tail.foreach(innerArray => require(innerArray.length == d,
        "Can not transpose, inner arrays don't have the same length."))
      (0 until d).map(i => array.map(innerArray => innerArray(i))).toArray
    }
    (evalPoints, (metricColumns zip transpose(metricValues)).toMap)
  }

  def readDF(hpSpace: HyperparameterSpace, df: DataFrame, metricColumns: Seq[String])
  : (Array[HyperparameterEvalPoint], Map[String, Array[Double]]) = readDF(hpSpace, df, metricColumns, "order")

  def readDF(hpSpace: HyperparameterSpace, df: DataFrame, metricColumnName: String = "metric", orderColumn: String = "order")
  : (Array[HyperparameterEvalPoint], Array[Double]) = {
    val (evalPoints, metricsMap) = readDF(hpSpace, df, Seq(metricColumnName), orderColumn)
    (evalPoints, metricsMap(metricColumnName))
  }
}
