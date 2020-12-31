package com.agoda.ml.spark.hpopt
package hyperparam

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.collection.JavaConverters._
import scala.util.Random

/**
 * A collection of [[HyperparameterType]] to represent all hyperparameters that might be tuned.
 * Unlike [[HyperparameterEvalPoint]] it does not contain any hyperparameter values.
 *
 * @param hyperparameterTypes hyperparameters to be tuned
 * @param seed random seed to create random hyperparameter values
 */
case class HyperparameterSpace(hyperparameterTypes: Array[HyperparameterType[_]], seed: Option[Long] = None) {

  val random: Random = seed.map(new Random(_)).getOrElse(new Random())

  lazy val hpNames: Seq[String] = hyperparameterTypes.map(_.paramName)
  lazy val hpNamesNumbered: Seq[String] = hpNames.zipWithIndex.map{case (hpName, i) => s"${hpName}__$i"}

  def createRandomEvalPoint: HyperparameterEvalPoint = HyperparameterEvalPoint(hyperparameterTypes.map(_.createRandomHp(random)))

  def createEvalPoint(values: Seq[Any]): HyperparameterEvalPoint = {
    checkArrayLength(values.toArray)
    HyperparameterEvalPoint(
      (hyperparameterTypes zip values)
        .map{case (ht, value) => ht.createHpFromAny(value)}
    )
  }

  def readDF(df: DataFrame, metricColumns: Seq[String], orderColumn: String)
  : (Array[HyperparameterEvalPoint], Map[String, Array[Double]]) = {
    val ordered = if (df.columns.contains(orderColumn)) df.orderBy(orderColumn) else df.orderBy(metricColumns.map(col(_).desc): _*)
    val evalPoints = ordered.selectExpr(hpNamesNumbered: _*).collect.map(row => createEvalPoint(row.toSeq))
    val metricValues: Array[Array[Double]] = ordered.selectExpr(metricColumns: _*).collect.map(_.toSeq.map{case d: Double => d}.toArray)
    def transpose(array: Array[Array[Double]]): Array[Array[Double]] = {
      val d = array.head.length
      array.tail.foreach(innerArray => require(innerArray.length == d,
        "Can not transpose, inner arrays don't have the same length."))
      (0 until d).map(i => array.map(innerArray => innerArray(i))).toArray
    }
    (evalPoints, (metricColumns zip transpose(metricValues)).toMap)
  }

  def readDF(df: DataFrame, metricColumns: Seq[String])
  : (Array[HyperparameterEvalPoint], Map[String, Array[Double]]) = readDF(df, metricColumns, "order")

  def readDF(df: DataFrame, metricColumnName: String = "metric", orderColumn: String = "order")
  : (Array[HyperparameterEvalPoint], Array[Double]) = {
    val (evalPoints, metricsMap) = readDF(df, Seq(metricColumnName), orderColumn)
    (evalPoints, metricsMap(metricColumnName))
  }

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Seq[(String, Array[Double])], orderColumn: String)
              (implicit sqlContext: SQLContext): DataFrame = {
    hyperparameters.foreach(hep => checkArrayLength(hep.hyperparameters))
    metrics.foreach(m => require(m._2.length == hyperparameters.length, "All metric value arrays must be same length as hyperparameters"))
    val schema = StructType(
      hyperparameterTypes.zipWithIndex.map{case (ht, i) => StructField(s"${ht.paramName}__$i", ht.dataType, nullable = false)}
        ++ metrics.map(metric => StructField(metric._1, DoubleType))
        :+ StructField(orderColumn, IntegerType)
    )
    val rows = (hyperparameters zip metrics.map(_._2).transpose).zipWithIndex.toSeq
      .map{case ((hp, metricValues), i) => Row.fromSeq(hp.hyperparameters.map(_.value).toSeq ++ metricValues :+ i)}

    sqlContext.sparkSession.createDataFrame(rows.asJava, schema).repartition(1).orderBy(orderColumn)
  }

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Seq[(String, Array[Double])])
              (implicit sqlContext: SQLContext): DataFrame = createDF(hyperparameters, metrics, "order")

  def createDF(hyperparameters: Array[HyperparameterEvalPoint], metrics: Array[Double],
               metricColumnName: String = "metric", orderColumn: String = "order")
              (implicit sqlContext: SQLContext): DataFrame =
    createDF(hyperparameters, Seq((metricColumnName, metrics)), orderColumn)

  private def checkArrayLength(array: Array[_]): Unit = require(array.length == hyperparameterTypes.length,
    s"Length mismatch: hyperparameterTypes length is ${hyperparameterTypes.length}," +
      s"but values length is ${array.length}")
}

object HyperparameterSpace {
  def apply(hyperparameterTypes: Array[HyperparameterType[_]], seed: Long): HyperparameterSpace =
    HyperparameterSpace(hyperparameterTypes, Some(seed))
}
