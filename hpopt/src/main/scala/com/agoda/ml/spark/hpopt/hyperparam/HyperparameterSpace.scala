package com.agoda.ml.spark.hpopt
package hyperparam

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

  private def checkArrayLength(array: Array[_]): Unit = require(array.length == hyperparameterTypes.length,
    s"Length mismatch: hyperparameterTypes length is ${hyperparameterTypes.length}," +
      s"but values length is ${array.length}")
}

object HyperparameterSpace {
  def apply(hyperparameterTypes: Array[HyperparameterType[_]], seed: Long): HyperparameterSpace =
    HyperparameterSpace(hyperparameterTypes, Some(seed))
}
