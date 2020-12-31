package com.agoda.ml.spark.hpopt
package hyperparam

import org.apache.spark.sql.types._

import scala.util.Random

/**
 * Represents a hyperparameter including its value
 * @param hyperparameterType the underlying hyperparameter type
 * @param value the value of the hyperparameter
 * @tparam A value type of the hyperparameter
 */
case class Hyperparameter[A] (hyperparameterType: HyperparameterType[A], value: A) {

  /**
   * Calculates the residual between this and another hyperparameter. This residual is used to calculate values of the
   * Gaussian Process Kernel inside [[gaussian.GaussianProcessKernel]]. This makes a kernel robust against all types of
   * hyperparameters, as soon as a method to calculate the residuals is provided.
   * @param other the other hyperparameter
   * @return residual between the 2 hyperparameters
   */
  def residualTo(other: Hyperparameter[_]): Double = other match {
    case sameType: Hyperparameter[A] => hyperparameterType.getResidual(this, sameType)
  }
}

/**
 * Represents a type of Hyperparameter. Unlike [[Hyperparameter]], it does not have any value fields.
 * Provided a value or a [[Random]] generator, it can create a [[Hyperparameter]] object.
 */
trait HyperparameterType[A] {

  /**
   * Referring to [[org.apache.spark.ml.param.Param.name]]
   */
  val paramName: String

  /**
   * The maximum residual that should exist between to values of this [[HyperparameterType]]
   */
  val maxResidual: Double

  /**
   * The spark SQL data type
   */
  val dataType: DataType

  /**
   * Creates a [[Hyperparameter]] with this type and the provided value
   * @param x hyperparameter value
   * @return hyperparameter
   */
  def createHp(x: A): Hyperparameter[A] = Hyperparameter(this, x)

  /**
   * Creates a [[Hyperparameter]] with this type and the provided value
   * @param x hyperparameter value should match type [[A]]
   * @return hyperparameter
   */
  def createHpFromAny(x: Any): Hyperparameter[A] = x match {case a: A => createHp(a)}

  /**
   * Creates a [[Hyperparameter]] with a random value
   * @param random random generator
   * @return hyperparameter
   */
  def createRandomHp(random: Random): Hyperparameter[A]

  /**
   * Calculates the residual between 2 hyperparameters of this type. This residual is used in the [[GaussianProcessKernel]]
   * @param left first hyperparameter
   * @param right second hyperparameter
   * @return residual between the 2 hyperparameters
   */
  def getResidual(left: Hyperparameter[A], right: Hyperparameter[A]): Double
}

case class CategoricalHyperparameterType(paramName: String, categories: Seq[String], maxResidual: Double = 1.0)
  extends HyperparameterType[String] {
  override val dataType: DataType = StringType
  override def createRandomHp(random: Random): Hyperparameter[String] = createHp(categories(random.nextInt(categories.size)))
  override def getResidual(left: Hyperparameter[String], right: Hyperparameter[String]): Double = if (left == right) 0 else maxResidual
}

case class BooleanHyperparameterType(paramName: String, maxResidual: Double = 1.0) extends HyperparameterType[Boolean] {
  override val dataType: DataType = BooleanType
  override def createRandomHp(random: Random): Hyperparameter[Boolean] = createHp(random.nextBoolean())
  override def getResidual(left: Hyperparameter[Boolean], right: Hyperparameter[Boolean]): Double =
    if (left == right) 0 else maxResidual
}

trait NumericalHyperparameterType[A] extends HyperparameterType[A] {
  val scale: Scale
  def createHpFromUnitValue(x: Double): Hyperparameter[A]
  override def createRandomHp(random: Random): Hyperparameter[A] = createHpFromUnitValue(random.nextDouble())
}

case class DoubleHyperparameterType(paramName: String, scale: Scale, maxResidual: Double = 1.0)
  extends NumericalHyperparameterType[Double] {
  override val dataType: DataType = DoubleType
  override def createHpFromUnitValue(x: Double): Hyperparameter[Double] = createHp(scale.fromUnitInterval(x))
  override def getResidual(left: Hyperparameter[Double], right: Hyperparameter[Double]): Double =
    (scale.toUnitInterval(right.value) - scale.toUnitInterval(left.value)) * maxResidual
}

case class IntHyperparameterType(paramName: String, scale: Scale, maxResidual: Double = 1.0)
  extends NumericalHyperparameterType[Int] {
  override val dataType: DataType = IntegerType
  override def createHpFromUnitValue(x: Double): Hyperparameter[Int] = createHp(scale.fromUnitToInt(x))
  override def getResidual(left: Hyperparameter[Int], right: Hyperparameter[Int]): Double =
    (scale.toUnitInterval(right.value) - scale.toUnitInterval(left.value)) * maxResidual
}
