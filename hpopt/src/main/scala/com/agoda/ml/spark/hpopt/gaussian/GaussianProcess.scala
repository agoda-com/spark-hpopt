package com.agoda.ml.spark.hpopt
package gaussian

import breeze.linalg.{DenseMatrix, DenseVector}
import com.agoda.ml.spark.hpopt.hyperparam.HyperparameterEvalPoint

/**
 * Represents a Gaussian Process for Bayesian Optimization.
 * @param evalPoints The hyperparameters where the objective function is already evaluated
 * @param values The values of the objective functions, no scaling necessary
 * @param kernel Gaussian Process Kernel to calculate the correlations
 */
class GaussianProcess(val evalPoints: Array[HyperparameterEvalPoint], val values: DenseVector[Double],
                      val kernel: GaussianProcessKernel = SEKernel(lengthScale = 0.5)) {

  require(evalPoints.nonEmpty, "evalPoints need to be non-empty")
  require(evalPoints.length == values.length, "Evaluation points array and objective functions value Array must be same length")
  require(evalPoints.tail.forall(_.hyperparameters.length == evalPoints.head.hyperparameters.length), "All hyperparameter vectors must be same length")

  val epsilon: Double = 1e-8

  /** Hyperparameter dimension */
  val d: Int = evalPoints.head.hyperparameters.length
  /** Number of evaluations */
  val t: Int = evalPoints.length

  /** Statistic values */
  lazy val valueArray: Array[Double] = values.toArray
  lazy val max: Double = valueArray.max
  lazy val min: Double = valueArray.min
  lazy val argmax: Int = valueArray.indexOf(max)
  lazy val bestEvalPoint: HyperparameterEvalPoint = evalPoints(argmax)
  lazy val normalizedValues: DenseVector[Double] = {
    val diameter = max - min
    if (diameter > 0) new DenseVector(valueArray.map(v => (v - min) / diameter - 0.5))
    else values - values
  }

  /** Do heavy calculation only once */
  lazy val matrixK: DenseMatrix[Double] = new DenseMatrix[Double](
    t, t, for {xi <- evalPoints; xj <- evalPoints} yield (kernel(xi, xj))
  )
  lazy val normalizedMatrixKInvF: DenseVector[Double] = matrixK \ normalizedValues

  /**
   * Normalized expected value of the next possible element of the Gaussian Process
   * @param x evaluation point
   * @return expected value
   */
  def normalizedExpectedValue(x: HyperparameterEvalPoint): Double = {
    requireVectorLength(x)
    getVectorK(x) dot normalizedMatrixKInvF
  }

  def expectedValue(x: HyperparameterEvalPoint): Double = (normalizedExpectedValue(x) + 0.5) * (max - min) + min

  /**
   * Variance of the next possible element of the Gaussian Process
   * @param x evaluation point
   * @return variance
   */
  def variance(x: HyperparameterEvalPoint): Double = {
    requireVectorLength(x)
    val vectorK = getVectorK(x)
    Math.max(kernel(x, x) - vectorK.t * (matrixK \ vectorK), epsilon)
  }

  def getVectorK(x: HyperparameterEvalPoint): DenseVector[Double] = {
    requireVectorLength(x)
    new DenseVector[Double](evalPoints.map(xj => kernel(x, xj)))
  }

  private def requireVectorLength(x: HyperparameterEvalPoint): Unit =
    require(x.hyperparameters.length == d, s"x has ${x.hyperparameters.length} Hyperparameters, but needs to be $d")
}

object GaussianProcess {

  def apply(evalPoints: Array[HyperparameterEvalPoint], values: DenseVector[Double],
            kernel: GaussianProcessKernel = SEKernel(lengthScale = 0.5)): GaussianProcess =
    new GaussianProcess(evalPoints, values, kernel)

  def apply(evalPoints: Array[HyperparameterEvalPoint], values: Array[Double], kernel: GaussianProcessKernel)
  : GaussianProcess =
    new GaussianProcess(evalPoints, new DenseVector[Double](values), kernel)

}

/**
 * Creates a [[GaussianProcess]] instance as soon as provided with evaluation points and objective function values
 * @param kernel kernel for the gaussian process
 */
case class GaussianProcessFactory(kernel: GaussianProcessKernel = SEKernel(lengthScale = 0.5)){

  def createGaussianProcess(evalPoints: Array[HyperparameterEvalPoint], values: DenseVector[Double]): GaussianProcess =
    GaussianProcess(evalPoints, values, kernel)

  def createGaussianProcess(evalPoints: Array[HyperparameterEvalPoint], values: Array[Double]): GaussianProcess =
    GaussianProcess(evalPoints, values, kernel)

}

