package com.agoda.ml.spark.hpopt
package gaussian

import com.agoda.ml.spark.hpopt.hyperparam.HyperparameterEvalPoint

/**
 * This is an interface for a Gaussian Process kernel. The kernel will be used to calculate the correlation between 2
 * random variables inside the Gaussian Process.
 * In the domain of hyperparameter optimization, the evaluation points of a Gaussian Process are hyperparameters. In
 * consequence, the kernel function maps a pair of [[HyperparameterEvalPoint]] to [[Double]].
 */
abstract class GaussianProcessKernel extends ((HyperparameterEvalPoint, HyperparameterEvalPoint) => Double)

/**
 * The squared exponential kernel is a common choice as a Gaussian Process kernel for continuous input. As it is only
 * using the residuals of a pair of evaluation points, it's definition can be extended to any input type that provides
 * a residual method. As residuals are calculated component-wise, input types can be mixed.
 * Example for the residual of a non-continuous input: If a categorical input pair equals each other the residual is 0,
 * if not, it is a constant positive value.
 * @param lengthScale the larger the lengthScale parameter the less the kernel function value / correlation decreases with larger residuals
 */
case class SEKernel(lengthScale: Double) extends GaussianProcessKernel {

  override def apply(x: HyperparameterEvalPoint, y: HyperparameterEvalPoint): Double = {
    require(x.hyperparameters.length == y.hyperparameters.length, "Number of hyperparameters must be the same")
    require(lengthScale > 0, s"lengthScale must be positive, but is $lengthScale")
    val residuals = x residualTo y
    val squaredDistance: Double = residuals dot residuals
    math.exp( - squaredDistance / (2 * lengthScale * lengthScale))
  }

}

