package com.agoda.ml.spark.hpopt
package gaussian

import com.agoda.ml.spark.hpopt.hyperparam.HyperparameterEvalPoint
import org.apache.commons.math3.distribution.NormalDistribution

/**
 * Interface acquisition function that is used in Bayesian Optimization with Gaussian Process Prior
 */
abstract class GPAcquisitionFct {

  /**
   * This function will be optimized inside the Gaussian Process
   * @param gp Gaussian Process prior containing previous evaluations
   * @param x evaluation point
   * @return acquisition function value
   */
  def acquisitionFct(gp: GaussianProcess)(x: HyperparameterEvalPoint): Double

}

/**
 * Upper confidence bound. The used formula can be found in A Tutorial on Bayesian Optimization of Expensive Cost
 * Functions [...] (https://arxiv.org/abs/1012.2599) section 2.3.3
 *
 * @param delta For greater delta, the importance of exploration grows more slowly
 * @param nu The greater nu, the more important is exploration
 */
case class UCB(delta: Double = 1d, nu: Double = 4d) extends GPAcquisitionFct {

  override def acquisitionFct(gp: GaussianProcess)(x: HyperparameterEvalPoint): Double = {
    lazy val tauParam: Double = 2 * Math.log(Math.pow(gp.t, gp.d.toDouble / 2 + 2) * Math.PI * Math.PI / (3 * delta))
    gp.normalizedExpectedValue(x) + Math.sqrt(nu * tauParam * gp.variance(x))
  }
}

/**
 * Expected Improvement
 * @param minimalImprovement minimal improvement to search for
 */
case class ExpectedImprovement(minimalImprovement: Double = 0.01) extends GPAcquisitionFct {

  override def acquisitionFct(gp: GaussianProcess)(x: HyperparameterEvalPoint): Double = {
    val ev = gp.normalizedExpectedValue(x)
    val sd = Math.sqrt(gp.variance(x))
    val z = (ev - gp.max - minimalImprovement) / sd
    val norm = new NormalDistribution()
    if (sd > 0) (ev - gp.max - minimalImprovement) * norm.cumulativeProbability(z) + sd * norm.density(z)
    else 0d
  }

}
