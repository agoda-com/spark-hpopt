package com.agoda.ml.spark.hpopt
package gaussian

import breeze.stats.distributions.{Rand, RandBasis}
import com.agoda.ml.spark.hpopt.hyperparam.{HyperparameterEvalPoint, HyperparameterSpace}

/**
 * Provides methods for Monte Carlo optimization for functions with [[HyperparameterEvalPoint]] as input
 * @param seed random seed
 */
case class MonteCarlo(seed: Option[Int] = None) {

  /** Create a random number generator, with seed if provided */
  val randBasis: RandBasis = seed.map(RandBasis.withSeed).getOrElse(Rand)

  /**
   * Calculates the argmax for the given objective based on Monte Carlo optimization
   * @param objective objective function
   * @param hpSpace Hyperparameter Space
   * @param iterations number of iteration steps
   * @return
   */
  def argmax(objective: HyperparameterEvalPoint => Double, hpSpace: HyperparameterSpace, iterations: Int = 10000): HyperparameterEvalPoint = {

    @scala.annotation.tailrec
    def loop(argmax: HyperparameterEvalPoint, maxValue: Double, step: Int): (HyperparameterEvalPoint, Double) = {
      val evalPoint = hpSpace.createRandomEvalPoint
      val eval = objective(evalPoint)
      val (newArgmax: HyperparameterEvalPoint, newMaxValue: Double) = if (eval > maxValue) (evalPoint, eval) else (argmax, maxValue)
      if (step >= iterations - 1) (newArgmax, newMaxValue)
      else loop(newArgmax, newMaxValue, step + 1)
    }

    val startEvalPoint = hpSpace.createRandomEvalPoint
    loop(startEvalPoint, objective(startEvalPoint), 0)._1
  }
}

object MonteCarlo {

  def apply(seed: Int): MonteCarlo = MonteCarlo(Some(seed))

}
