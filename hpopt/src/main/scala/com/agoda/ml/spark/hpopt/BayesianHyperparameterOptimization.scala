package com.agoda.ml.spark.hpopt

import com.agoda.ml.spark.hpopt.gaussian._
import com.agoda.ml.spark.hpopt.hyperparam.{HyperparameterEvalPoint, HyperparameterSpace}

import scala.util.Random

/**
 * This class proposes which hyperparameter values to evaluate in the next step
 * Each [[org.apache.spark.ml.param.Params]] / [[org.apache.spark.ml.PipelineStage]] instance can be updated by using
 * {{{hyperparameterEvalPoint.copyPipelineStage(paramsInstance)}}}
 * or
 * {{{paramsInstance.copy(hyperparameterEvalPoint.createParamMap(paramsInstance))}}}
 *
 * @param hpSpace search inside this [[HyperparameterSpace]] for optimal values
 * @param seed random seed, will only override seed of hpSpace if the seed of the latter is [[None]]
 * @param gpf [[GaussianProcessFactory]] to use for Bayesian Optimization
 * @param acquisitionFct the acquisition function to optimize
 * @param maximize if true, problem will be treated as maximization problem, if false, as minimization problem
 */
class BayesianHyperparameterOptimization(hpSpace: HyperparameterSpace,
                                         val seed: Option[Long] = None,
                                         val gpf: GaussianProcessFactory = GaussianProcessFactory(),
                                         val acquisitionFct: GPAcquisitionFct = UCB(),
                                         val monteCarloSteps: Int = 10000,
                                         val maximize: Boolean = true) {

  val (_hpSpace, random) = (seed, hpSpace.seed) match {
    case (Some(s), Some(_)) => (hpSpace, new Random(s))
    case (Some(s), None)    => (hpSpace.copy(seed = seed), new Random(s))
    case (None, Some(s))    => (hpSpace, new Random(s))
    case (None, None)       => (hpSpace, new Random())
  }
  val monteCarloSeed: Option[Int] = if (seed.isDefined) Some(random.nextInt()) else None
  val monteCarlo: MonteCarlo = MonteCarlo(monteCarloSeed)

  /**
   * Creates a new hyperparameter evaluation point randomly
   * @return random hyperparameter evalution point
   */
  def getNextRandom: HyperparameterEvalPoint = _hpSpace.createRandomEvalPoint

  /**
   * Creates a new hyperparameter evaluation point by bayesian optimization
   * @param previousHp previous evaluated hyperparameter values
   * @param previousMetrics previous evaluation results
   * @return optimal hyperparameter evalution point
   */
  def getNext(previousHp: Array[HyperparameterEvalPoint], previousMetrics: Array[Double]): HyperparameterEvalPoint =
    getNextPlusGP(previousHp, previousMetrics)._1


  /**
   * Creates a new hyperparameter evaluation point by bayesian optimization plus returns a Map of properties of the
   * Gaussian Prior for the next evaluation point.
   * This values can be used for better understanding or early stopping criteria.
   * - expectation: Expectation
   * - variance: Variance
   * - normalized-expectation: Expectation calculated by using normalized metrics (invertal [-0.5,0.5]). This might be used to calculate the acquisition function.
   * - acquisition-fct-value: Value of the acquisition function, very likely to be based on normalized expectation.
   *
   * @param previousHp previous evaluated hyperparameter values
   * @param previousMetrics previous evaluation results
   * @return A Tuple containing the optimal hyperparameter evaluation point and a map of properties
   */
  def getNextPlusProperties(previousHp: Array[HyperparameterEvalPoint], previousMetrics: Array[Double]): (HyperparameterEvalPoint, Map[String, Double]) = {
    import BayesianHyperparameterOptimization.GPPropertyNames._
    val (nextHp, gp) = getNextPlusGP(previousHp, previousMetrics)
    val gpProperties = Map(
      EXPECTATION            -> gp.expectedValue(nextHp),
      VARIANCE               -> gp.variance(nextHp),
      NORMALIZED_EXPECTATION -> gp.normalizedExpectedValue(nextHp),
      ACQUISITION_FCT_VALUE  -> acquisitionFct.acquisitionFct(gp)(nextHp)
    )

    (nextHp, gpProperties)
  }

  /**
   * Creates a new hyperparameter evaluation point by bayesian optimization plus returns the [[GaussianProcess]] and the
   * scaled x-value to calculate properties of the evaluation point.
   *
   * @param previousHp previous evaluated hyperparameter values
   * @param previousMetrics previous evaluation results
   * @return (nextHp, gp): nextHp is the optimal hyperparameter evalution point, you can get properties of
   *         the GaussianProcess by doing gp.variance(nextHp)
   */
  def getNextPlusGP(previousHp: Array[HyperparameterEvalPoint], previousMetrics: Array[Double])
  : (HyperparameterEvalPoint, GaussianProcess) = {

    val objectiveValues = if (maximize) previousMetrics else previousMetrics.map(m => -m)
    val gp = gpf.createGaussianProcess(previousHp, objectiveValues)
    val nextHp = monteCarlo.argmax(acquisitionFct.acquisitionFct(gp), _hpSpace, monteCarloSteps)

    (nextHp, gp)
  }

}

object BayesianHyperparameterOptimization {

  def apply(hpSpace: HyperparameterSpace,
            seed: Long,
            gpf: GaussianProcessFactory = GaussianProcessFactory(),
            acquisitionFct: GPAcquisitionFct = UCB(),
            monteCarloSteps: Int = 10000,
            maximize: Boolean = true): BayesianHyperparameterOptimization =
    new BayesianHyperparameterOptimization(hpSpace, Some(seed), gpf, acquisitionFct, monteCarloSteps, maximize)

  object GPPropertyNames {

    val EXPECTATION            = "expectation"
    val VARIANCE               = "variance"
    val NORMALIZED_EXPECTATION = "normalized-expectation"
    val ACQUISITION_FCT_VALUE  = "acquisition-fct-value"

  }
}
