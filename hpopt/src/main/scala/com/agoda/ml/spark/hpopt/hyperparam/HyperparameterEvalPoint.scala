package com.agoda.ml.spark.hpopt
package hyperparam

import breeze.linalg.DenseVector
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{ParamMap, ParamPair, Params}

/**
 * Represents a collection of [[Hyperparameter]], supposed to be created by [[HyperparameterSpace]]
 * @param hyperparameters
 */
case class HyperparameterEvalPoint(hyperparameters: Array[Hyperparameter[_]]) {

  def residualTo(other: HyperparameterEvalPoint): DenseVector[Double] = new DenseVector[Double](
    (this.hyperparameters zip other.hyperparameters).map{case (left, right) => left residualTo right}
  )

  /**
   * Update a pipeline stage with the new hyperparameter values
   * @param ps pipeline stage
   * @return pipeline stage with updated parameters
   */
  def copyPipelineStage(ps: PipelineStage): PipelineStage = ps.copy(createParamMap(ps))

  /**
   * Create a [[ParamMap]] out of the hyperparameters
   * @param params pipeline stage
   * @return the new parameter values as a [[ParamMap]]
   */
  def createParamMap(params: Params): ParamMap = HyperparameterEvalPoint.createParamMap(params, hyperparameters)

}

/**
 * Companion object containing static methods for HyperparameterEvalPoint.
 */
object HyperparameterEvalPoint {

  /**
   * Create a [[ParamMap]] out of hyperparameters
   * @param params pipeline stage
   * @param hyperparameters hyperparameters used to create the [[ParamMap]]
   * @return the new parameters as a [[ParamMap]]
   */
  def createParamMap(params: Params, hyperparameters: Array[Hyperparameter[_]]): ParamMap = {
    val paramPairs = hyperparameters.map(hp => ParamPair(params.getParam(hp.hyperparameterType.paramName), hp.value))
    ParamMap(paramPairs: _*)
  }

  /**
   * Create a Seq of [[ParamMap]] out of the hyperparameters
   * @param paramsList Seq of pipeline stages and hyperparameters they take
   * @return Seq of [[ParamMap]] in the same order as the pipeline stages in paramsList
   */
  def createParamMaps(paramsList: Seq[(Params, Array[Hyperparameter[_]])]): Seq[ParamMap] = {
    paramsList.map{case (params, hyperparameters) => createParamMap(params, hyperparameters)}
  }

  /**
   * Update multiple pipeline stages with different hyperparameter values
   * @param psList Seq of pipeline stages and hyperparameters they take
   * @return Seq of pipeline stages with updated params in the same order as in psList
   */
  def copyPipelineStages(psList: Seq[(PipelineStage, Array[Hyperparameter[_]])]): Seq[PipelineStage] =
    psList.map{case (ps, hps) => ps.copy(createParamMap(ps, hps))}
}

