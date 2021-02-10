package com.agoda.ml.spark.hpopt
package sparkutil

import com.agoda.ml.spark.hpopt.hyperparam.{Hyperparameter, HyperparameterEvalPoint}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{ParamMap, ParamPair, Params}

/**
 * Provides methods to extract or transfer hyperparameters from [[HyperparameterEvalPoint]] instances
 */
object ParamsUtil {

  /**
   * Update a pipeline stage with new hyperparameter values
   * @param ps pipeline stage
   * @param hpEvalPoint New hyperparameter values
   * @return pipeline stage with updated parameters
   */
  def copyPipelineStage(ps: PipelineStage, hpEvalPoint: HyperparameterEvalPoint): PipelineStage =
    ps.copy(createParamMap(ps, hpEvalPoint))

  /**
   * Create a [[ParamMap]] out of a [[HyperparameterEvalPoint]]
   * @param params pipeline stage
   * @param hpEvalPoint contains the parameter values
   * @return the new parameter values as a [[ParamMap]]
   */
  def createParamMap(params: Params, hpEvalPoint: HyperparameterEvalPoint): ParamMap =
    createParamMap(params, hpEvalPoint.hyperparameters)

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
