package com.agoda.ml.spark.hpopt
package hyperparam

import breeze.linalg.DenseVector

/**
 * Represents a collection of [[Hyperparameter]], supposed to be created by [[HyperparameterSpace]]
 * @param hyperparameters
 */
case class HyperparameterEvalPoint(hyperparameters: Array[Hyperparameter[_]]) {

  def residualTo(other: HyperparameterEvalPoint): DenseVector[Double] = new DenseVector[Double](
    (this.hyperparameters zip other.hyperparameters).map{case (left, right) => left residualTo right}
  )

}
