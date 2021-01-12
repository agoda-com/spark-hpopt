package com.agoda.ml.spark.hpopt.hyperparam

import org.scalatest.FunSuite

class HyperparameterTest extends FunSuite {

  private val precision = 1e-6

  test("Categorical hyperparameter"){

    val maxResidual = 0.7
    val hyperparamterType = CategoricalHyperparameterType("category", Seq("one", "two", "three"), maxResidual)

    val hpOne = hyperparamterType.createHp("one")
    val hpTwo = hyperparamterType.createHp("two")
    val hpAlsoOne = hyperparamterType.createHp("one")

    assert(hpOne.residualTo(hpTwo) == maxResidual, "Categorical hyperparameters with distinct values should" +
      s"produce max residual $maxResidual")
    assert(hpOne.residualTo(hpAlsoOne) == 0, "Categorical hyperparameters with same values should produce" +
      "residual 0")
  }

  test("Boolean hyperparameter"){

    val maxResidual = 1.3
    val hyperparamterType = BooleanHyperparameterType("boolean", maxResidual)

    val hpTrue = hyperparamterType.createHp(true)
    val hpFalse = hyperparamterType.createHp(false)
    val hpAlsoTrue = hyperparamterType.createHp(true)
    val hpAlsoFalse = hyperparamterType.createHp(false)

    assert(hpTrue.residualTo(hpFalse) == maxResidual, "Boolean hyperparameters with distinct values should" +
      s"produce max residual $maxResidual")
    assert(hpTrue.residualTo(hpAlsoTrue) == 0, "Boolean hyperparameters with same values should produce" +
      "residual 0")
    assert(hpFalse.residualTo(hpAlsoFalse) == 0, "Boolean hyperparameters with same values should produce" +
      "residual 0")
  }

  test("Double hyperparameter with linear scale"){

    val hyperparameterType = DoubleHyperparameterType("double", LinearScale(0.2, 0.7))
    val hp = hyperparameterType.createHp(0.3)
    val hp2 = hyperparameterType.createHp(0.6)
    val hp3 = hyperparameterType.createHp(0.3)

    assert(Math.abs(hp.residualTo(hp2) - 0.6) <= precision)
    assert(Math.abs(hp.residualTo(hp3)) <= precision, "Double hyperparameter with same values should produce" +
      "residual 0")
  }

  test("Integer hyperparameter with linear scale"){

    val hyperparameterType = IntHyperparameterType("integer", LinearScale(1, 100))
    val hp = hyperparameterType.createHp(10)
    val hp2 = hyperparameterType.createHp(100)
    val hp3 = hyperparameterType.createHp(10)

    assert(Math.abs(hp.residualTo(hp2) - 90d / 99) <= precision)
    assert(Math.abs(hp.residualTo(hp3)) <= precision, "Integer hyperparameter with same values should produce" +
      "residual 0")
  }

  test("Integer hyperparameter with log scale"){

    val hyperparameterType = IntHyperparameterType("integer", LogScale(1, 100))
    val hp = hyperparameterType.createHp(10)
    val hp2 = hyperparameterType.createHp(100)
    val hp3 = hyperparameterType.createHp(10)

    assert(Math.abs(hp.residualTo(hp2) - 0.5) <= precision)
    assert(Math.abs(hp.residualTo(hp3)) <= precision, "Integer hyperparameter with same values should produce" +
      "residual 0")
  }
}
