package com.agoda.ml.spark.hpopt.hyperparam

import org.scalatest.FunSuite

class ScaleTest extends FunSuite{

  test("LinearScale throws error if lower >= upper"){
    assertThrows[IllegalArgumentException](LinearScale(lower = 5, upper = 5))
    assertThrows[IllegalArgumentException](LinearScale(lower = 6, upper = 2))
  }

  test("LinearScale throws error for value out of bounds"){
    val ls = LinearScale(lower = 5, upper = 7)
    assertThrows[IllegalArgumentException](ls.toUnitInterval(4))
    assertThrows[IllegalArgumentException](ls.toUnitInterval(8))
    assertThrows[IllegalArgumentException](ls.fromUnitInterval(-0.1))
    assertThrows[IllegalArgumentException](ls.fromUnitInterval(1.1))
  }

  test("Check LinearScale mapping on several values"){
    val ls = LinearScale(-1, 3)
    val originalValues = List(-1, 0, 1, 3)
    val expectedUnitValues = List(0d, 0.25, 0.5, 1d)
    (originalValues.map(ls.toUnitInterval(_)) zip expectedUnitValues).foreach{case (x,y) => assert(x == y)}
    (expectedUnitValues.map(ls.fromUnitInterval) zip originalValues).foreach{case (x,y) => assert(x == y)}
  }

  test("LogScale throws error if lower <= 0"){
    assertThrows[IllegalArgumentException](LogScale(lower = 0, upper = 2))
    assertThrows[IllegalArgumentException](LogScale(lower = -1, upper = 2))
  }

  test("Check LogScale mapping on several values"){
    val ls = LogScale(1, 10000)
    val originalValues = List(1, 10, 100, 10000)
    val expectedUnitValues = List(0d, 0.25, 0.5, 1d)
    (originalValues.map(ls.toUnitInterval(_)) zip expectedUnitValues).foreach{case (x,y) => assert(x == y)}
    (expectedUnitValues.map(ls.fromUnitInterval) zip originalValues).foreach{case (x,y) => assert(x == y)}
  }
}
