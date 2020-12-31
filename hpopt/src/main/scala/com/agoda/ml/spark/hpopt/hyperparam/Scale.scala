package com.agoda.ml.spark.hpopt
package hyperparam

/**
 * Trait for a Scale. A Scale defines how to map any interval to the unit interval [0,1].
 * This can be used for hyperparameters to map pseudo-random numbers to the interval of the hyperparameter
 */
trait Scale {

  /** lower bound of the interval */
  val lower: Double
  /** upper bound of the interval */
  val upper: Double
  require(upper > lower, s"Upper boundary $upper must be greater than lower boundary $lower")

  /**
   * Maps a value from the original interval {@literal [}[[lower]], [[upper]]{@literal ]} to the unit interval [0,1]
   * @param x value of original interval
   * @return value between 0 and 1
   */
  def toUnitInterval(x: Double): Double

  /**
   * Maps a value from the unit interval [0,1] back to the original interval {@literal [}[[lower]], [[upper]]{@literal ]}
   * @param x value between 0 and 1
   * @return value of original interval
   */
  def fromUnitInterval(x: Double): Double

  /**
   * Maps a value from the unit interval [0,1] to an integer between [[lower]] and [[upper]] inclusively.
   * @param x value between 0 and 1
   * @return integer
   */
  def fromUnitToInt(x: Double): Int = {
    val z = fromUnitInterval(x)
    (z + (z - lower) / (upper - lower)).toInt.min(upper.toInt)
  }

  /**
   * Throw error if the value is not in the unit interval
   * @param x
   */
  def requireUnitInterval(x: Double): Unit =
    require(x >= 0 && x <= 1, s"Provided value $x is not inside the unit interval [0,1]")

  /**
   * Throw error if the value is not in the defined interval
   * @param x
   */
  def requireInterval(x: Double): Unit = {
    require(x >= lower, s"Provided value $x is lower than the lower boundary $lower")
    require(x <= upper, s"Provided value $x is higher than the upper boundary $upper")
  }
}

/**
 * Simple linear mapping between any interval and the unit interval [0,1]
 * @param lower lower bound of the original interval
 * @param upper upper bound of the original interval
 */
case class LinearScale(lower: Double, upper: Double) extends Scale {

  override def toUnitInterval(x: Double): Double = {
    requireInterval(x)
    (x - lower) / (upper - lower)
  }

  override def fromUnitInterval(x: Double): Double = {
    requireUnitInterval(x)
    x * (upper - lower) + lower
  }

}

/**
 * Logarithmic mapping between any interval and the unit interval [0,1]
 * @param lower lower bound of the original interval, must be positive
 * @param upper upper bound of the original interval, must be positive
 */
case class LogScale(lower: Double, upper: Double) extends Scale {

  require(lower > 0, s"lower must be positive, but is $lower")
  private val innerLinearScale = LinearScale(Math.log10(lower), Math.log10(upper))

  override def toUnitInterval(x: Double): Double = {
    requireInterval(x)
    innerLinearScale.toUnitInterval(Math.log10(x))
  }

  override def fromUnitInterval(x: Double): Double = {
    requireUnitInterval(x)
    Math.pow(10, innerLinearScale.fromUnitInterval(x))
  }
}
