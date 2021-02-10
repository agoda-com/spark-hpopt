import sbt.Keys._
import sbt._


val baseSettings = Seq(
  organization := "com.agoda.ml.spark.hpopt",
  scalaVersion := "2.11.12",
  fork in Test := true,
  javaOptions in Test += "-Xmx4G",
  concurrentRestrictions in Global += Tags.limit(Tags.Test, 1),
  updateOptions := updateOptions.value.withCachedResolution(true)
)

val assemblySettings = Seq(
  assemblyJarName in assembly := "assembly.jar"
)

val SparkVersion = "2.4.7"

val v = new {
  val scalatest = "3.0.4"
  val scalamock = "3.6.0"
  val breeze    = "0.13.2" // Should be compatible with the spark version
}

val baseDependencies = Seq(
  "org.scalatest" %% "scalatest" % v.scalatest % "test",
  "org.scalamock" %% "scalamock-scalatest-support" % v.scalamock % "test"
)

val mainDependencies = baseDependencies ++ Seq(
  "org.scalanlp" %% "breeze" % v.breeze
)

val sparkDependencies = baseDependencies ++ Seq(
  "org.apache.spark" %% "spark-sql" % SparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % SparkVersion % "provided"
)


// Modules
lazy val hpopt = project
  .settings(baseSettings: _*)
  .settings(assemblySettings: _*)
  .settings(
    libraryDependencies ++= mainDependencies,
    name := "hpopt"
  )

lazy val hpopt_sparkutil = project
  .settings(baseSettings: _*)
  .settings(assemblySettings: _*)
  .settings(
    libraryDependencies ++= sparkDependencies,
    name:= "hpopt_sparkutil"
  ).dependsOn(hpopt)
