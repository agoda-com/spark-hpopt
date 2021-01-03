ARG BASE_CONTAINER=jupyter/all-spark-notebook
FROM $BASE_CONTAINER

COPY ./hpopt/target/scala-2.12/assembly.jar /home/jovyan/jars/assembly.jar
