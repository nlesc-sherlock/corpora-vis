# corpora-vis
Visualising results from topic models

# Run server

Run server.py to using spark

```
$SPARK_HOME/bin/spark-submit --packages=com.databricks:spark-avro_2.10:2.0.1 server.py
```
# Run notebook

IPYTHON_OPTS=notebook $SPARK_HOME/bin/pyspark --packages=com.databricks:spark-avro_2.10:2.0.1,com.databricks:spark-csv_2.10:1.4.0

# Install spark:

http://blog.prabeeshk.com/blog/2014/10/31/install-apache-spark-on-ubuntu-14-dot-04/
