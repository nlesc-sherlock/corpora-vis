from flask import Flask, jsonify, send_from_directory
from flask_restful import reqparse
from flask.ext.cors import CORS

import numpy as np

app = Flask(__name__)
CORS(app)

## Loading of data
# Extract data from avro and csv files
from pyspark.sql.functions import size, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import array_contains
# # Load metadata avro
reader = sqlContext.read.format('com.databricks.spark.avro')
meta = reader.load('data/spark_metadata.avro')
# # Loading topic distributions
topdisFile = 'data/enron_small_topic_distributions.tuples'
csvLoader = sqlContext.read.format('com.databricks.spark.csv')
topdis = csvLoader.options(delimiter=',',header='false', inferschema='true').load(topdisFile)
strip_first_col = udf(lambda row: int(row[1:]), IntegerType())
topdis = topdis.withColumn('C0',strip_first_col(topdis['C0']))
# # Load dictionary CSV
dicFile = 'enron_small_dic.csv'
csvLoader = sqlContext.read.format('com.databricks.spark.csv')
dic = csvLoader.options(delimiter='\t', header='false', inferschema='true').load(dicFile)
dic = dic.select(dic['C0'].alias('id'), dic['C1'].alias('word'), dic['C2'].alias('count'))
# # Load clustertopics CSV
clutoFile = 'enron_small_clustertopics.csv'
csvLoader = sqlContext.read.format('com.databricks.spark.csv')
cluto = csvLoader.options(delimiter=',', header='false', inferschema='true').load(clutoFile)
# # Load topicswords CSV
towoFile = 'enron_small_lda_transposed.csv'
csvLoader = sqlContext.read.format('com.databricks.spark.csv')
towo = csvLoader.options(delimiter=',', header='false', inferschema='true').load(towoFile)
# # Merge topdis which has document id and with metadata, based on document id
metasmall = meta.select('id','date')
newdf = topdis.join(metasmall, metasmall.id == topdis.C0,'inner')



# Topic x words matrix
_data = np.loadtxt('enron_small_lda_transposed.csv', delimiter=',')

# Cluster x topics matrix
_data2 = np.loadtxt('enron_small_clustertopics.csv', delimiter=',')

# Load word => id dictionary
_idToWord = {}
with open('enron_small_dic.csv', 'r') as fin:
    for line in fin:
        line = line.strip()
        cols = line.split('\t')
        num = int(cols[0])
        word = cols[1]
        _idToWord[num] = word

## Building JSON

def getWordsInTopic(n):
    pct = 0.20

    wordIds = _data[:,n]
    wordIds /= wordIds.sum()
    sortIdx = wordIds.argsort()
    sortIdx = sortIdx[::-1]
    topWordIds = sortIdx[wordIds[sortIdx].cumsum()<pct]

    wordArray = [ {"name": _idToWord[wId], "size": wordIds[wId]} for wId in topWordIds ]
    return wordArray

def getTopicsInCluster(n): #newly added
    pct = 0.40
    topicIds = _data2[:,n]
    topicIds /= topicIds.sum()
    sortIdx = topicIds.argsort()
    sortIdx = sortIdx[::-1]
    clustertopicIds = sortIdx[topicIds[sortIdx].cumsum()<pct]

    topicArray = [ getTopic(cId) for cId in clustertopicIds ]
    return {
        'name'    : 'Cluster %d'%n,
        'children': topicArray
    }

def getTopic(n):
    topic = {
        'name'    : 'Topic %d'%n,
        'children': getWordsInTopic(n)
    }
    return topic

def getCluster(n): #newly added
    cluster = {
        'name'    : 'Cluster %d'%n,
        'children': getTopicsInCluster(n)
    }
    return cluster

@app.route('/visualisation/sunburst')
def visualisationData():
    clusters = [ getTopicsInCluster(n) for n in range(_data2.shape[1]) ]
    data = {
        'name'    : 'ENRON',
        'children': clusters
    }
    return jsonify(data=data)

## Serve static files

@app.route('/')
def root():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def send_content(path):
    return send_from_directory('static', path)

## Start app

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',threaded=True)
