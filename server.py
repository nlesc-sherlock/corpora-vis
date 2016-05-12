from flask import Flask, jsonify, send_from_directory
from flask_restful import reqparse
from flask.ext.cors import CORS
import numpy as np
# Extract data from avro and csv files
from pyspark.sql.functions import size, udf, unix_timestamp, max, min
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import array_contains
from pyspark.sql import SQLContext
from pyspark import SparkContext


app = Flask(__name__)
CORS(app)

@app.before_first_request
def do_something_only_once():
    # the command I use to run this script:
    #~/spark-1.6.1/bin/spark-submit --packages=com.databricks:spark-avro_2.10:2.0.1,com.databricks:spark-csv_2.10:1.4.0 server.py
    global topdis, meta, dic, towo, cluto, doctopdat, maxdate, mindate
    ## Loading of data
    sc = SparkContext(appName='Simple App') #"local"
    sqlContext = SQLContext(sc)
    # Load metadata avro
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
    # # # Load clustertopics CSV
    # clutoFile = 'enron_small_clustertopics.csv'
    # csvLoader = sqlContext.read.format('com.databricks.spark.csv')
    # cluto = csvLoader.options(delimiter=',', header='false', inferschema='true').load(clutoFile)
    # # # Load topicswords CSV
    # towoFile = 'enron_small_lda_transposed.csv'
    # csvLoader = sqlContext.read.format('com.databricks.spark.csv')
    # towo = csvLoader.options(delimiter=',', header='false', inferschema='true').load(towoFile)
    # # Merge topdis which has document id and with metadata, based on document id
    metasmall = meta.select('id',unix_timestamp(meta['date'],"yyyy-MM-dd'T'HH:mm:ssX").alias("timestamp"))
    doctopdat = topdis.join(metasmall, metasmall.id == topdis.C0,'inner')
    maxdate = doctopdat.select(max('timestamp').alias('maxtimestamp')).collect()[0]['maxtimestamp']
    mindate = doctopdat.select(min('timestamp').alias('mintimestamp')).collect()[0]['mintimestamp']

# Topic x words matrix
towo = np.loadtxt('enron_small_lda_transposed.csv', delimiter=',')
# Cluster x topics matrix
cluto = np.loadtxt('enron_small_clustertopics.csv', delimiter=',')

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
    wordIds = towo[:,n]
    wordIds /= wordIds.sum()
    sortIdx = wordIds.argsort()
    sortIdx = sortIdx[::-1]
    topWordIds = sortIdx[wordIds[sortIdx].cumsum()<pct]
    wordArray = [ {"name": _idToWord[wId], "size": wordIds[wId]} for wId in topWordIds ]
    return wordArray

def getTopicsInCluster(n):
    pct = 0.40
    topicIds = cluto[:,n]
    topicIds /= topicIds.sum()
    sortIdx = topicIds.argsort()
    sortIdx = sortIdx[::-1]
    clustertopicIds = sortIdx[topicIds[sortIdx].cumsum()<pct]
    topicArray = [ getTopic(cId) for cId in clustertopicIds ]
    return {
        'name'    : 'Cluster %d'%n,
        'children': topicArray
    }

def getDocTopDat(n): #newly added
    pct = 0.40
    docIds = doctopdat[:,n]
    docIds /= doc.sum()
    sortIdx = docIds.argsort()
    sortIdx = sortIdx[::-1]
    doctopicIds = sortIdx[docIds[sortIdx].cumsum()<pct]
    topicArray = [ getTopic(cId) for cId in doctopicIds ]
    return {
        'name'    : 'Document %d'%n,
        'children': topicArray
    }

def getTopic(n):
    topic = {
        'name'    : 'Topic %d'%n,
        'children': getWordsInTopic(n)
    }
    return topic

def getCluster(n):
    cluster = {
        'name'    : 'Cluster %d'%n,
        'children': getTopicsInCluster(n)
    }
    return cluster
#
# @app.route('/visualisation/sunburst')
# def visualisationData():
#     clusters = [ getTopicsInCluster(n) for n in range(cluto.shape[1]) ]
#     data = {
#         'name'    : 'ENRON',
#         'children': clusters
#     }
#     return jsonify(data=data)

@app.route('/visualisation/daniela')
def visulisationData():
    request.arg.get('nbins')
    binborders =map(int,list(np.linspace(start=mindate,stop=maxdate,num=nbins,endpoint=False)))
    def f(x):
        for i in range(len(binborders)):
            if x < binborders[i]:
                return binborders[i-1]
        return binborders[-1]
    stato = doctopdat.withColumn('bin',udf(f,LongType())(doctopdat['timestamp']))
    stato2 = stato.groupBy('bin').avg()
    info = [getDocTopDat(n) for n in range(len(stato2.columns)) ]
    data = {
        'name'    : 'ENRON',
        'children': info
    }
    return jsonify(data=data)


# Serve static files
@app.route('/')
def root():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def send_content(path):
    return send_from_directory('static', path)

## Start app
if __name__ == '__main__':
    pass
    app.debug = True
    app.run(host='0.0.0.0',threaded=True)
    #http://localhost:5000/visualisation/sunburst
