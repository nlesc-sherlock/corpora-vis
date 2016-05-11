from flask import Flask, jsonify
from flask_restful import reqparse
from flask.ext.cors import CORS

import numpy as np

app = Flask(__name__)
CORS(app)

# Topic x words matrix
_data = np.loadtxt('enron_small_lda_transposed.csv', delimiter=',')

# Cluster x topics matrix
_data2 = np.loadtxt('enron_small_clustertopics.csv', delimiter=',')


_idToWord = {}
with open('enron_small_dic.csv', 'r') as fin:
    for line in fin:
        line = line.strip()
        cols = line.split('\t')
        num = int(cols[0])
        word = cols[1]
        _idToWord[num] = word

def getWordsInTopic(n):
    pct = 0.20

    wordIds = _data[:,n]
    wordIds /= wordIds.sum()
    sortIdx = wordIds.argsort()
    sortIdx = sortIdx[::-1]
    topWordIds = sortIdx[wordIds[sortIdx].cumsum()<pct]

    wordArray = [ {"name": _idToWord[wId], "size": 1} for wId in topWordIds ]
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

# def getTopicsInCluster(nCluster):
#     topics = [ getTopic(n) for n in range(_data.shape[1]) ]
#     return {
#         'name': 'Cluster %d'%nCluster,
#         'children': topics
#     }

@app.route('/visualisation/sunburst')
def visualisationData():
    clusters = [ getTopicsInCluster(n) for n in range(_data2.shape[1]) ]
    data = {
        'name'    : 'ENRON',
        'children': clusters
    }
    return jsonify(data=data)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',threaded=True)
