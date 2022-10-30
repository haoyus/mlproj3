import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn import metrics


RICE_PATH = './Rice_Cammeo_Osmancik.csv'
BEAN_PATH = './Dry_Bean_Dataset.csv'

rice_cls2int = {'Cammeo':0, 'Osmancik':1}
bean_cls2int = {'SEKER':0, 'BARBUNYA':1, 'BOMBAY':2, 'CALI':3, 'HOROZ':4, 'SIRA':5, 'DERMASON':6}

def load_data(path):
    if 'Cammeo' in path:
        classes = ['Cammeo', 'Osmancik']
        cls2int = rice_cls2int
    elif 'Bean' in path:
        classes = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']
        cls2int = bean_cls2int
    else:
        raise Exception('Please name your dataset path following the RICE_PATH and RICE_MSC_PATH')

    loaded = csv.reader(open(path))
    n = 0
    attributes = []
    contents = []
    for row in loaded:
        if n==0:
            attributes = row
        else:
            for i in range(len(row)-1):
                if row[i]=='':
                    row[i]=0
                row[i] = float(row[i])
            row[-1] = cls2int[row[-1]]
            contents.append(row)
        n += 1
    
    attr_inds = {attr:ind for ind, attr in enumerate(attributes)}
    ind_attrs = {ind:attr for ind, attr in enumerate(attributes)}
    # print(len(contents))
    # print(contents[1])
    contents = np.array(contents)
    x = contents[:,:-1]
    y = contents[:, -1]

    return x, y, attributes[:-1], attr_inds, ind_attrs, classes

def do_split(X, Y):
    """do 80/20 split on list of samples"""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test

def prepare_data(path):
    X, Y, attributes, attr_inds, ind_attrs, classes = load_data(path)

    print(f'Dataset of {path}:')
    print(f'Total number of samples: {len(X)}')
    print(f'Attributes: {attributes}, total {len(attributes)} attributes')
    print(f'Classes: {classes}')
    C = int(np.max(Y)) + 1
    for i in range(C):
        this_count = np.sum(Y==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(Y),2)} of total {len(Y)}')
    # print('Basic statistics: total num of samples - ' , len(Y), ', class balance: ', round(1-np.sum(Y)/len(Y),3), ':', round(np.sum(Y)/len(Y),3))

    return X, Y, attributes, attr_inds, ind_attrs, classes


def test_kmeans(data, labels, k, iter_limit):
    score_list = []
    time_list = []
    for i in range(iter_limit):
        max_iter = i+1
        kmeans = KMeans(k, max_iter=max_iter)

        t0 = time.time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        t1 = time.time()
        
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]

        scores = [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        score_list.append(scores)
        time_list.append(t1 - t0)

