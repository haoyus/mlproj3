import numpy as np
from matplotlib import pyplot as plt
import sys
import csv
import time
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import train_test_split

from learners import study_nn


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

def get_split(X, Y, ratio):
    _, _, x, y = train_test_split(X, Y, test_size=ratio)
    return x, y

def prepare_data(path):
    X, Y, attributes, attr_inds, ind_attrs, classes = load_data(path)
    print('------------------------------------')
    print(f'Dataset of {path}:')
    print(f'Total number of samples: {len(X)}')
    print(f'Attributes: {attributes}, total {len(attributes)} attributes')
    # print(f'Classes: {classes}')
    # C = int(np.max(Y)) + 1
    # for i in range(C):
    #     this_count = np.sum(Y==i)
    #     print(f'  class {i} has {this_count} samples, takes {round(this_count/len(Y),2)} of total {len(Y)}')
    # print('Basic statistics: total num of samples - ' , len(Y), ', class balance: ', round(1-np.sum(Y)/len(Y),3), ':', round(np.sum(Y)/len(Y),3))

    return X, Y, attributes, attr_inds, ind_attrs, classes

def plot_scores(score_list, max_iters, dataset_name, algo_name):
    fig, ax = plt.subplots(1,2, figsize=(10,8))
    fig.suptitle(f'{algo_name}_performance_on_{dataset_name}')
    ylim_max = 1.0
    ax[0].plot(max_iters, score_list[:,0], color='b', label='homogeneity_score')
    ax[0].set_ylim(0, ylim_max)
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('max iterations')
    ax[0].set_ylabel('score')

    ax[1].plot(max_iters, score_list[:,1], color='b', label='completeness_score')
    ax[1].set_ylim(0, ylim_max)
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('max iterations')
    ax[1].set_ylabel('score')

    # ax[1,0].plot(max_iters, score_list[:,3], color='b', label='adjusted_rand_score')
    # ax[1,0].set_ylim(0, ylim_max)
    # ax[1,0].grid()
    # ax[1,0].legend()
    # ax[1,0].set_xlabel('max iterations')
    # ax[1,0].set_ylabel('score')

    # ax[1,1].plot(max_iters, score_list[:,4], color='b', label='adjusted_mutual_info_score')
    # ax[1,1].set_ylim(0, ylim_max)
    # ax[1,1].grid()
    # ax[1,1].legend()
    # ax[1,1].set_xlabel('max iterations')
    # ax[1,1].set_ylabel('score')
    fig.tight_layout()
    plt.savefig(f'{algo_name}_scores_{dataset_name}.png')
    plt.clf()

def plot_fittime(time_list, max_iters, dataset_name, algo_name):
    fig, ax = plt.subplots(figsize=(5,4))
    fig.suptitle(f'{algo_name}_fittime_on_{dataset_name}')
    ax.plot(max_iters, time_list, color='b', label='fit time')
    ax.grid()
    ax.legend()
    ax.set_xlabel('max iterations')
    ax.set_ylabel('time (sec)')
    plt.savefig(f'{algo_name}_fittime_{dataset_name}.png')
    plt.clf()

def search_k_wrapper(algo_name, dataset_name, dataset_path, max_k=20):
    data, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(dataset_path)
    scaler = StandardScaler()
    data= scaler.fit_transform(data)
    search_k(algo_name, dataset_name, data, max_k)

def search_k(algo_name, dataset_name, data, max_k=20):
    score_list=[]
    inertias = []
    k_list = range(2, max_k+1, 1)
    for k in k_list:
        k = k+1
        if algo_name == 'kmeans':
            algo = KMeans(k, n_init=10, random_state=0)
            algo.fit(data)
            sil_score = metrics.silhouette_score(
                data, algo.labels_, metric='euclidean',sample_size=None, random_state=0
            )
            inertias.append(algo.inertia_)
        elif algo_name == 'em':
            algo = EM(n_components=k, n_init=10, random_state=0)
            algo.fit(data)
            pred_labels = algo.predict(data)
            sil_score = metrics.silhouette_score(
                data, pred_labels, metric='euclidean',sample_size=None, random_state=0
            )
        else:
            raise Exception('wrong algo name string')
        
        score_list.append(sil_score)
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(k_list, score_list, color='b', label='silhouette score')
    ax.grid()
    ax.legend()
    ax.set_xlim(0,max_k)
    plt.xticks(range(0,max_k+1))
    ax.set_xlabel('num of clusters')
    ax.set_ylabel('score')
    plt.savefig(f'{algo_name}_search_k_sil_{dataset_name}.png')
    plt.clf()
    if algo_name=='kmeans':
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(k_list, inertias, color='b', label='inertia')
        ax.grid()
        ax.legend()
        ax.set_xlim(0,max_k)
        plt.xticks(range(0,max_k+1))
        ax.set_xlabel('num of clusters')
        ax.set_ylabel('inertia')
        plt.savefig(f'{algo_name}_search_k_inertia_{dataset_name}.png')
        plt.clf()
    return score_list, k_list

def test_kmeans(data, labels, fea_indices, k, iter_limit):
    print('Testing K Means clustering...')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if fea_indices is not None:
        data = data[:, fea_indices]
    print(' on {} samples, selected {} features'.format(data.shape[0], data.shape[1]))
    score_list = []
    time_list = []
    max_iters = []
    sil_scores = []
    for i in range(1, iter_limit+1):
        max_iter = i
        kmeans = KMeans(n_clusters=k, max_iter=max_iter, init='random', n_init=10)

        t0 = time.time()
        kmeans.fit(data)
        # print(estimator[-1].labels_)
        # return
        t1 = time.time()
        
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            # metrics.v_measure_score,
            # metrics.adjusted_rand_score,
            # metrics.adjusted_mutual_info_score,
        ]
        pred_labels = kmeans.predict(data)
        sil_score = metrics.silhouette_score(
            data, pred_labels, metric='euclidean',sample_size=None, random_state=0
        )

        scores = [m(labels, pred_labels) for m in clustering_metrics]

        score_list.append(scores)
        time_list.append(t1 - t0)
        max_iters.append(max_iter)
        sil_scores.append(sil_score)

    score_list = np.array(score_list)
    time_list = np.array(time_list)
    sil_scores = np.array(sil_scores)
    return score_list, time_list, max_iters, sil_scores

def test_em(data, labels, fea_indices, k, iter_limit):
    print('Testing EM clustering...')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if fea_indices is not None:
        data = data[:, fea_indices]
    print(' on {} samples, selected {} features'.format(data.shape[0], data.shape[1]))
    score_list = []
    time_list = []
    max_iters = []
    sil_scores = []
    for i in range(1, iter_limit+1):
        max_iter = i
        em = EM(n_components=k, n_init=10, max_iter=max_iter)

        t0 = time.time()
        em.fit(data)
        # print(estimator[-1].labels_)
        # return
        t1 = time.time()
        
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            # metrics.v_measure_score,
            # metrics.adjusted_rand_score,
            # metrics.adjusted_mutual_info_score,
        ]

        pred_labels = em.predict(data)
        scores = [m(labels, pred_labels) for m in clustering_metrics]
        sil_score = metrics.silhouette_score(
            data, pred_labels, metric='euclidean',sample_size=None, random_state=0
        )

        score_list.append(scores)
        time_list.append(t1 - t0)
        max_iters.append(max_iter)
        sil_scores.append(sil_score)

    score_list = np.array(score_list)
    time_list = np.array(time_list)
    sil_scores = np.array(sil_scores)
    return score_list, time_list, max_iters, sil_scores

def test_dataset(path, name, num_clusters, fea_indices):
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(path)

    score_list, time_list, max_iters, _ = test_kmeans(X,Y, fea_indices, num_clusters, 100)
    score_list, time_list, max_iters = score_list[1:], time_list[1:], max_iters[1:]
    plot_scores(score_list, max_iters, name, 'kmeans')
    plot_fittime(time_list, max_iters, name, 'kmeans')

    score_list, time_list, max_iters, _ = test_em(X,Y, fea_indices, num_clusters, 100)
    score_list, time_list, max_iters = score_list[1:], time_list[1:], max_iters[1:]
    plot_scores(score_list, max_iters, name, 'em')
    plot_fittime(time_list, max_iters, name, 'em')

def run_dimRed_wrapper(path, dataset_name):
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_pca = run_PCA(X, dataset_name)
    X_ica = run_ICA(X, dataset_name)
    X_rpj = run_RPJ(X, dataset_name)
    X_svd = run_SVD(X, dataset_name)

def run_PCA(X, dataset_name, double_dim=False):
    pca = PCA(n_components=None)
    pca.fit(X)
    # print(pca.components_)
    np.set_printoptions(suppress=True)
    print('PCA Eigen Value ratios: ',np.array(pca.explained_variance_ratio_).round(6))
    # print(pca.singular_values_)
    X_trans = pca.transform(X)
    # print('samples of trans ',X_trans[0:3])
    if dataset_name=='rice':
        if double_dim:
            X_trans = X_trans[:, 0:6]
        else:
            X_trans = X_trans[:, 0:3]
    elif dataset_name=='bean':
        if double_dim:
            X_trans = X_trans[:, 0:4]
        else:
            X_trans = X_trans[:, 0:2]
    print('PCA selected shape', X_trans.shape)
    return X_trans

def run_ICA(X, dataset_name,  double_dim=False):
    ica = FastICA()
    ica.fit(X, dataset_name)
    # print(ica.mean_)
    # print(ica.whitening_)
    X_trans = ica.transform(X)
    # print(X_trans[0:3])
    kurt = kurtosis(X_trans)
    if dataset_name=='rice':
        if not double_dim:
            X_trans = X_trans[:, [0,1,4]]
        else:
            X_trans = X_trans[:, [0,1,3,4,5,6]]
    elif dataset_name=='bean':
        if not double_dim:
            X_trans = X_trans[:, [8,14]]
        else:
            X_trans = X_trans[:, [8,9,12,14]]
    print('ICA selected shape', X_trans.shape)
    return X_trans

def kurtosis(x):
    n = np.shape(x)[0]
    mean = np.mean(x,axis=0)
    var = np.sum((x-mean)**2, axis=0) / float(n)
    kurt = np.sum((x-mean)**4, axis=0)/float(n)
    kurt = kurt / (var**2) - 3
    print('Kurtosis: ',np.array(kurt).round(3))
    return kurt

def run_RPJ(X, dataset_name, double_dim=False):
    if double_dim:
        nr = 2
    else:
        nr = 1
    if dataset_name=='rice':
        rpj = GaussianRandomProjection(n_components=3*nr)
    elif dataset_name=='bean':
        rpj = GaussianRandomProjection(n_components=2*nr)
    # rpj = SparseRandomProjection()
    X_trans = rpj.fit_transform(X)
    X_reproj= rpj.inverse_transform(X_trans)
    diff = X - X_reproj
    diff = np.sum(np.abs(diff), axis=0) / X.shape[0]
    print(diff.round(4))
    # eps=0.5
    # D = 9*1/(eps**2 - 2*eps**3/3) * math.log(X.shape[0]) + 1
    # print(D)
    # print(X_trans.shape)
    return X_trans

def run_SVD(X, dataset_name, double_dim=False):
    n_ratio = 2 if double_dim else 1
    if dataset_name=='rice':
        svd = TruncatedSVD(n_components=3*n_ratio)
    elif dataset_name=='bean':
        svd = TruncatedSVD(n_components=2*n_ratio)
    X_trans = svd.fit_transform(X)
    # print(X_trans.shape)
    return X_trans

def test_cluster_with_reduced_once(path, dataset_name, algo_name):
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_pca = run_PCA(X, dataset_name, False)
    X_ica = run_ICA(X, dataset_name, False)
    X_rpj = run_RPJ(X, dataset_name, False)
    X_svd = run_SVD(X, dataset_name, False)
    if dataset_name=='rice':
        K = 2
    elif dataset_name=='bean':
        K = 4
    else:
        raise Exception('wrong dataset name')

    if algo_name=='kmeans':
        kmeans_pca = KMeans(n_clusters=K, init='random', n_init=10)
        kmeans_pca.fit(X_pca)
        y_pred_pca = kmeans_pca.predict(X_pca)
        kmeans_ica = KMeans(n_clusters=K, init='random', n_init=10)
        kmeans_ica.fit(X_ica)
        y_pred_ica = kmeans_ica.predict(X_ica)
    elif algo_name=='em':
        kmeans_pca = EM(n_components=K, n_init=10)
        kmeans_pca.fit(X_pca)
        y_pred_pca = kmeans_pca.predict(X_pca)
        kmeans_ica = EM(n_components=K, n_init=10)
        kmeans_ica.fit(X_ica)
        y_pred_ica = kmeans_ica.predict(X_ica)

    if dataset_name=='bean':
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle(f'{algo_name}_cluster_on_{dataset_name} after Dim Reduction')
        ax[0].scatter(X_pca[:,0], X_pca[:,1], c=y_pred_pca)
        ax[0].grid()
        ax[0].set_xlabel('PCA 0th dim')
        ax[0].set_ylabel('PCA 1st dim')
    # 
        ax[1].scatter(X_ica[:,0], X_ica[:,1], c=y_pred_ica)
        ax[1].grid()
        ax[1].set_xlabel('ICA 0th dim')
        ax[1].set_ylabel('ICA 1st dim')
    # 
        fig.tight_layout()
        plt.savefig(f'{algo_name}_cluster_on_{dataset_name} after Dim Reduction.png')
        plt.clf()
    elif dataset_name == 'rice':
        fig, ax = plt.subplots(2,3, figsize=(9,6))
        fig.suptitle(f'{algo_name}_cluster_on_{dataset_name} after Dim Reduction')
        ax[0,0].scatter(X_pca[:,0], X_pca[:,1], c=y_pred_pca)
        ax[0,0].grid()
        ax[0,0].set_xlabel('PCA 0th dim')
        ax[0,0].set_ylabel('PCA 1st dim')
        ax[0,1].scatter(X_pca[:,1], X_pca[:,2], c=y_pred_pca)
        ax[0,1].grid()
        ax[0,1].set_xlabel('PCA 1th dim')
        ax[0,1].set_ylabel('PCA 2nd dim')
        ax[0,2].scatter(X_pca[:,0], X_pca[:,2], c=y_pred_pca)
        ax[0,2].grid()
        ax[0,2].set_xlabel('PCA 0th dim')
        ax[0,2].set_ylabel('PCA 2nd dim')

        ax[1,0].scatter(X_ica[:,0], X_ica[:,1], c=y_pred_ica)
        ax[1,0].grid()
        ax[1,0].set_xlabel('ICA 0th dim')
        ax[1,0].set_ylabel('ICA 1st dim')
        ax[1,1].scatter(X_ica[:,1], X_ica[:,2], c=y_pred_ica)
        ax[1,1].grid()
        ax[1,1].set_xlabel('ICA 1th dim')
        ax[1,1].set_ylabel('ICA 2nd dim')
        ax[1,2].scatter(X_ica[:,0], X_ica[:,2], c=y_pred_ica)
        ax[1,2].grid()
        ax[1,2].set_xlabel('ICA 0th dim')
        ax[1,2].set_ylabel('ICA 2nd dim')

        fig.tight_layout()
        plt.savefig(f'{algo_name}_cluster_on_{dataset_name} after Dim Reduction.png')
        plt.clf()
    

def test_cluster_with_reduced(path, dataset_name, algo_name, double_dim=False, is_search_k=False):
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_pca = run_PCA(X, dataset_name, double_dim)
    X_ica = run_ICA(X, dataset_name, double_dim)
    X_rpj = run_RPJ(X, dataset_name, double_dim)
    X_svd = run_SVD(X, dataset_name, double_dim)

    if is_search_k:
        search_k('kmeans', dataset_name+'_pca', X_pca, 20)
        return

    if dataset_name=='rice':
        K = 2
    elif dataset_name=='bean':
        K = 4
    else:
        raise Exception('wrong dataset name')
    iter_limit = 20

    # Y = 1 - Y
    if algo_name == 'kmeans':
        scores_ori, times_ori, max_iters, sil_scores_ori = test_kmeans(X, Y, None, K, iter_limit)
        scores_pca, times_pca, max_iters, sil_scores_pca = test_kmeans(X_pca, Y, None, K, iter_limit)
        scores_ica, times_ica, max_iters, sil_scores_ica = test_kmeans(X_ica, Y, None, K, iter_limit)
        scores_rpj, times_rpj, max_iters, sil_scores_rpj = test_kmeans(X_rpj, Y, None, K, iter_limit)
        scores_svd, times_svd, max_iters, sil_scores_svd = test_kmeans(X_svd, Y, None, K, iter_limit)
    elif algo_name == 'em':
        scores_ori, times_ori, max_iters, sil_scores_ori = test_em(X, Y, None, K, iter_limit)
        scores_pca, times_pca, max_iters, sil_scores_pca = test_em(X_pca, Y, None, K, iter_limit)
        scores_ica, times_ica, max_iters, sil_scores_ica = test_em(X_ica, Y, None, K, iter_limit)
        scores_rpj, times_rpj, max_iters, sil_scores_rpj = test_em(X_rpj, Y, None, K, iter_limit)
        scores_svd, times_svd, max_iters, sil_scores_svd = test_em(X_svd, Y, None, K, iter_limit)

    fig, ax = plt.subplots(2,2, figsize=(10,8))
    fig.suptitle(f'{algo_name}_performance_on_{dataset_name} after Dim Reduction')
    ylim_max = 1.0
    ax[0,0].plot(max_iters, scores_pca[:,0], color='b', label='homogeneity_original')
    ax[0,0].plot(max_iters, scores_pca[:,0], color='k', label='homogeneity_pca')
    ax[0,0].plot(max_iters, scores_ica[:,0], color='g', label='homogeneity_ica')
    ax[0,0].plot(max_iters, scores_rpj[:,0], color='r', label='homogeneity_rpj')
    ax[0,0].plot(max_iters, scores_svd[:,0], 'y.',      label='homogeneity_svd')
    ax[0,0].grid()
    ax[0,0].legend()
    ax[0,0].set_xlabel('max iterations')
    ax[0,0].set_ylabel('score')

    ax[0,1].plot(max_iters, scores_ori[:,1], color='b', label='completeness_original')
    ax[0,1].plot(max_iters, scores_pca[:,1], color='k', label='completeness_pca')
    ax[0,1].plot(max_iters, scores_ica[:,1], color='g', label='completeness_ica')
    ax[0,1].plot(max_iters, scores_rpj[:,1], color='r', label='completeness_rpj')
    ax[0,1].plot(max_iters, scores_svd[:,1], 'y.',      label='completeness_svd')
    ax[0,1].grid()
    ax[0,1].legend()
    ax[0,1].set_xlabel('max iterations')
    ax[0,1].set_ylabel('score')

    ax[1,0].plot(max_iters, sil_scores_ori, color='b', label='silhouette original')
    ax[1,0].plot(max_iters, sil_scores_pca, color='k', label='silhouette pca')
    ax[1,0].plot(max_iters, sil_scores_ica, color='g', label='silhouette ica')
    ax[1,0].plot(max_iters, sil_scores_rpj, color='r', label='silhouette rpj')
    ax[1,0].plot(max_iters, sil_scores_svd, 'y.',      label='silhouette svd')
    ax[1,0].grid()
    ax[1,0].legend()
    ax[1,0].set_xlabel('max iterations')
    ax[1,0].set_ylabel('silhouette')

    ax[1,1].plot(max_iters, times_ori, color='b', label='fit time original')
    ax[1,1].plot(max_iters, times_pca, color='k', label='fit time pca')
    ax[1,1].plot(max_iters, times_ica, color='g', label='fit time ica')
    ax[1,1].plot(max_iters, times_rpj, color='r', label='fit time rpj')
    ax[1,1].plot(max_iters, times_svd, 'y.',      label='fit time svd')
    ax[1,1].grid()
    ax[1,1].legend()
    ax[1,1].set_xlabel('max iterations')
    ax[1,1].set_ylabel('time (sec)')
   
    fig.tight_layout()
    plt.savefig(f'{algo_name}_scores_{dataset_name}_dim_reduce.png')
    plt.clf()

def test_datasize_with_reduced(path, dataset_name, algo_name, double_dim=False, is_search_k=False):
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_pca = run_PCA(X, dataset_name, double_dim)
    X_ica = run_ICA(X, dataset_name, double_dim)
    X_rpj = run_RPJ(X, dataset_name, double_dim)
    X_svd = run_SVD(X, dataset_name, double_dim)

    if is_search_k:
        search_k('kmeans', dataset_name+'_pca', X_pca, 20)
        return

    if dataset_name=='rice':
        K = 2
    elif dataset_name=='bean':
        K = 4
    else:
        raise Exception('wrong dataset name')
    iter_limit = 100

    # Y = 1 - Y
    if algo_name == 'kmeans':
        scores_ori, times_ori, max_iters, sil_scores_ori = test_kmeans(X, Y, None, K, iter_limit)
        scores_pca, times_pca, max_iters, sil_scores_pca = test_kmeans(X_pca, Y, None, K, iter_limit)
        scores_ica, times_ica, max_iters, sil_scores_ica = test_kmeans(X_ica, Y, None, K, iter_limit)
        scores_rpj, times_rpj, max_iters, sil_scores_rpj = test_kmeans(X_rpj, Y, None, K, iter_limit)
        scores_svd, times_svd, max_iters, sil_scores_svd = test_kmeans(X_svd, Y, None, K, iter_limit)
    elif algo_name == 'em':
        scores_ori, times_ori, max_iters, sil_scores_ori = test_em(X, Y, None, K, iter_limit)
        scores_pca, times_pca, max_iters, sil_scores_pca = test_em(X_pca, Y, None, K, iter_limit)
        scores_ica, times_ica, max_iters, sil_scores_ica = test_em(X_ica, Y, None, K, iter_limit)
        scores_rpj, times_rpj, max_iters, sil_scores_rpj = test_em(X_rpj, Y, None, K, iter_limit)
        scores_svd, times_svd, max_iters, sil_scores_svd = test_em(X_svd, Y, None, K, iter_limit)

    fig, ax = plt.subplots(2,2, figsize=(10,8))
    fig.suptitle(f'{algo_name}_performance_on_{dataset_name} after Dim Reduction')
    ylim_max = 1.0
    ax[0,0].plot(max_iters, scores_pca[:,0], color='b', label='homogeneity_original')
    ax[0,0].plot(max_iters, scores_pca[:,0], color='k', label='homogeneity_pca')
    ax[0,0].plot(max_iters, scores_ica[:,0], color='g', label='homogeneity_ica')
    ax[0,0].plot(max_iters, scores_rpj[:,0], color='r', label='homogeneity_rpj')
    ax[0,0].plot(max_iters, scores_svd[:,0], 'y.',      label='homogeneity_svd')
    ax[0,0].grid()
    ax[0,0].legend()
    ax[0,0].set_xlabel('max iterations')
    ax[0,0].set_ylabel('score')

    ax[0,1].plot(max_iters, scores_ori[:,1], color='b', label='completeness_original')
    ax[0,1].plot(max_iters, scores_pca[:,1], color='k', label='completeness_pca')
    ax[0,1].plot(max_iters, scores_ica[:,1], color='g', label='completeness_ica')
    ax[0,1].plot(max_iters, scores_rpj[:,1], color='r', label='completeness_rpj')
    ax[0,1].plot(max_iters, scores_svd[:,1], 'y.',      label='completeness_svd')
    ax[0,1].grid()
    ax[0,1].legend()
    ax[0,1].set_xlabel('max iterations')
    ax[0,1].set_ylabel('score')

    ax[1,0].plot(max_iters, sil_scores_ori, color='b', label='silhouette original')
    ax[1,0].plot(max_iters, sil_scores_pca, color='k', label='silhouette pca')
    ax[1,0].plot(max_iters, sil_scores_ica, color='g', label='silhouette ica')
    ax[1,0].plot(max_iters, sil_scores_rpj, color='r', label='silhouette rpj')
    ax[1,0].plot(max_iters, sil_scores_svd, 'y.',      label='silhouette svd')
    ax[1,0].grid()
    ax[1,0].legend()
    ax[1,0].set_xlabel('max iterations')
    ax[1,0].set_ylabel('silhouette')

    ax[1,1].plot(max_iters, times_ori, color='b', label='fit time original')
    ax[1,1].plot(max_iters, times_pca, color='k', label='fit time pca')
    ax[1,1].plot(max_iters, times_ica, color='g', label='fit time ica')
    ax[1,1].plot(max_iters, times_rpj, color='r', label='fit time rpj')
    ax[1,1].plot(max_iters, times_svd, 'y.',      label='fit time svd')
    ax[1,1].grid()
    ax[1,1].legend()
    ax[1,1].set_xlabel('max iterations')
    ax[1,1].set_ylabel('time (sec)')
   
    fig.tight_layout()
    plt.savefig(f'{algo_name}_scores_{dataset_name}_dim_reduce.png')
    plt.clf()

def exp_4():
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(BEAN_PATH)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dataset_name = 'bean'
    X_pca = run_PCA(X, dataset_name)
    # X_ica = run_ICA(X, dataset_name)
    # X_rpj = run_RPJ(X, dataset_name)
    # X_svd = run_SVD(X, dataset_name)

    x_train, y_train, x_test, y_test = do_split(X, Y)

    study_nn(x_train, y_train, x_test, y_test, attributes, classes)

def exp_5():
    X, Y, attributes, attr_inds, ind_attrs, classes = prepare_data(BEAN_PATH)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dataset_name = 'bean'

    X_pca = run_PCA(X, dataset_name)
    em = EM(n_components=4, n_init=10)
    em.fit(X_pca)

    y_cluster = em.predict(X_pca)
    print(X_pca.shape, y_cluster.shape)
    X_new = np.concatenate((X_pca, y_cluster.reshape(-1,1)),axis=1)
    print('X new shape', X_new.shape)

    x_train, y_train, x_test, y_test = do_split(X_new, Y)

    study_nn(x_train, y_train, x_test, y_test, attributes, classes)



if __name__ == "__main__":
    # a = range(7)
    # aa = np.array([a,a])
    # print(aa)
    # aa = aa[:, [2,3,6]]
    # # print(aa)
    search_k_wrapper('kmeans', 'rice', RICE_PATH)
    search_k_wrapper('kmeans', 'bean', BEAN_PATH)
    search_k_wrapper('em', 'rice', RICE_PATH)
    search_k_wrapper('em', 'bean', BEAN_PATH)

    test_dataset(RICE_PATH, 'rice', 2, fea_indices=None)
    test_dataset(BEAN_PATH, 'bean', 4, fea_indices=None)

    run_dimRed_wrapper(RICE_PATH, 'rice')
    run_dimRed_wrapper(BEAN_PATH, 'bean')

    test_cluster_with_reduced(RICE_PATH, 'rice', 'kmeans', double_dim=False)
    test_cluster_with_reduced(BEAN_PATH, 'bean', 'kmeans', double_dim=False)

    test_cluster_with_reduced(RICE_PATH, 'rice', 'em')
    test_cluster_with_reduced(BEAN_PATH, 'bean', 'em')

    exp_4()
    exp_5()

    test_cluster_with_reduced_once(BEAN_PATH, 'bean', 'kmeans')
    test_cluster_with_reduced_once(RICE_PATH, 'rice', 'kmeans')
    test_cluster_with_reduced_once(BEAN_PATH, 'bean', 'em')
    test_cluster_with_reduced_once(RICE_PATH, 'rice', 'em')
