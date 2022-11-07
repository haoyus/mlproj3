import csv
import numpy as np
from sklearn import tree, neighbors, neural_network, svm, ensemble
import graphviz
import time
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

RICE_PATH = './Rice_Dataset/Rice_Cammeo_Osmancik.csv'
BEAN_PATH = './Dry_Bean_Dataset/Dry_Bean_Dataset.csv'

rice_cls2int = {'Cammeo':0, 'Osmancik':1}
rice_msc_cls2int = {'Arborio':0, 'Basmati':1, 'Ipsala':2, 'Jasmine':3, 'Karacadag':4}
bean_cls2int = {'SEKER':0, 'BARBUNYA':1, 'BOMBAY':2, 'CALI':3, 'HOROZ':4, 'SIRA':5, 'DERMASON':6}

def load_data(path):
    if 'Cammeo' in path:
        classes = ['Cammeo', 'Osmancik']
        cls2int = rice_cls2int
    elif 'MSC' in path:
        classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        cls2int = rice_msc_cls2int
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
    print(f'Attributes: {attributes}, total {len(attributes)} attributes')
    print(f'Classes: {classes}')
    C = int(np.max(Y)) + 1
    for i in range(C):
        this_count = np.sum(Y==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(Y),2)} of total {len(Y)}')
    # print('Basic statistics: total num of samples - ' , len(Y), ', class balance: ', round(1-np.sum(Y)/len(Y),3), ':', round(np.sum(Y)/len(Y),3))

    x_train, y_train, x_test, y_test = do_split(X, Y)
    print('After train test split, train vs test samples: ' ,len(x_train), len(x_test))
    print('Train set balance:')
    for i in range(C):
        this_count = np.sum(y_train==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(y_train),2)} of total {len(y_train)}')
    print('Test set balance:')
    for i in range(C):
        this_count = np.sum(y_test==i)
        print(f'  class {i} has {this_count} samples, takes {round(this_count/len(y_test),2)} of total {len(y_test)}')
    print('========================')
    return x_train, y_train, x_test, y_test, attributes, classes


def train_dtclf(x_train, y_train):
    dt_clf = tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(x_train, y_train)
    return dt_clf

def test_dtclf(dt_clf, predicate:str, x_train, y_train, x_test, y_test):
    train_pred = dt_clf.predict(x_train)
    
    pred = dt_clf.predict(x_test)
    correct = np.sum(pred == y_test)
    total = len(y_test)
    print(predicate)
    print('Train Accuracy: ', np.sum(train_pred==y_train) / len(y_train), dt_clf.score(x_train,y_train))
    print('Test Accuracy: ', correct/total, dt_clf.score(x_test, y_test))


def viz_dtclf(dt_clf, pdf_name:str, attributes, classes):
    dotdata = tree.export_graphviz(dt_clf, out_file=None, feature_names=attributes, class_names=classes,
        filled=True, rounded=True)
    graph = graphviz.Source(dotdata)
    graph.render(pdf_name)
    # tree.plot_tree(dt_clf, filled=True)
    # plt.savefig(pdf_name)

def try_pruning(x_trainsub, y_trainsub, x_val, y_val):
    # pruning
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(x_trainsub, y_trainsub)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-5], impurities[:-5], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig('pruning impurity vs alpha.png')
    plt.clf()

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_trainsub, y_trainsub)
        clfs.append(clf)
    # print(
    #     "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
    #         clfs[-1].tree_.node_count, ccp_alphas[-1]
    #     )
    # )
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlim(0, 0.005)
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[0].grid()
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlim(0, 0.005)
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    ax[1].grid()
    fig.tight_layout()
    plt.savefig('pruning path.png')
    plt.clf()

    train_scores = [clf.score(x_trainsub, y_trainsub) for clf in clfs]
    val_scores = [clf.score(x_val, y_val) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 0.005)
    ax.set_ylim(0.8, 1)
    ax.set_title("Accuracy vs alpha for training and validation sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
    ax.legend()
    ax.grid()
    plt.savefig('pruning accuracy.png')
    plt.clf()
def try_split(x_train,y_train, alpha):
    clf_best = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha, splitter='best')
    clf_rand = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha, splitter='random')
    scores_best = cross_validate(clf_best, x_train,y_train, cv=5, scoring='f1_weighted',return_train_score=False)
    scores_rand = cross_validate(clf_rand, x_train,y_train, cv=5, scoring='f1_weighted',return_train_score=False)
    print('Tune best val score', np.mean(scores_best['test_score']))
    print('Tune rand val score', np.mean(scores_rand['test_score']))

def study_dt(x_train, y_train, x_test, y_test, attributes, classes):
    x_trainsub, y_trainsub, x_val, y_val = do_split(x_train, y_train)

    # try pruning
    try_pruning(x_trainsub, y_trainsub, x_val, y_val)
    # try splitter
    try_split(x_train,y_train, alpha=0.002)

    # pruned_clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.0004)
    # pruned_clf.fit(x_trainsub, y_trainsub)
    # compare confusion matrix on val data
    # clf = tree.DecisionTreeClassifier(random_state=0)
    # clf.fit(x_trainsub, y_trainsub)
    # plot_confusion_matrix('DecisionTree',DATASET_NAME,'raw',clf, x_val,y_val,classes,'true')
    # plot_confusion_matrix('DecisionTree',DATASET_NAME,'pruned',pruned_clf, x_val,y_val,classes,'true')

    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.002, splitter='best')
    # print(clf.score(x_test, y_test))
    # scores = cross_val_score(clf, x_train, y_train, cv=5)
    # scores = cross_validate(clf, x_train,y_train, cv=5, scoring='accuracy',return_train_score=True)
    # train_sizes_abs, train_scores, test_scores, fit_times, score_times = do_learning_curve(
    #     clf, x_train, y_train, 5, 'accuracy'
    # )
    # print('Decision Tree train scores:', train_scores, test_scores, fit_times)
    plot_learning_curve('DecisionTree',DATASET_NAME,'pruned',clf,x_train,y_train, SCORING)

    final_test(clf, x_train, y_train, x_test, y_test, 'DecisionTree', DATASET_NAME, classes)
    # viz_dtclf(clf, 'dt_clf_msc_depth3', attributes, classes)

    # dtclf = train_dtclf(x_train, y_train)
    # test_dtclf(dtclf, 'Original training', x_train,y_train,x_test,y_test)
    # viz_dtclf(dtclf, 'original_dtclf')

    # print(clf.score(x_test, y_test))

######################################### KNN #############################################

def try_knn_k(x_trainsub, y_trainsub, x_val, y_val):
    metric_val, metric_train = [],[]
    klist = np.linspace(1,100,100).astype(int)
    for k in klist:
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_trainsub,y_trainsub)
        y_val_pred = clf.predict(x_val)
        y_train_pred = clf.predict(x_trainsub)

        metric_val.append(f1_score(y_val,y_val_pred,average='weighted'))
        metric_train.append(f1_score(y_trainsub, y_train_pred, average='weighted'))
    
    fig, ax = plt.subplots()
    ax.set_xlabel("size of K")
    ax.set_ylabel('weighted f1')
    ax.set_title("weighted f1 vs K size for training and validation")
    plt.plot(klist, metric_val, color='r', label='Val Weighted F1 Score')
    plt.plot(klist, metric_train, color = 'b', label='Train Weighted F1 Score')
    plt.plot([10,10], [0.85,1])
    ax.legend()
    ax.grid()
    # plt.show()
    plt.savefig(f'KNN_{DATASET_NAME}_valcurve_ksize.png')
    plt.clf()

def try_knn_dist(x_train,y_train, x_trainsub, y_trainsub, x_val, y_val, classes):
    clf_uniform = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
    clf_distance= neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')

    scores_uniform = cross_validate(clf_uniform, x_train,y_train, cv=5, scoring='f1_weighted',return_train_score=False)
    scores_distance = cross_validate(clf_distance, x_train,y_train, cv=5, scoring='f1_weighted',return_train_score=False)
    print('Tune uniform val score', np.mean(scores_uniform['test_score']))
    print('Tune distanc val score', np.mean(scores_distance['test_score']))

    clf_uniform.fit(x_trainsub, y_trainsub)
    clf_distance.fit(x_trainsub, y_trainsub)
    plot_confusion_matrix('KNN',DATASET_NAME,'uniform',clf_uniform, x_val,y_val,classes,'true')
    plot_confusion_matrix('KNN',DATASET_NAME,'distance',clf_distance, x_val,y_val,classes,'true')

def study_knn(x_train, y_train, x_test, y_test, attributes, classes):
    x_trainsub, y_trainsub, x_val, y_val = do_split(x_train, y_train)
    # # tune Hyper
    try_knn_k(x_trainsub, y_trainsub, x_val, y_val)
    try_knn_dist(x_train, y_train, x_trainsub, y_trainsub, x_val, y_val, classes)

    clf = neighbors.KNeighborsClassifier(
        n_neighbors=3,
        weights='uniform',#'distance',callable
        algorithm='auto', p=2)
    plot_learning_curve('K-NN',DATASET_NAME, 'n3', clf, x_train, y_train, SCORING)
    final_test(clf, x_train, y_train, x_test, y_test, 'kNN', DATASET_NAME, classes)

########################################### NN ##########################################

def grid_search(x_train, y_train):
    # hidden_units = [10, 20, 40, 80]
    hidden_units = [3]#, 5, 10, 80]
    learning_rates = [0.00005]#, 0.0001, 0.0002, 0.0005, 0.001]
    batch_sizes = [50]#,100,150,200]
    grid = {'hidden_layer_sizes': hidden_units, 'learning_rate_init': learning_rates, 'batch_size':batch_sizes}

    searcher = GridSearchCV(estimator = neural_network.MLPClassifier(
                                solver='adam',activation='logistic',random_state=0,
                                learning_rate='adaptive', max_iter=3000),
                            param_grid=grid, cv=5)
    searcher.fit(x_train, y_train)
    print("During grid search, best params:")
    print(searcher.best_params_)
    return searcher.best_params_

def tune_nn(x_train, y_train, x_val, y_val):
    f1_val = []
    f1_train = []
    hlist = np.linspace(5,100,20).astype('int')
    for i in hlist:         
            clf = neural_network.MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                                learning_rate_init=0.0001, learning_rate='adaptive',
                                random_state=0, batch_size=100, max_iter=800)
            clf.fit(x_train, y_train)
            y_pred_val = clf.predict(x_val)
            y_pred_train = clf.predict(x_train)
            f1_val.append(f1_score(y_val, y_pred_val, average='weighted'))
            f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
      
    plt.plot(hlist, f1_val, 'o-', color='g', label='Val Weighted F1 Score')
    plt.plot(hlist, f1_train, 'o-', color = 'r', label='Train Weighted F1 Score')
    plt.ylabel('Weighted F1 Score')
    plt.xlabel('Hidden layer size')
    plt.grid()
    
    plt.title(f'Tune hidden size {DATASET_NAME}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'NN tune h on {DATASET_NAME}.png')
    plt.clf()

def study_nn(x_train, y_train, x_test, y_test, attributes, classes, dataset_name='bean'):
    x_trainsub, y_trainsub, x_val, y_val = do_split(x_train, y_train)
    # tune_nn(x_trainsub, y_trainsub, x_val, y_val)
    # best_params = grid_search(x_train, y_train)
    # print(best_params)
    # h, l, b = best_params['hidden_layer_sizes'],best_params['learning_rate_init'],best_params['batch_size']
    h, l, b = 10, 0.0001, 50
    clf = neural_network.MLPClassifier(
        hidden_layer_sizes=(h,),
        activation='logistic',
        early_stopping=False,
        learning_rate_init=l,
        learning_rate='adaptive',
        batch_size=b,
        max_iter=3000
    )

    plot_learning_curve('Neural Network', dataset_name, f'hidden{h}', clf, x_train, y_train, 'f1_weighted')

    clf = neural_network.MLPClassifier(
        hidden_layer_sizes=(h,),
        activation='logistic',
        early_stopping=False,
        learning_rate_init=l,
        learning_rate='adaptive',
        batch_size=b,
        max_iter=3000
    )

    # clf.fit(x_train,y_train)
    # fig, ax = plt.subplots()
    # ax.set_xlabel("iterations")
    # ax.set_ylabel('loss')
    # ax.set_title("loss vs iteration")
    # plt.plot(clf.loss_curve_, label='training losss')
    # # plt.plot(clf.validation_scores_, label='val loss')
    # ax.legend()
    # plt.savefig(f'NN_{DATASET_NAME}_loss.png')
    # plt.clf()
    # print(len(clf.loss_curve_), clf.n_iter_, clf.t_)
    
    final_test(clf, x_train, y_train, x_test, y_test, 'NN', dataset_name, classes)


##################################### SVM ######################################
def tune_svm(x_trainsub, y_trainsub, x_val, y_val):

    f1_val = []
    f1_train = []
    kernel_func = ['linear','poly','rbf','sigmoid']
    for f in kernel_func:         
        if f == 'poly':
            for j in [2,3,4]:
                clf = svm.SVC(kernel=f, degree=j,random_state=0)
                clf.fit(x_trainsub, y_trainsub)
                y_pred_val = clf.predict(x_val)
                y_pred_train = clf.predict(x_trainsub)
                f1_val.append(f1_score(y_val, y_pred_val,  average='weighted'))
                f1_train.append(f1_score(y_trainsub, y_pred_train, average='weighted'))
        else:    
            clf = svm.SVC(kernel=f, random_state=0)
            clf.fit(x_trainsub, y_trainsub)
            y_pred_train = clf.predict(x_trainsub)
            y_pred_val = clf.predict(x_val)
            f1_train.append(f1_score(y_trainsub, y_pred_train, average='weighted'))
            f1_val.append(f1_score(y_val, y_pred_val, average='weighted'))
            
                
    xvals = ['linear','poly2','poly3','poly4', 'rbf','sigmoid']
    plt.plot(xvals, f1_val, 'o-', color='g', label='Val F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'r', label='Train F1 Score')
    plt.ylabel('Weighted average F1 Score')
    plt.xlabel('Kernel Function')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'Tune SVM {DATASET_NAME}.png')
    plt.clf()

def study_svm(x_train, y_train, x_test, y_test, attributes, classes):
    x_trainsub, y_trainsub, x_val, y_val = do_split(x_train, y_train)
    tune_svm(x_trainsub, y_trainsub, x_val, y_val)

    clf = svm.SVC(
        kernel='linear'
    )
    # scores = cross_val_score(clf, x_train, y_train, cv=5)
    # scores = cross_validate(clf, x_train,y_train, cv=5, scoring='accuracy',return_train_score=True)
    # print('SVM cross val scores:', scores)
    plot_learning_curve('SVM', DATASET_NAME, 'linear', clf, x_train, y_train, SCORING)
    final_test(clf, x_train, y_train, x_test, y_test, 'SVM', DATASET_NAME, classes)

################################# boosting ####################################
def tune_adaboost(x_trainsub, y_trainsub, x_val, y_val, max_depth):
    colors=['r', 'g', 'b', 'k']
    c = colors[max_depth-1]
    f1_val = []
    f1_train = []
    n_estimators = np.linspace(1,200,20).astype('int')
    for n in n_estimators:
        weak_learner =  tree.DecisionTreeClassifier(max_depth=max_depth)
        clf = ensemble.AdaBoostClassifier(base_estimator=weak_learner, n_estimators=n, random_state=0)
        clf.fit(x_trainsub, y_trainsub)
        y_pred_train = clf.predict(x_trainsub)
        y_pred_val = clf.predict(x_val)
        f1_train.append(f1_score(y_trainsub, y_pred_train, average='weighted'))
        f1_val.append(f1_score(y_val, y_pred_val, average='weighted'))
      
    plt.plot(n_estimators, f1_val, 'o-', color=c)
    plt.plot(n_estimators, f1_train, 'o-', color = c, label=f'max depth {max_depth}')
    plt.ylabel('Weighted Average F1 Score')
    plt.xlabel('Num of Weak Learners')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'Tune Adaboost depth{max_depth}.png')
    plt.clf()

def study_boosting(x_train, y_train, x_test, y_test, attributes, classes):
    x_trainsub, y_trainsub, x_val, y_val = do_split(x_train, y_train)
    for d in [1,2,3,4]:
        tune_adaboost(x_trainsub, y_trainsub, x_val, y_val, max_depth=d)

    weak_learner = tree.DecisionTreeClassifier(max_depth=3)
    clf = ensemble.AdaBoostClassifier(
        base_estimator=weak_learner,
        n_estimators=60,
        random_state=0)
    # scores = cross_val_score(clf, x_train, y_train, cv=5)
    # scores = cross_validate(clf, x_train,y_train, cv=5, scoring='accuracy',return_train_score=True)
    # print('AdaBoost cross val scores:', scores)
    plot_learning_curve('Adaboosting',DATASET_NAME,'60trees', clf, x_train,y_train, SCORING)

    final_test(clf, x_train, y_train, x_test, y_test, 'Adaboost', DATASET_NAME, classes)


################################################################################################################

def final_test(clf, x_train, y_train, x_test, y_test, clf_name, DATASET_NAME, class_names):
    train_start = time.time()
    clf.fit(x_train, y_train)
    train_stop = time.time()

    test_start = time.time()
    y_pred = clf.predict(x_test)
    test_stop = time.time()

    f1 = f1_score(y_test,y_pred,average='weighted')
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average='weighted')
    recall = recall_score(y_test,y_pred, average='weighted')

    print("Final Evaluation Metrics Using Held-out Test Dataset")
    print("*****************************************************")
    print("Training Time (s): "+"{:.6f}".format(train_stop - train_start))
    print("Query Time (s): "+"{:.6f}\n".format(test_stop - test_start))
    print("F1 Score: "+"{:.2f}".format(f1))
    print("Accuracy: "+"{:.2f}".format(accuracy))
    print("Precision: "+"{:.2f}".format(precision))
    print("Recall: "+"{:.2f}".format(recall))
    print("*****************************************************")
    plot_confusion_matrix(clf_name,DATASET_NAME,'Final CM', clf, x_test,y_test, class_names, None)


def plot_confusion_matrix(clf_name, dataset_name, suffix, fitted_clf, x_test, y_test, class_names, normalize='true'):
    titles_options = [
        (f"Confusion matrix, without normalization ({suffix})", None),
        (f"Normalized confusion matrix ({suffix})", "true"),
    ]
    for title, norm in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(fitted_clf,x_test,y_test,
            display_labels=class_names, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        # print(title)
        # print(disp.confusion_matrix)
    save_name = '_'.join([clf_name,dataset_name,suffix,'confusion'])
    plt.savefig(save_name+ '.png')
    plt.clf()

def do_learning_curve(clf, x_train, y_train, cv, scoring):
    train_sizes_abs, train_scores, test_scores, fit_times, score_times = learning_curve(
        clf, x_train,y_train, train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],cv=cv,scoring=scoring,
        random_state=0, return_times=True
    )
    return train_sizes_abs, train_scores, test_scores, fit_times, score_times

def plot_learning_curve(clf_name, dataset_name, suffix, clf, x_train,y_train, scoring):
    train_sizes_abs, train_scores, test_scores, fit_times, score_times = do_learning_curve(
        clf, x_train, y_train, 5, scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    save_name = '_'.join([clf_name, dataset_name, suffix, scoring])

    # plt.figure(figsize=(48,16))
    plt.title(save_name)
    _, axes = plt.subplots(3,1, figsize=(8,15))
    axes[0].set_title(f'Learning Curve ({clf_name} on {dataset_name})')
    axes[0].set_xlabel("Training examples ratio")
    axes[0].set_ylabel(f"Score ({scoring})")
    axes[0].set_xlim(0.05, 1.05)
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples ratio")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title(f"Scalability of the model ({clf_name} on {dataset_name})")
    axes[1].set_xlim(0.05, 1.05)

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # print(fit_times_mean, fit_time_argsort, fit_time_sorted)
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title(f"Performance of ({clf_name} on {dataset_name})")
    # plt.legend()
    plt.savefig(f'{save_name}.png')
    # plt.show()
    plt.clf()



def study_learners(path):
    x_train, y_train, x_test, y_test, attributes, classes = prepare_data(path)

    study_dt(x_train, y_train, x_test, y_test, attributes, classes)

    study_boosting(x_train, y_train, x_test, y_test, attributes, classes)

    study_knn(x_train, y_train, x_test, y_test, attributes, classes)

    study_nn(x_train, y_train, x_test, y_test, attributes, classes)

    study_svm(x_train, y_train, x_test, y_test, attributes, classes)



if __name__ == '__main__':
    paths = [RICE_PATH, BEAN_PATH]
    dataset_names = ['RiceDataset', 'DryBeanDataset']
    SCORING = 'f1_weighted'
    for path, dataset_name in zip(paths, dataset_names):
        DATASET_NAME = dataset_name
        study_learners(path)