 #import KMeans as cumlKMeans
import sklearn.cluster as skcls
import numpy as np
import scipy as sp
import metrics
from tqdm import tqdm 

def train_kmeans(input_data, n_clusters, max_iter = 300, mode='cpu', random_state = None):
    if mode == 'cpu':
        kmeans = skcls.KMeans(n_clusters = n_clusters, max_iter = max_iter, random_state = random_state)
    elif mode == 'gpu':
        import cuml.cluster as cuml
        kmeans = cuml.KMeans(n_clusters = n_clusters, max_iter = max_iter, random_state = 42 if (random_state is None or type(random_state)!=int) else random_state)
    
    kmeans.fit(input_data)
    predicted_clusters = kmeans.predict(input_data)

    return predicted_clusters, kmeans


def train_meanshift(input_data, bandwidth=None, n_jobs=-1, random_state = None):
    ms = skcls.MeanShift(bandwidth=bandwidth, n_jobs=n_jobs, random_state = random_state)
    ms.fit(input_data)
    predicted_clusters = ms.predict(input_data)

    return predicted_clusters, ms

def train_birch(input_data,  n_clusters, threshold=0.5, branching_factor=50):
     
    birch = skcls.Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
    birch.fit(input_data)
    predicted_clusters = birch.predict(input_data)

    return predicted_clusters, birch


def train_bisect(input_data, n_clusters, max_iter = 300, mode='cpu', random_state = None):
    bisect = skcls.BisectingKMeans(n_clusters = n_clusters, max_iter = max_iter, random_state = random_state)
    
    bisect.fit(input_data)
    predicted_clusters = bisect.predict(input_data)

    return predicted_clusters, bisect


def train_dbscan(input_data, epsilon, minpts, mode='cpu', n_jobs=-1, max_mbytes_per_batch=None, 
                 random_state = None):

    print (f'The input data shape that will go into DBSCAN training: {input_data.shape}')
    
    if mode == 'cpu':
        dbscan = skcls.DBSCAN(eps=epsilon, min_samples=minpts, n_jobs=n_jobs, algorithm='kd_tree')
        predicted_clusters = dbscan.fit_predict(input_data)
    elif mode == 'gpu':
        import cuml.cluster as cuml
        dbscan = cuml.DBSCAN(eps=epsilon, min_samples=minpts, calc_core_sample_indices=True)
        predicted_clusters = dbscan.fit_predict(input_data, out_dtype='int64')
    else :
        raise Exception('Unimplemented mode parameter: {}.'.format(mode))
    
    

    return predicted_clusters, dbscan

def dbscan_predict_ver1(model, test_X, train_X, metric=sp.spatial.distance.euclidean, mode='cpu'):
    """ 
    Iterates through core points. If core point is within the radius of epsilon, it assings the new sample 
    to this core point (and its cluster). Does not do an exhaustive search.
    """
    # Result is noise by default
    y_new = np.ones(shape=len(test_X), dtype=int)*-1 
    
    if mode == 'cpu':
        components = model.components_
    elif mode == 'gpu':
        components = train_X[model.core_sample_indices_, :]

    # Iterate all input samples for a label
    for j, x_new in enumerate(test_X):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(components): 
            if metric(x_new, x_core) < model.eps:
                # Assign label of x_core to x_new
                y_new[j] = model.labels_[model.core_sample_indices_[i]]
                break

    return y_new


def dbscan_predict_ver2(model, test_X, train_X, mode='cpu'):
    """
    Computes distances with all core points. Chooses the -closest- core point to assign the cluster label.
    We need train_X data for the GPU version so that we can get the model components.
    """

    nr_samples = test_X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    
    if mode == 'cpu':
        components = model.components_
    elif mode == 'gpu':
        components = train_X[model.core_sample_indices_, :]

    for i in range(nr_samples):
        diff = components - test_X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def train_hdbscan(input_data, mode='cpu', random_state = None):
    pass

def train_hac(input_data, n_clusters, n_neighbors, mode='cpu', random_state = None):
    if mode == 'cpu':
        hac = skcls.AgglomerativeClustering(n_clusters=n_clusters, compute_full_tree=False, linkage='ward')
        # predicted_clusters = hac.fit_predict(input_data)
    elif mode == 'gpu':
        import cuml.cluster as cuml
        hac = cuml.AgglomerativeClustering(n_clusters=n_clusters, n_neighbors=n_neighbors, connectivity='knn')
        # 'knn' connectivity doesn't compute the whole n^2 pairwise distance matrix and helps control the amount of memory used (with the n_neighbors param)
        
    else :
        raise Exception('Unimplemented mode parameter: {}.'.format(mode))
    
    predicted_clusters = hac.fit_predict(input_data)

    return predicted_clusters, hac


def create_neural_logreg(n_labels):
    import tensorflow as tf
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_labels, activation='softmax') # 'sigmoid'
                    ])
    model.compile(loss='sparse_categorical_crossentropy') # 'bce' ;;; 'categorical_crossentropy'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]
    return model, callbacks
    # model.fit(data, labels, epochs=10000, callbacks=[callback]) 

def hac_predict_v1(train_X, predicted_clusters, test_X, model='logreg', mode='cpu'):
    # use a classifier trained on cluster assignments to make new assignments
    allowed_models = ['logreg', 'dtree', 'rf', 'knn', 'softmax', 'sgdlogreg']
    if model.lower() not in allowed_models:
        raise Exception(f'The choice of {model} classifier for new data for HAC algorithm is not supported! Supported ones are: {allowed_models}.')
    


    if mode == 'cpu':
        from sklearn.neighbors import KNeighborsClassifier as skKNN
        from sklearn.tree import DecisionTreeClassifier as skDT
        from sklearn.linear_model import LogisticRegression as skLR
        from sklearn.linear_model import SGDClassifier as skSGD
        from sklearn.ensemble import RandomForestClassifier as skRF
        if model == 'knn':
            clf = skKNN(n_neighbors=3, n_jobs=-1)
        elif model  == 'dtree':
            clf = skDT(max_depth = None, max_features = None)
        elif model == 'logreg':
            clf = skLR(n_jobs=None, penalty='l2')
        elif model == 'sgdlogreg':
            try:
                clf = skSGD(loss='log_loss', penalty='l2', alpha=0.0001, fit_intercept=True, tol=0.001, shuffle=True, epsilon=0.1, n_jobs=-1, random_state=42)
            except ValueError:
                clf = skSGD(loss='log', penalty='l2', alpha=0.0001, fit_intercept=True, tol=0.001, shuffle=True, epsilon=0.1, n_jobs=-1, random_state=42) # this one should work
        elif model == 'rf':
            clf = skRF(n_estimators = 100, max_depth=10, n_jobs=-1)
        elif model == 'softmax': # multionomial logistic regression model
            clf, callbacks = create_neural_logreg(n_labels=len(np.unique(predicted_clusters)))
            

    elif mode == 'gpu':
        from cuml import LogisticRegression as cuLR
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.neighbors import KNeighborsClassifier as cuKNN
        if model == 'knn':
            clf = cuKNN(n_neighbors=3)
        elif model  == 'dtree':
            clf = cuRF(n_estimators = 1, max_depth = None, max_features = None)
        elif model == 'logreg':
            clf = cuLR(penalty='l2', max_iter=500)
        elif model == 'rf':
            clf = cuRF(n_estimators = 100, max_depth=None)
        elif model == 'softmax': # multionomial logistic regression model
            clf, callbacks = create_neural_logreg(n_labels=len(np.unique(predicted_clusters)))
    
    else:
        raise Exception('Unimplemented mode parameter: {}.'.format(mode))

    if model != 'softmax' and model != 'sgdlogreg':
        clf.fit(train_X, predicted_clusters)
        predicted_train = clf.predict(train_X)
        predicted_test = clf.predict(test_X)
    elif model == 'sgdlogreg':
        clf, predicted_train, predicted_test = train_sgdclf(clf,train_X, predicted_clusters, test_X, epochs=100, batch_size=100000, verbose=1)    
    else:
        import tensorflow as tf
        # with tf.device('/CPU:0'):
        clf.fit(train_X, predicted_clusters, epochs=100, callbacks=callbacks)
        predicted_train = clf.predict(train_X).argmax(axis=1)
        predicted_test = clf.predict(test_X).argmax(axis=1)
    acc_score = metrics.acc(predicted_clusters, predicted_train)
    print (f'Accuracy of the HAC-classifier {model} trained on {mode} on train data is: {acc_score}')
    

    return predicted_test, clf, acc_score

def train_sgdclf(clf, train_X, train_y, test_X, epochs=5, batch_size=10000, verbose=0):
    """ Logistic Regression implemented via batchwise SGD training. The model seems to get updated after each batch."""
    def batches(l, batch_size):
        for i in range(0, len(l), batch_size):
            yield l[i:i+batch_size]

    import random
    shuffledRange = list(range(len(train_X)))
    for n in range(epochs):
        random.shuffle(shuffledRange)
        shuffledX = [train_X[i] for i in shuffledRange]
        shuffledY = [train_y[i] for i in shuffledRange]
        for batch in tqdm(batches(range(len(shuffledX)), batch_size=batch_size)):
            clf.partial_fit(shuffledX[batch[0]:batch[-1]+1], shuffledY[batch[0]:batch[-1]+1], classes=np.unique(train_y))
        predicted_train = clf.predict(train_X)
        acc_score = metrics.acc(train_y, predicted_train)
        if verbose == 1:
            print (f'Accuracy of SGDClassifier after epoch {n}: {acc_score * 100}%.')
    
    predicted_test = clf.predict(test_X)

    return clf, predicted_train, predicted_test
