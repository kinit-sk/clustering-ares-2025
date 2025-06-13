import numpy as np
import representation_learning as rl
import clustering as cl
import data_loaders as dl
import sklearn.preprocessing as prep
from distutils import util as dstutil
from time import time
import os
import pickle
import argparse
import sys
import metrics
from datetime import datetime
import bz2, gzip
from collections import Counter
import pandas as pd
import gc

DEFAULT_BODMAS_PATH = "/data/datasets/bodmas"
DEFAULT_EMBERDS_PATH = "/data/datasets/Ember/2018/ember2018"
DEFAULT_EMBER_MODULE_PATH = "/data/datasets/Ember/"

# add your own paths
COMPANY_BODMAS_PATH = ""
COMPANY_EMBERDS_PATH = ""
COMPANY_EMBER_MODULE_PATH = ""

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def check_path_created(path):
    """
    Checks whether the folders in the path are created. If not, it creates the folders.
    """
    filename = os.path.basename(path)
    if '.' in filename:
        # the file path ends with a folder; not a file!
        only_directories = os.path.dirname(path)
    else:
        only_directories = path
        # we will assume the path ends with a file name (not necessarily an existing file)
    if only_directories != '': # no directories in the path
        os.makedirs(only_directories, exist_ok=True)

def save_pickle(obj, path):
    # check if all folders in the path exist; if not create them
    check_path_created(path)

    # based on the file extension, we support pickling with compression and pickling with no compression
    filepath, extension = os.path.splitext(path)
    if extension == '.bz2pkl': # use bz2 compression
        with bz2.BZ2File(path, 'wb') as f: 
            pickle.dump(obj, f)
    elif extension == '.gzpkl': # use gzip compression
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f)
    else: # use classic pickle with no compression
        with open(path, 'wb') as fil:
            pickle.dump(obj, fil)

def load_pickle(path):
    filepath, extension = os.path.splitext(path)
    print (f'the extension is {extension}')
    if extension == '.bz2pkl':
        with bz2.BZ2File(path, 'rb') as f:
            obj = pickle.load(f)
    elif extension == '.gzpkl':
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(path, 'rb') as fil:
            obj = pickle.load(fil)
    
    return obj

def has_method(inst,func_name):
    """ 
    Checks whether an object (instance) has a method with a specific name.
    """
    attr = getattr(inst, func_name, None)
    if attr is not None:
        if callable(attr):
            return True
    return False


def parse_args(args):
    parser = argparse.ArgumentParser(description='RO1 main script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='bodmas',
                        choices=['ember', 'bodmas'])
    # parser.add_argument('--batch-size', default=256, type=int)
    # parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--repre-mode', default='cpu',
                        choices=['cpu', 'gpu']) # either 'cpu' or 'gpu' --> uses the mode whenever possible
    parser.add_argument('--cls-mode', default='cpu',
                        choices=['cpu', 'gpu'])
    # parser.add_argument('--cluster-centers', default=None)
    parser.add_argument('--clusterer', default='kmeans',
                        choices=['kmeans', 'dbscan', 'hdbscan', 'birch', 'bisect', 'meanshift', 'hac'])
    parser.add_argument('--n-clusters', type=int, default=3)
    parser.add_argument('--preproc-json', type=str, default=None)
    parser.add_argument('--repre', default='pca',
                        choices=['pca', 'tsne', 'umap', 'ae'])
    parser.add_argument('--repre-preload', type=str, default=None) # preload a representation that will be used
    parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon parameter for DBSCAN')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='For NN representation models such as AE, CVAE, PacMAP. Not for UMAP/TSNE/etc.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='For NN representation models such as AE, CVAE, PacMAP. Not for UMAP/TSNE/etc.')
    parser.add_argument('--machine', type=str, default='kinit',
                        help="Chooses different paths to dataset folders based on what machine we are using.")
    parser.add_argument('--basepreproc', type=int, default=0,
                        help='What kind of base preprocessing should be used -- default (0) means to use \
                            [RobustScaler,StandardScaler, MinMaxScaler]. 1 means to use 0 + QuantileTransformer. \
                            2 means use QuantileTransformer only.')
    parser.add_argument('--hac-clf', type=str, default='logreg', # added on 24.7.2023
                        help='Classifier for HAC clusterer - we need to use it since HAC cannot be computed on the whole dataset.')
    parser.add_argument('--feature-sel', type=int, default=None,
                        help='So far meant only to be used with EMBER dataset to choose the feature selection we want.')

    parser.add_argument('--cols-leaveout', type=int, default=0, help = "Which set of columns to leave out. Possible options: 0,1,2,3. ")
    parser.add_argument('--randomize-train', type=dstutil.strtobool, default=False, 
    help="Tells the script whether we should randomize train samples which get fed into the model. The dataloader for the specific dataset must implement randomization" \
    "for this to work.")
    parser.add_argument('--save-labels', type=dstutil.strtobool, default=False, 
    help="Whether to save clustering and true labels that were used for evaluation.")
    parser.add_argument('--clf-mode', default='cpu',
                        choices=['cpu', 'gpu'],
                        help='This is only relevant for setting HAC as the clusterer. This mode of the classification algorithm (either cpu or gpu) will be used instead of argument `cls-mode`.')
    parser.add_argument('--logreg-max-iter', type=int, default=500, help = "For hac-clf context where logistic regression is also the classifier of choice, use the specified " \
    "maximum number of iterations.")
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='N_jobs argument for DBSCAN-sklearn (right now used -only- for DBSCAN).')
    parser.add_argument('--remove-benign', type=dstutil.strtobool, default=False, 
    help="Remove benign samples from the datasets.")

    args = parser.parse_args()
    return args


def tf_allow_memory_growth(tf):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main(args):
    global_start = time()

    # 0. Parse arguments
    args = parse_args(args)

    integer_time_now = int(datetime.utcnow().timestamp())
    save_dir = args.save_dir + '_{}'.format(integer_time_now)

    if args.machine == 'kinit':
        bodmas_ds_path = DEFAULT_BODMAS_PATH
        ember_ds_path = DEFAULT_EMBERDS_PATH
        ember_module_path = DEFAULT_EMBER_MODULE_PATH 
        
    elif args.machine == 'company':
        bodmas_ds_path = COMPANY_BODMAS_PATH
        ember_ds_path = COMPANY_EMBERDS_PATH
        ember_module_path = COMPANY_EMBER_MODULE_PATH 



# 1. load data ;;;; report RAM usage of loaded data
    if args.basepreproc == 0:
        scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler()]
    elif args.basepreproc == 1:
        scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler(), prep.QuantileTransformer()]
    elif args.basepreproc == 2:
        scalers = [prep.QuantileTransformer()]
    else:
        raise Exception('Invalid number used for "basepreproc" argument. The allowed values are 0,1,2.')

    if args.dataset == 'bodmas':
        train_ds = dl.BODMAS(dataset_path = bodmas_ds_path, 
        module_path = ember_module_path, 
        train=True, scalers=scalers, batch_size = args.batchsize, random_order = False)
        
        test_ds = dl.BODMAS(dataset_path = bodmas_ds_path,  
        module_path = ember_module_path,
        train=False, scalers=scalers, batch_size = args.batchsize, random_order = False)

    elif args.dataset == 'ember':
        train_ds = dl.EMBER(dataset_path = ember_ds_path,
        module_path = ember_module_path, 
        train=True, scalers=scalers, batch_size = args.batchsize, random_order = False, docker_friendly=True, 
        feature_selection = args.feature_sel, remove_benign=args.remove_benign)
        
        test_ds = dl.EMBER(dataset_path = ember_ds_path, 
        module_path = ember_module_path, 
        train=False, scalers=scalers, batch_size = args.batchsize, random_order = False, docker_friendly=True, 
        feature_selection = args.feature_sel, remove_benign=args.remove_benign)

    elif args.dataset == 'artificial':
        artif = dl.ArtificialDataset(n_samples = 10000, n_features = 2235, centers = 100, test_size = 0.2,
                                      shuffle = True, scalers=scalers)

    else:
        print ('Unimplemented dataset. Exiting.. ')
        exit()

    if args.dataset == 'artificial':
        train_X = artif.train_X
        train_y = artif.train_y
        test_X = artif.test_X
        test_y = artif.test_y

    else: 
        train_X = train_ds.X
        train_y = train_ds.y
        test_X = test_ds.X
        test_y = test_ds.y

# 2. preprocess /// representation learning
    repre_start = time()
    if args.repre_preload is None:
        if args.repre == 'pca':
            embedding_train, embedder = rl.train_pca(input_data = train_X, n_components = args.n_components, mode=args.repre_mode)
            embedding_test = embedder.transform(test_X)

        elif args.repre == 'tsne':
            embedding_train, embedder = rl.train_tsne(input_data = train_X, n_components = args.n_components,
                                                    perplexity=30, mode=args.repre_mode)
            embedding_test = embedder.transform(test_X)
        
        elif args.repre == 'umap':
            if args.dataset in ['ember']:
                print ('Running UMAP in low memory mode.')
                if args.dataset == 'ember' and args.repre_mode == 'cpu':
                    embedding_train, embedder = rl.train_umap(input_data = train_X[:400000], n_components = args.n_components,
                                                        n_neighbors=30, low_memory = True, mode=args.repre_mode)

                    embedding_train_rest = embedder.transform(train_X[400000:])
                    embedding_train = np.concatenate([embedding_train, embedding_train_rest])

                else:
                    embedding_train, embedder = rl.train_umap(input_data = train_X, n_components = args.n_components,
                                                        n_neighbors=30, low_memory = True, mode=args.repre_mode)
            else:
                embedding_train, embedder = rl.train_umap(input_data = train_X, n_components = args.n_components,
                                                    n_neighbors=30, mode=args.repre_mode)
            embedding_test = embedder.transform(test_X)
            gc.collect()

        elif args.repre == 'ae':
            import tensorflow as tf
            # tf_allow_memory_growth(tf) # doesn't work anyway when called as a function
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            
            input_ftrs = train_X.shape[1]
            zero_one_output = False
            init = 'glorot_uniform'
            dims = [input_ftrs, 500, 500, 2000, args.n_components]
            model, embedder = rl.autoencoder(dims, init=init, zero_one_output=zero_one_output)
            check_path_created(save_dir) # for csv logs from AE training
            rl.pretrain(model, x=train_X, y=train_y, training_generator = None, epochs=args.epochs, 
                        batch_size=args.batchsize, save_dir=save_dir, args=args)
            
            embedding_train = embedder.predict(train_X, batch_size=args.batchsize) # train_X; train_ds
            embedding_test = embedder.predict(test_X, batch_size=args.batchsize) # test_X

        else:
            print ('Unimplemented representation; quitting.....')
            exit()

    else:
        print ('Loading representation from saved file.')
        if args.repre != 'ae':
            embedder = load_pickle(args.repre_preload)
            if args.repre != 'umap':
                embedding_train = embedder.transform(train_X)
            else:
                print (f"UMAP's embedding_ matrix type: {type(embedder.embedding_)} and shape: {embedder.embedding_.shape}.")
                embedding_train = embedder.embedding_ 
                umap_module = type(embedder).__module__.split('.')[0]
                if args.dataset == 'ember' and umap_module == 'umap': # if umap_module == 'umap' we presume to be using the CPU umap implementation
                    embedding_train_rest = embedder.transform(train_X[400000:])
                    embedding_train = np.concatenate([embedding_train, embedding_train_rest])
            
            print ('')
            embedding_test = embedder.transform(test_X)
            gc.collect() # quick fix to try to get umap.transform to not encounter memory errors
        else:
            import tensorflow as tf
            tf_allow_memory_growth(tf)
            # allow for the option to load a ModelCheckpoint saved model instead of the model we have at the end of training
            # the ModelCheckpoint model has to be loaded via loading a "whole folder"
            # find out whether the repre_preload path is a file or a folder
            if os.path.isdir(args.repre_preload): # we're loading a ModelCheckpoint-saved model
                print ('Loading a ModelCheckpoint-saved Autoencoder model.')
                ae_modelcp = tf.keras.models.load_model(args.repre_preload)
                embedder = tf.keras.Model(inputs=ae_modelcp.input, outputs=ae_modelcp.get_layer('encoder_3').output, name='encoder')
                embedding_train = embedder.predict(train_X, batch_size=args.batchsize) # train_X; train_ds
                embedding_test = embedder.predict(test_X, batch_size=args.batchsize)
            else:
                embedder = tf.keras.models.load_model(args.repre_preload)
                embedding_train = embedder.predict(train_X, batch_size=args.batchsize) # train_X; train_ds
                embedding_test = embedder.predict(test_X, batch_size=args.batchsize)
        

    repre_end = time()

    print (f'{args.repre} took {repre_end - repre_start} seconds.')

# 3. cluster; with appropriate settings
    cluster_start = time()
    print (f'Running clustering in {args.cls_mode} mode.')
    if args.clusterer == 'kmeans':
        y_pred, clusterer = cl.train_kmeans(input_data = embedding_train, n_clusters=args.n_clusters, mode=args.cls_mode, max_iter = 300)
        y_pred_test = clusterer.predict(embedding_test)

    elif args.clusterer == 'dbscan':
        y_pred, clusterer = cl.train_dbscan(input_data = embedding_train, epsilon=args.eps, minpts=5, mode=args.cls_mode)
        y_pred_test = cl.dbscan_predict_ver2(clusterer, embedding_test, embedding_train, mode=args.cls_mode)

    elif args.clusterer == 'birch':
        bf = 50 
        if args.n_clusters > 30000 : 
            threshold = 0.01
        elif args.n_clusters > 20000: 
            threshold = 0.03
        elif args.n_clusters > 10000:
            threshold = 0.05
        elif args.n_clusters > 1500:
            threshold = 0.1
        elif args.n_clusters > 500:
            threshold = 0.2
        else:
            threshold = 0.5

        if args.dataset == 'ember' and args.n_clusters >= 50000: #added 1.12.2023
            random_samples = np.random.RandomState(seed=42).permutation(len(train_X))
            take_howmany = 130000 # is this the best number we can work with for ember? idk
            perm_train = random_samples[:take_howmany]
            perm_rest = random_samples[take_howmany:]


            y_pred_perm_train, clusterer = cl.train_birch(input_data = embedding_train[perm_train], n_clusters=args.n_clusters,
                                            threshold=threshold, branching_factor=bf)
            
            y_pred_perm_rest = clusterer.predict(embedding_train[perm_rest])
            y_pred = np.concatenate([y_pred_perm_train, y_pred_perm_rest])
            argsortz = random_samples.argsort()
            y_pred = y_pred[argsortz]
            y_pred_test = clusterer.predict(embedding_test)

        else:
            y_pred, clusterer = cl.train_birch(input_data = embedding_train, n_clusters=args.n_clusters,
                                            threshold=threshold, branching_factor=bf)
            y_pred_test = clusterer.predict(embedding_test)

    elif args.clusterer == 'hac':
        if (args.dataset in ['ember']):
            random_samples = np.random.RandomState(seed=42).permutation(len(train_X))
            take_howmany = 130000
            perm_train = random_samples[:take_howmany]
            perm_rest = random_samples[take_howmany:]
            
            # embedding_train_rest = embedding_train[perm_rest]
            # embedding_train_actual = embedding_train[perm_train]

            y_pred_perm_train, clusterer = cl.train_hac(input_data = embedding_train[perm_train], n_clusters = args.n_clusters, n_neighbors=15, mode=args.cls_mode) # n_neighbors=80 results in OOM on 32GB GPU
            # y_pred_test = np.empty(len(y_pred)) # predicting for test data so far unimplemented
            # y_pred_test[:] = np.nan
            # clf_model = 'logreg'
            clf_model = args.hac_clf
            y_pred_test, hac_clf, clf_acc_score = cl.hac_predict_v1(embedding_train[perm_train], y_pred_perm_train, embedding_test, model=clf_model, mode = args.clf_mode)

            # y_pred_perm_rest = hac_clf.predict(embedding_train[perm_rest]) # this is what causes memory issues; if we do it in 'batches' instead of everything at once, the issue should be solved
            y_pred_perm_rest = [] 
            for subset in np.array_split(embedding_train[perm_rest], 100): # 10 parts is still very RAM-heavy; let's do 100
                y_pred_perm_rest.append(hac_clf.predict(subset))
            
            y_pred_perm_rest = np.concatenate(y_pred_perm_rest)
            y_pred = np.concatenate([y_pred_perm_train, y_pred_perm_rest])
            argsortz = random_samples.argsort()
            y_pred = y_pred[argsortz]



        else:
            y_pred, clusterer = cl.train_hac(input_data = embedding_train, n_clusters = args.n_clusters, n_neighbors=15, mode=args.cls_mode) # n_neighbors=80 results in OOM on 32GB GPU
            # y_pred_test = np.empty(len(y_pred)) # predicting for test data so far unimplemented
            # y_pred_test[:] = np.nan
            # clf_model = 'logreg'
            clf_model = args.hac_clf
            y_pred_test, hac_clf, clf_acc_score = cl.hac_predict_v1(embedding_train, y_pred, embedding_test, model=clf_model, mode = args.cls_mode)


    elif args.clusterer == 'hdbscan':
        pass

    elif args.clusterer == 'meanshift':
        y_pred, clusterer = cl.train_meanshift(input_data = embedding_train, bandwidth=None) # if 'None', the bandwidth is computed
        y_pred_test = clusterer.predict(embedding_test)

    elif args.clusterer == 'bisect':
        y_pred, clusterer = cl.train_bisect(input_data = embedding_train, n_clusters=args.n_clusters, max_iter = 300)
        y_pred_test = clusterer.predict(embedding_test)
    else :
        print ('Unimplemented clusterer! Exiting...')
    cluster_end = time()


# 3,5.) monitor the learning process and then save results in an appropriate manner
# 4. evaluate
    metrics_computed = dict(
        nmi = np.round(metrics.nmi(train_y, y_pred), 5),
        ari = np.round(metrics.ari(train_y, y_pred), 5),
        hom = np.round(metrics.hom(train_y, y_pred), 5),
        vme = np.round(metrics.vme(train_y, y_pred), 5),
        vme_half = np.round(metrics.vme(train_y, y_pred, beta=0.5), 5),

        nmi_test = np.round(metrics.nmi(test_y, y_pred_test), 5),
        ari_test = np.round(metrics.ari(test_y, y_pred_test), 5),
        hom_test = np.round(metrics.hom(test_y, y_pred_test), 5),
        vme_test = np.round(metrics.vme(test_y, y_pred_test), 5),
        vme_half_test = np.round(metrics.vme(test_y, y_pred_test, beta=0.5), 5),
    )

# 5. save
    global_end = time()
    embedder_params = None
    clusterer_params = None
    if has_method(embedder, "get_params"):
        embedder_params = embedder.get_params()
    if has_method(clusterer, "get_params"):
        clusterer_params = clusterer.get_params()

    effective_n_clusters = len(Counter(y_pred))

    results = {'times': {'time_all':global_end-global_start, 
               'time_clustering':cluster_end - cluster_start,
               'time_representation':repre_end - repre_start},
               'metrics':metrics_computed,
               'effective_n_clusters':effective_n_clusters,
               'labels':{ # LABELS added only on 24.5.23 ! therefore they won't be found in experiments conducted sooner
                   'train_y':train_y,
                   'y_pred':y_pred,
                   'test_y':test_y,
                   'y_pred_test':y_pred_test
               },
            #    'instances': {'embedder':embedder, 'clusterer':clusterer},
               'args':args,
               'embedder_params':embedder_params, # these two params were added also only on 24.5.23
               'clusterer_params':clusterer_params
               }
    # save results to pickle
    save_pickle(results, path = os.path.join(save_dir, 'results.pkl'))
    del results['labels']
    save_pickle(results, path = os.path.join(save_dir, 'results_nolabels.pkl'))
    del results['embedder_params']
    del results['clusterer_params']
    save_pickle(results, path = os.path.join(save_dir, 'results_nolabels_noparams.pkl'))
    csvresults = pd.DataFrame([{**metrics_computed, **results['times'], 'effective_n_clusters':effective_n_clusters}])
    csvresults.to_csv(os.path.join(save_dir,'results.csv'))
    # save representation
    embedder_filename = 'embedder.pkl' # decide on the name based on the compression level we need
    if args.repre == 'umap':
        embedder_filename = 'embedder.gzpkl'
    if args.repre != 'ae':
        save_pickle(embedder, path = os.path.join(save_dir, embedder_filename))
    else:
        save_path = os.path.join(save_dir, 'encoder_model.h5')
        tf.keras.models.save_model(embedder, save_path)
        pass # do nothing, the saving should have 
    # save clusterer
    if args.clusterer != 'birch': # for some reason saving birch (with higher amounts of clusters) results in a RecursionError which I'm not sure how to deal with
        save_pickle(clusterer, path = os.path.join(save_dir, 'clusterer.pkl'))

    if args.clusterer == 'hac': # save the classifier used to predict on new data for HAC algorithm
        save_pickle(hac_clf, path = os.path.join(save_dir, f'hac_clf_{clf_model}.gzpkl'))

    print ('Results saved. Script finished sucessfully. Exitting...')


if __name__ == '__main__':
    main(sys.argv[1:])