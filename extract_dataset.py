#### original purpose of script: extract embedding data for dataset so that we have the right data to put into ELKI clustering (which is a java-based app)

import numpy as np
import representation_learning as rl
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
                        choices=['pca', 'tsne', 'umap', 'cvae', 'trimap', 'pacmap', 'ae'])
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
    parser.add_argument('--firstx', type=int, default=None,
                        help='Use this when we only want the first x number of samples to be saved (i.e. creating subsets).')
    parser.add_argument('--randsubsetx', type=int, default=None,
                        help='Use this when we only want -random- x number of samples to be saved (i.e. creating subsets).')
    parser.add_argument('--randsubsetseed', type=int, default=None,
                        help='Use this to set the seed of the random subset.')
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
    parser.add_argument('--include-sha', type=dstutil.strtobool, default=False, 
    help="Whether sha should be included at the end of the dataset or not.")

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
    save_dir = args.save_dir

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
        # scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler()]
                
        train_ds = dl.BODMAS(dataset_path = bodmas_ds_path, 
        module_path = ember_module_path, 
        train=True, scalers=scalers, batch_size = args.batchsize, random_order = False)
        # args.batch_size, random_order = args.randomize_train
        
        test_ds = dl.BODMAS(dataset_path = bodmas_ds_path,  
        module_path = ember_module_path, 
        train=False, scalers=scalers, batch_size = args.batchsize, random_order = False)

    elif args.dataset == 'ember':
        # scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler()]
        
        train_ds = dl.EMBER(dataset_path = ember_ds_path,
        module_path = ember_module_path, 
        train=True, scalers=scalers, batch_size = args.batchsize, random_order = False, docker_friendly=True, feature_selection = args.feature_sel)
        
        test_ds = dl.EMBER(dataset_path = ember_ds_path,
        module_path = ember_module_path, 
        train=False, scalers=scalers, batch_size = args.batchsize, random_order = False, docker_friendly=True, feature_selection = args.feature_sel)


    elif args.dataset == 'artificial':
        # scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler()]
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
            gc.collect() # added 2.8.23; quick fix to try to get umap.transform to not encounter memory errors
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
                # embedding_train = embedder.predict(train_X, batch_size=args.batchsize) # train_X; train_ds
                # embedding_test = embedder.predict(test_X, batch_size=args.batchsize)
            else:
                embedder = tf.keras.models.load_model(args.repre_preload)

            embedding_train = embedder.predict(train_X, batch_size=args.batchsize) # train_X; train_ds
            embedding_test = embedder.predict(test_X, batch_size=args.batchsize)

        

    repre_end = time()

    print (f'{args.repre} took {repre_end - repre_start} seconds.')


# 5. save
    global_end = time()

    print (f'Type of embedding_train : {type(embedding_train)}')
    
    csvresults_embed_train = pd.DataFrame(embedding_train)
    csvresults_embed_test = pd.DataFrame(embedding_test)

    mapping = {0:'benign', 1:'pua', 2:'malware'}
    train_y = train_y.apply(lambda x: mapping[x])
    test_y = test_y.apply(lambda x: mapping[x])

    print (f'Shape of train before concat: {csvresults_embed_train.shape}')
    csvresults_embed_train = pd.concat([csvresults_embed_train, train_y.reset_index(drop=True)], axis=1, ignore_index=True)
    csvresults_embed_test = pd.concat([csvresults_embed_test, test_y.reset_index(drop=True)], axis=1, ignore_index=True)
    print (f'Shape of train: {csvresults_embed_train.shape}')
    print (f'Shape of test: {csvresults_embed_test.shape}')

    save = True
    if save:
        if args.firstx is None: # if we dont save the first X amount of samples (without any randommness)
            if args.randsubsetx is None: # save the embedding of the full dataset
                csvresults_embed_train.to_csv(os.path.join(save_dir,'sec24_ae_train.csv'), sep=' ', header=False, index=False)
                csvresults_embed_test.to_csv(os.path.join(save_dir,'sec24_ae_test.csv'), sep=' ', header=False, index=False)
                 
            else: # save embedding of only a random subset of size `args.randsubsetx`
                random_samples_order = np.random.RandomState(seed=args.randsubsetseed).permutation(len(csvresults_embed_train))
                take_howmany = args.randsubsetx
                random_sample_subset = random_samples_order[:take_howmany]
                # rest = random_samples_order[take_howmany:]
                csvresults_embed_train.iloc[random_sample_subset].to_csv(os.path.join(save_dir,f'sec24_ae_train_random_subset={args.randsubsetx}_seed={args.randsubsetseed}.csv'), 
                                                                    sep=' ', header=False, index=False)
                
                
        else:
            csvresults_embed_train[:args.firstx].to_csv(os.path.join(save_dir,f'sec24_ae_train_subset={args.firstx}.csv'), sep=' ', header=False, index=False)
            # csvresults_embed_test.to_csv(os.path.join(save_dir,f'sec24_ae_test_subset={args.firstx}.csv'), sep=' ', header=False, index=False)
            


if __name__ == '__main__':
    main(sys.argv[1:])