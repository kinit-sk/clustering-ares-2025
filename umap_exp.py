import argparse
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from collections import Counter
import sklearn.preprocessing as prep
import umap
import bz2, gzip
import data_loaders as dl
import sys
from sklearn.decomposition import PCA

KINIT_bodmas_dataset_path = '/data/datasets/bodmas'
KINIT_ember_module_path = "/data/datasets/Ember/"
KINIT_ember_dataset_path = "/data/datasets/Ember/2018/ember2018"

# add your paths
COMPANY_bodmas_dataset_path = ""
COMPANY_ember_module_path = ""
COMPANY_ember_dataset_path = ""


def save_pickle(obj, path):
    # check if all folders in the path exist; if not create them
    check_path_created(path)

    # based on the file extension, we support pickling with compression and pickling with no compression
    filepath, extension = os.path.splitext(path)
    if extension == 'bz2pkl': # use bz2 compression
        with bz2.BZ2File(path, 'wb') as f: 
            pickle.dump(obj, f)
    elif extension == 'gzpkl': # use gzip compression
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f)
    else: # use classic pickle with no compression
        with open(path, 'wb') as fil:
            pickle.dump(obj, fil)
            
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

def parse_args(args):
    parser = argparse.ArgumentParser(description='RO1 main script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--batch-size', default=256, type=int)
    # parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--save-dir', default='results')
    # parser.add_argument('--repre-mode', default='cpu',
    #                     choices=['cpu', 'gpu']) # either 'cpu' or 'gpu' --> uses the mode whenever possible
    # parser.add_argument('--cls-mode', default='cpu',
    #                     choices=['cpu', 'gpu'])
    parser.add_argument('--option', type=int, default=1)
    # parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--machine', type=str, default='kinit',
                        help="Chooses different paths to dataset folders based on what machine we are using.")
    parser.add_argument('--repre', type=str, default='umap',
                    help="The representation to use.")

    args = parser.parse_args()
    return args





def main(args):
    global_start = time()

    # 0. Parse arguments
    args = parse_args(args)


    if args.machine == 'kinit':
        bodmas_dataset_path = KINIT_bodmas_dataset_path
        ember_dataset_path = KINIT_ember_dataset_path
        ember_module_path = KINIT_ember_module_path
    elif args.machine == 'company':
        bodmas_dataset_path = COMPANY_bodmas_dataset_path
        ember_dataset_path = COMPANY_ember_dataset_path
        ember_module_path = COMPANY_ember_module_path
    else:
        raise Exception('Unknown machine: {}'.format(args.machine))


    # integer_time_now = int(datetime.utcnow().timestamp())
    if args.repre == 'umap':
        uinst = umap.UMAP(n_neighbors=30, n_components=2,  init='random', low_memory=False, random_state=42)
    elif args.repre == 'pca':
        uinst = PCA(n_components=2, random_state = 42)
    else:
        raise Exception(f'Representation {args.repre} not implemented!')

    scalers = None

    train_ds = dl.BODMAS(dataset_path = bodmas_dataset_path, 
    module_path = ember_module_path, 
    train=True, scalers=scalers, batch_size = 256, random_order = False)
    # args.batch_size, random_order = args.randomize_train

    test_ds = dl.BODMAS(dataset_path = bodmas_dataset_path,  
    module_path = ember_module_path,
    train=False, scalers=scalers, batch_size = 256, random_order = False)


    train_X = train_ds.X
    train_y = train_ds.y
    test_X = test_ds.X
    test_y = test_ds.y

    bodmas_X = np.concatenate([train_X, test_X])
    bodmas_y = np.concatenate([train_y, test_y])

    del train_X, train_y, test_X, test_y, train_ds, test_ds

    scalers = None

    train_ds = dl.EMBER(dataset_path = ember_dataset_path, 
    module_path = ember_module_path, 
    train=True, scalers=scalers, batch_size = 256, random_order = False, docker_friendly = True)

    test_ds = dl.EMBER(dataset_path = ember_dataset_path,
    module_path = ember_module_path, 
    train=False, scalers=scalers, batch_size = 256, random_order = False, docker_friendly = True)

    train_X = train_ds.X
    train_y = train_ds.y
    test_X = test_ds.X
    test_y = test_ds.y

    ember_X = np.concatenate([train_X[:400000], test_X])
    ember_y = np.concatenate([train_y, test_y])

    del train_X, train_y, test_X, test_y, train_ds, test_ds

    # now we can finally apply basic preprocessing
    qt = prep.QuantileTransformer()

    # mock_data = np.random.random((10000,1000))
    if args.option == 1: # both
        both = np.concatenate([bodmas_X, ember_X])
    #     mock_data = qt.fit_transform(mock_data)
        both = qt.fit_transform(both)
        out_data = uinst.fit_transform(both)
        to_save = {'both_transformed':out_data, 'bodmas_n':len(bodmas_X), 
                   'time_taken':time() - global_start}
        save_path = f'results/{args.repre}_exp/both/data.gzpkl'
        check_path_created(save_path)
        save_pickle(to_save, path=save_path)
    elif args.option == 2: # train on ember, transform bodmas
        ember_X = qt.fit_transform(ember_X)
        bodmas_X = qt.transform(bodmas_X)
        ember_transformed = uinst.fit_transform(ember_X)
        bodmas_transformed = uinst.transform(bodmas_X)
        save_path = f'results/{args.repre}_exp/ember_only/data.gzpkl'
        to_save = {'ember_transformed':ember_transformed, 'bodmas_transformed':bodmas_transformed, 
                   'bodmas_n':len(bodmas_X), 'time_taken':time() - global_start}
        check_path_created(save_path)
        save_pickle(to_save, path=save_path)
    else:
        raise Exception('Unsupported option {}.'.format(args.option)) 



if __name__ == '__main__':
    main(sys.argv[1:])