
from time import time
import sys
import argparse

import data_loaders as dl
import sklearn.preprocessing as prep
import os
import pickle
import gzip, bz2
from time import time

from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import accuracy_score, precision_recall_fscore_support as prfs

COMPANY_EMBERDS_PATH = "" # add ember dataset path
COMPANY_EMBER_MODULE_PATH = "" # add ember module path

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



def main(args):
    start = time()
    ember_ds_path = COMPANY_EMBERDS_PATH
    ember_module_path = COMPANY_EMBER_MODULE_PATH 

    parser = argparse.ArgumentParser(description='important features extraction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str,default=None)
    args = parser.parse_args(args)

    scalers = [prep.RobustScaler(), prep.StandardScaler(), prep.MinMaxScaler()]

    train_ds = dl.EMBER(dataset_path = ember_ds_path, 
    module_path = ember_module_path, 
    train=True, scalers=scalers, batch_size = 256, random_order = False, docker_friendly=True)
    
    test_ds = dl.EMBER(dataset_path = ember_ds_path, 
    module_path = ember_module_path, 
    train=False, scalers=scalers, batch_size = 256, random_order = False, docker_friendly=True)

    train_X = train_ds.X
    train_y = train_ds.y
    test_X = test_ds.X
    test_y = test_ds.y


    rfclf = RF(n_estimators=500, max_depth=None, max_features='sqrt', n_jobs=-1)
    rfclf.fit(train_X, train_y)
    preds_train = rfclf.predict(train_X)
    # acc = rfclf.score(train_X, train_y)
    preds_test = rfclf.predict(test_X)

    acc_train = accuracy_score(train_y, preds_train)
    acc_test = accuracy_score(test_y, preds_test)

    prfs_train_noavg = prfs(train_y, preds_train, average=None)
    prfs_train_macro = prfs(train_y, preds_train, average='macro')
    
    prfs_test_noavg = prfs(test_y, preds_test, average=None)
    prfs_test_macro = prfs(test_y, preds_test, average='macro')

    results = dict(
        acc_train = acc_train,
        acc_test = acc_test,
        prfs_train_noavg = prfs_train_noavg,
        prfs_train_macro = prfs_train_macro,
        prfs_test_noavg = prfs_test_noavg,
        prfs_test_macro = prfs_test_macro,
        time_taken = time() - start
    )

    importances = rfclf.feature_importances_

    save_pickle(results, path = os.path.join(args.save_dir, 'results.gzpkl'))
    save_pickle(importances, path = os.path.join(args.save_dir, 'importances.gzpkl'))
    save_pickle(rfclf, path = os.path.join(args.save_dir, 'rfinst.gzpkl'))

    print (f'The script exec time: {time() - start} . Exiting...')



if __name__ == '__main__':
    main(sys.argv[1:])