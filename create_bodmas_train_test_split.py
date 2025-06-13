import sys, os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def main(args):
    # basepath = '/mnt/data/martin.mocko/data/bodmas/'
    parser = argparse.ArgumentParser(description='testing cuml',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-path', type=str,
                        help='Will create the specific Bodmas dataset train-test split that was used for our experiment.')
    # parser.add_argument('--metafile-path', type=str,
    #                     help='Will create the specific Bodmas dataset train-test split that was used for our experiment.')
    parser.add_argument('--save-dir', type=str,
                        help='Will create the specific Bodmas dataset train-test split that was used for our experiment.')
    args = parser.parse_args(args)
    

    evr = np.load(args.dataset_path)

    evr_X = evr['X']
    evr_y = evr['y']

    train_idx, test_idx = train_test_split(range(len(evr_X)), test_size=0.1, random_state=42)

    
    np.save(os.path.join(args.save_dir, 'X_train.npy'), evr_X[train_idx])
    np.save(os.path.join(args.save_dir, 'y_train.npy'), evr_y[train_idx])

    np.save(os.path.join(args.save_dir, 'X_test.npy'), evr_X[test_idx])
    np.save(os.path.join(args.save_dir, 'y_test.npy'), evr_y[test_idx])

    np.save(os.path.join(args.save_dir, "train_idx.npy"), np.array(train_idx))
    np.save(os.path.join(args.save_dir, "test_idx.npy"), np.array(test_idx))


if __name__ == '__main__':
    main(sys.argv[1:])