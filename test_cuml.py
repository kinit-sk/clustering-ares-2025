import cuml.cluster as cuml
from sklearn.datasets import make_blobs
from time import time
import sys
import argparse

def main(args):
    parser = argparse.ArgumentParser(description='testing cuml',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repre', type=str,
                        help='Specify the representation that we want to run experiments for. A lot of the time,  \
                        the representation will already be trained in some previous run.')
    parser.add_argument('--clusters', type=int, default=None)
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args(args)



    start = time()
    print ('Initializing data.')
    X, y = make_blobs(n_samples=args.samples, centers=args.clusters, n_features=20, random_state=0)
    print ('Computing cuml KMeans.')
    
    kmeans = cuml.KMeans(n_clusters=args.clusters, max_iter = 100, random_state = 0)
    kmeans.fit(X)
    predicted_clusters = kmeans.predict(X)
    end = time()
    print(f'Time taken: {end-start} seconds.')
    print (f'Number of predicted clusters: {len(predicted_clusters)}')
    # print(predicted_clusters)

if __name__ == '__main__':
    main(sys.argv[1:])