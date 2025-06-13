import sys
import argparse
import os
from distutils import util as dstutil

# DEFAULT_SAVE_DIR = 'results/bodmas/'

TRAINED_REPRES_KINIT = {
    'bodmas':{
        'pca':'results/bodmas/pca_kmeans_k=10_comp=10/embedder.pkl',
        'umap':'results/bodmas/umap_kmeans_k=10_comp=10/embedder.pkl',
        'ae':'results/bodmas/umap_kmeans_k=10_comp=10/ae_kmeans_k=10_comp=10_1685017596/embedder.pkl'
    }, 
    'ember': {
        'pca': 'results/ember/pca_kmeans_k=10_comp=10_1684703561/embedder.pkl',
        'umap': 'results/ember/umap_kmeans_k=10_comp=10_1684865077/embedder.pkl',
        'ae': 'results/ember/ae_kmeans_k=10_comp=10_1685019003/embedder.pkl'
    }
}

# add your own paths
TRAINED_REPRES_COMPANY = {
    'bodmas':{
        'pca':'',
        'umap':'',
        'ae':''
    }, 
    'ember': {
        'pca': '',
        'umap': '',
        'ae': ''
    }
}

TRAINED_REPRES = TRAINED_REPRES_KINIT


def center_based(n_clusterz, args):
    # run_cmd = "python -u ro1_main.py --dataset {} --clusterer {} --basepreproc {} --repre {} --n-clusters {} " \
    # "--repre-preload {} --save-dir {} --repre-mode {} --cls-mode {} --machine {}"
    # run_cmd_no_preload = "python -u ro1_main.py --dataset {} --clusterer {} --basepreproc {} --repre {} --n-clusters {} " \
    # "--save-dir {} --repre-mode {} --cls-mode {} --machine {}"

    saved_repres_dataset = TRAINED_REPRES[args.dataset]

    run_cmd = "python -u ro1_main.py"
    run_cmd += f' --dataset {args.dataset}'
    run_cmd += f' --clusterer {args.clusterer}'
    run_cmd += f' --basepreproc {args.basepreproc}'
    run_cmd += f' --repre {args.repre}'
    if args.repre_override is not None:
        run_cmd += f' --repre-preload {args.repre_override}'
    # elif args.repre in saved_repres_dataset and args.n_components == 10:
    #     run_cmd += f' --repre-preload {saved_repres_dataset[args.repre]}'
    else:
        run_cmd += f' --n-components {args.n_components}'
    run_cmd += f' --repre-mode {args.repre_mode}'
    run_cmd += f' --cls-mode {args.cls_mode}'
    run_cmd += f' --machine {args.machine}'
    if args.clusterer == 'hac':
        run_cmd += f' --hac-clf {args.hac_clf}'
        run_cmd += f' --clf-mode {args.clf_mode}'
        if args.hac_clf == 'logreg':
            run_cmd += f' --logreg-max-iter {args.logreg_max_iter}'
    if args.feature_sel is not None:
        run_cmd += f' --feature-sel {args.feature_sel}'
    if args.cols_leaveout != 0:
        run_cmd += f' --cols-leaveout {args.cols_leaveout}'
    
    for n_cls in n_clusterz:
        save_dir = os.path.join(args.save_dir, '{}_{}_k={}_comp={}'.format(args.repre, args.clusterer, n_cls, args.n_components))
        new_cmd = run_cmd + ''
        new_cmd += f' --n-clusters {n_cls}'
        new_cmd += f' --save-dir {save_dir}'
        print (f'The run cmd is: {new_cmd}')
        os.system(new_cmd)

def dbscan_based(n_eps, args):
    run_cmd = "python -u ro1_main.py --dataset {} --clusterer {} --basepreproc {} --repre {} --eps {} " \
    "--repre-preload {} --save-dir {} --repre-mode {} --cls-mode {} --machine {}"
    run_cmd_no_preload = "python -u ro1_main.py --dataset {} --clusterer {} --basepreproc {} --repre {} --eps {} " \
    "--save-dir {} --repre-mode {} --cls-mode {} --machine {} --n-components {}"

    if args.cols_leaveout != 0:
        run_cmd += f' --cols-leaveout {args.cols_leaveout}'
        run_cmd_no_preload += f' --cols-leaveout {args.cols_leaveout}'
    
    saved_repres_dataset = TRAINED_REPRES[args.dataset]
    for eps in n_eps:
        save_dir = os.path.join(args.save_dir, '{}_{}_eps={}_comp={}'.format(args.repre, args.clusterer, eps, args.n_components))
        if args.repre_override is not None:
            os.system(run_cmd.format(args.dataset, args.clusterer, args.basepreproc, args.repre, eps, args.repre_override, save_dir, args.repre_mode, args.cls_mode, args.machine))
        elif args.repre in TRAINED_REPRES[args.dataset]:
            os.system(run_cmd.format(args.dataset, args.clusterer, args.basepreproc, args.repre, eps, saved_repres_dataset[args.repre], save_dir, args.repre_mode, args.cls_mode, args.machine))
        else:
            os.system(run_cmd_no_preload.format(args.dataset, args.clusterer, args.basepreproc, args.repre, eps, save_dir, args.repre_mode, args.cls_mode, args.machine, args.n_components))


def main(args):
    parser = argparse.ArgumentParser(description='RO1 many experiment runner',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str,
                    help='Which dataset to use.')
    parser.add_argument('--clusterer', default='kmeans',
                    choices=['kmeans', 'dbscan', 'hdbscan', 'birch', 'bisect', 'meanshift', 'hac'])
    parser.add_argument('--save-dir', type=str, # default=DEFAULT_SAVE_DIR,
                    help='Max number of iterations of DEC optimization.')
    parser.add_argument('--repre-mode', type=str, default='cpu',
                        choices=['cpu','gpu'],
                        help='Choose b/w cpu and gpu mode.')
    parser.add_argument('--cls-mode', type=str, default='cpu',
                        choices=['cpu','gpu'],
                        help='Choose b/w cpu and gpu mode.')
    parser.add_argument('--repre', type=str,
                        help='Specify the representation that we want to run experiments for. A lot of the time,  \
                        the representation will already be trained in some previous run.')
    parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--repre-override', type=str, default=None)
    parser.add_argument('--n-clusters-override', type=str, default=None)
    parser.add_argument('--eps-override', type=str, default=None)
    parser.add_argument('--machine', type=str, default='kinit',
                        help="Chooses different paths to dataset folders based on what machine we are using.")
    parser.add_argument('--basepreproc', type=int, default=0,
                        help='What kind of base preprocessing should be used -- default (0) means to use \
                            [RobustScaler,StandardScaler, MinMaxScaler]. 1 means to use 0 + QuantileTransformer. \
                            2 means use QuantileTransformer only.')
    parser.add_argument('--hac-clf', type=str, default='logreg', # added on 24.7.2023
        help='For NN representation models such as AE, CVAE, PacMAP. Not for UMAP/TSNE/etc.')
    parser.add_argument('--clf-mode', type=str, default='cpu',
                        choices=['cpu','gpu'],
                        help='Choose b/w cpu and gpu mode -- relevant only when `hac` is chosen as clusterer.')
    parser.add_argument('--feature-sel', type=int, default=None,
                    help='Feature selection for the EMBER dataset.')
    parser.add_argument('--logreg-max-iter', type=int, default=500, help = "For hac-clf context where logistic regression is also the classifier of choice, use the specified " \
    "maximum number of iterations.")
    parser.add_argument('--cols-leaveout', type=int, default=0, help = "Which set of columns to leave out. Possible options: 0,1,2,3. ")
    args = parser.parse_args(args)

    if args.n_clusters_override is not None:
        args.n_clusters_override = [int(c) for c in args.n_clusters_override.split(',')]

    if args.eps_override is not None:
        args.eps_override = [float(c) for c in args.eps_override.split(',')]

    args.repre = args.repre.lower()

    global TRAINED_REPRES
    if args.machine == 'company':
        TRAINED_REPRES = TRAINED_REPRES_COMPANY
    
    # run_cmd = "python -u ro1_main.py --dataset bodmas --clusterer kmeans --repre pca --n-clusters {} " \
    # "--repre-preload results/bodmas/pca_kmeans_10cls_10comp/embedder.pkl --save-dir {}"

    # for kmeans
    n_clusters_for_datasets = {
        'bodmas':[100,500,581,1000,2000],
        'ember':[10,100,500,1000,2000,10000,20000,30000,50000],
        # 'ember':[100,500,1000],
    }

    # eps for DBSCAN
    eps_for_datasets = {
        'bodmas':[0.025, 0.05,0.1,0.2,0.3,0.5, 1,2, 3],
        # 'ember':[0.05,0.1,0.2,0.3,0.5, 1,2, 3],
        'ember':[0.05,0.1,0.2,0.3,0.5, 1,2, 3], # up until 5.9.2023 the least value here was 0.2
    }

    args.dataset = args.dataset.lower()
    n_clusterz = n_clusters_for_datasets[args.dataset]
    n_eps = eps_for_datasets[args.dataset]

    if args.n_clusters_override is not None:
        n_clusterz = args.n_clusters_override

    if args.eps_override is not None:
        n_eps = args.eps_override

    if args.clusterer in ['kmeans', 'bisect', 'birch', 'meanshift', 'hac']:
        center_based(n_clusterz, args)
    elif args.clusterer == 'dbscan':
        dbscan_based(n_eps, args)
    else:
        raise Exception(f'Clusterer {args.clusterer} not yet implemented!')

if __name__ == '__main__':
    main(sys.argv[1:])


