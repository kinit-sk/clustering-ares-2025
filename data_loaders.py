import numpy as np
import pandas as pd
import pickle
import math
import os
import sys
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import util as ut


# if tensorflow is available, our dataloaders will subclass the Sequence class
# if it's not, we probably don't wanna use tensorflow models to process data, so we are probably okay 
# without subclassing
try:
    from tensorflow.keras.utils import Sequence as Seq
except ImportError:
    Seq = object

class ArtificialDataset():
    """
    Create an artificial clustering dataset for the purpose of testing our methods.
    """
    def __init__(self, n_samples, n_features, centers, test_size=0.2, scalers=None, shuffle=True, random_state=None):
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                    cluster_std=1.0, shuffle=shuffle,random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.train_X = X_train
        self.train_y = y_train
        self.test_X = X_test
        self.test_y = y_test
        if scalers is not None:
            for scaler in self.scalers:
                    self.train_X = scaler.fit_transform(self.train_X)
                    self.test_X = scaler.transform(self.test_X)



class BODMAS(Seq): # tf.keras.utils.Sequence
    """For the BODMAS dataset."""
    # inspiration comes from https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658
    def __init__(self, dataset_path, module_path, train, scalers=None, batch_size = 256, random_order = True, keep_first_X_features = None):
        self.dataset_path = dataset_path
        self.module_path = module_path # needs to point to the parent of the module directory
        self.train = train
        self.scalers = scalers
        self.batch_size = batch_size
        self.random_order = random_order
        self.autoencoder_mode = False
        self.keep_first_X_features = keep_first_X_features
        self.cols_leaveout = [512, 513, 514, 612, 616, 617, 625, 626, 685, 686, 687, 693, 694, 695, 696, 697, 698, 699, 700,
         701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723,
          724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 793, 794, 795, 796,
           797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819,
            820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842,
             1576, 1935, 2131, 2165, 2202, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364,
              2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380]
        ftrs = np.arange(2381)
        self.boolean_leaveout = np.array([False if col in self.cols_leaveout else True for col in ftrs])
        self.initialize_datasets()

    def initialize_datasets(self):
        # import tensorflow as tf
        # load the dataset through the ember library
        # add the ember module on the syspath so python recognizes it
        # sys.path.append(self.module_path)
        # import ember # do we add the syspath to be able to import ember library ? probably so
        if self.train:
            self.X = np.load(os.path.join(self.dataset_path,'X_train.npy'))
            self.y = np.load(os.path.join(self.dataset_path,'y_train.npy'))
        else: # test
            self.X = np.load(os.path.join(self.dataset_path,'X_test.npy'))
            self.y = np.load(os.path.join(self.dataset_path,'y_test.npy'))

        self.X = np.array(self.X).astype(np.float32)
        if len(self.cols_leaveout) > 0: # for performance purposes
            self.X = self.X[:,self.boolean_leaveout]

        # self.len = self.X.shape[0]
        self.len = math.ceil(len(self.X) / self.batch_size)
        if self.scalers is not None:
            for scaler in self.scalers:
                if self.train:
                    self.X = scaler.fit_transform(self.X)
                else:
                    self.X = scaler.transform(self.X)

        if self.random_order:
            print (f'Randomizing BODMAS on initialization ! Self.random_order has value {self.random_order}')
            self.randomize_data()

    def randomize_data(self):
        self.random_sample_order = np.random.permutation(np.arange(len(self.X)))
        self.X = self.X[self.random_sample_order]
        self.y = self.y[self.random_sample_order]

    
    def unpickle(self,path):
        with open(path,'rb') as fil:
            return pickle.load(fil)

    def __getitem__(self, index):
        # if self.X_is_sparse:
        #     sample_X = self.X[index].todense()
        # else:
        #     sample_X = self.X[index]

        # if I don't encapsulate the sample_X in a `np.array` class, it stays `np.matrix` class and that's something that the PyTorch DataLoader does not like
        # return np.array(sample_X, dtype=np.float32), self.y[index]

        batch_x = self.X[index * self.batch_size:(index + 1) *
        self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) *
        self.batch_size]

        # if index + 1 == self.len: # we hit the dataset boundary
        #     # print (f'[EMBER Dataloader] Dataset bounary. index+1 ({index+1}) equals self.len ({self.len})')
        #     if self.random_order: # if we have randomized samples, then randomize them again
        #         print (f'Hit the dataset boundary! Randomizing! Self.random order has value {self.random_order}')
        #         self.randomize_data()

        if self.autoencoder_mode:
            return batch_x, batch_x

        return batch_x, batch_y
        # return torch.from_numpy(sample_X).float(), self.y[index] # torch.Tensor(self.y[index]).long()

    def __len__(self):
        return self.len 

    def _set_len(self, num) :
        self.len = num
        self.X = self.X[:num*self.batch_size]
        self.y = self.y[:num*self.batch_size]

    def set_autoencoder_mode(self, bool_val):
        self.autoencoder = bool_val
    
    def set_random_batch_mode(self, bool_val):
        self.random_order = bool_val

    def get_random_batch_mode(self):
        return self.random_order


class EMBER(Seq): # tf.keras.utils.Sequence
    """For the EMBER dataset."""
    # inspiration comes from https://medium.com/codex/saving-and-loading-transformed-image-tensors-in-pytorch-f37b4daa9658
    def __init__(self, dataset_path, module_path, train, scalers=None, batch_size = 256, random_order = True, keep_first_X_features = None,
                    docker_friendly=False, feature_selection=None, remove_benign=False):
        self.dataset_path = dataset_path
        self.module_path = module_path # needs to point to the parent of the module directory
        self.train = train
        self.scalers = scalers
        self.batch_size = batch_size
        self.random_order = random_order
        self.autoencoder_mode = False
        self.keep_first_X_features = keep_first_X_features
        self.docker_friendly = docker_friendly
        self.feature_selection = feature_selection
        # self.cols_leaveout = []
        self.cols_leaveout = [512, 513, 514, 612, 616, 617, 625, 626, 685, 686, 687, 693, 694, 695, 696, 697, 698, 699, 700,
         701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723,
          724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 793, 794, 795, 796,
           797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819,
            820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842,
             1576, 1935, 2131, 2165, 2202, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364,
              2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380]
        ftrs = np.arange(2381)
        self.boolean_leaveout = np.array([False if col in self.cols_leaveout else True for col in ftrs])
        self.remove_benign = remove_benign
        self.initialize_datasets()

    def load_ember_metadata(self, full_df):
        meta_path = os.path.join(self.dataset_path, 'metadata.csv')
        meta = pd.read_csv(meta_path)
        meta = meta[meta['label'] != -1].drop('Unnamed: 0', axis=1).copy(deep=True)
        all_sha256 = set(meta['sha256'].values)
        found_sha256 = set(full_df['parent_sha256'].unique())
        intrs = all_sha256.intersection(found_sha256)
        found = meta[meta['sha256'].isin(intrs)]
        subset_counts = found['subset'].value_counts()
        print (f'Found {len(intrs)} samples from Ember2018 ({((len(intrs) / len(all_sha256)) * 100):.2f}%).\nTrain set: {subset_counts["train"]}\nTest set: {subset_counts["test"]}')
        # found['label'].value_counts()
        return meta, found


    def initialize_datasets(self):
        # self.X = self.unpickle(self.X_path)
        # self.X_is_sparse = scipy.sparse.issparse(self.X)
        # self.y = self.unpickle(self.y_path)
        # load the dataset through the ember library
        # add the ember module on the syspath so python recognizes it
        sys.path.append(self.module_path)
        subset = 'train' if self.train == True else 'test'
        if not self.docker_friendly:
            import ember # do we add the syspath to be able to import ember library ? probably so
            self.X, self.y = ember.read_vectorized_features(self.dataset_path, subset=subset)

        else: 
            self.X = np.load(f'/mnt/data/martin.mocko/data/Ember/ember2018/{subset}_X.npy')
            self.y = np.load(f'/mnt/data/martin.mocko/data/Ember/ember2018/{subset}_y.npy')

        self.X = np.array(self.X).astype(np.float32)
        # np.save(f'/mnt/data/martin.mocko/data/Ember/ember2018/{subset}_X.npy', self.X)
        # np.save(f'/mnt/data/martin.mocko/data/Ember/ember2018/{subset}_y.npy', self.y)

        # exit()

        if len(self.cols_leaveout) > 0: 
            self.X = self.X[:,self.boolean_leaveout]

        if self.feature_selection is not None:
            # load the order of the most important features
            feature_order = ut.load_pickle(path='results/ember_rf/sorted_indices.pkl')
            if self.feature_selection == 1:
                cur_ftrs = np.arange(self.X.shape[1])
                top_150 = feature_order[:150]
                keep_cols = np.array([True if col in top_150 else False for col in cur_ftrs])
                self.X = self.X[:,keep_cols]
            elif self.feature_selection == 2:
                cur_ftrs = np.arange(self.X.shape[1])
                top_600 = feature_order[:600]
                keep_cols = np.array([True if col in top_600 else False for col in cur_ftrs])
                self.X = self.X[:,keep_cols]
            else:
                raise Exception(f'Feature selection on EMBER with value {self.feature_selection} not unimplemented.')


        # self.len = self.X.shape[0]
        if self.remove_benign:
            print (f'[Ember] Removing samples with benign label from {subset} set.')
            s_y = pd.Series(self.y)
            nonbenign_indices = s_y[s_y != 0].index
            self.X = self.X[nonbenign_indices]
            self.y = self.y[nonbenign_indices]
            print (f'[Ember] Dataset size after benign removal: {self.X.shape[0]}')


        self.len = math.ceil(len(self.X) / self.batch_size)
        if self.scalers is not None:
            for scaler in self.scalers:
                if self.train:
                    self.X = scaler.fit_transform(self.X)
                else:
                    self.X = scaler.transform(self.X)

        if self.random_order:
            self.randomize_data()
        # self.X = torch.from_numpy(self.X).float()
        # print ('Printing the dtype: {}'.format(self.X.dtype))
        # self.y = torch.from_numpy(self.y).long()

    def randomize_data(self):
        self.random_sample_order = np.random.permutation(np.arange(len(self.X)))
        self.X = self.X[self.random_sample_order]
        self.y = self.y[self.random_sample_order]

    
    def unpickle(self,path):
        with open(path,'rb') as fil:
            return pickle.load(fil)

    def __getitem__(self, index):
        # if self.X_is_sparse:
        #     sample_X = self.X[index].todense()
        # else:
        #     sample_X = self.X[index]

        # if I don't encapsulate the sample_X in a `np.array` class, it stays `np.matrix` class and that's something that the PyTorch DataLoader does not like
        # return np.array(sample_X, dtype=np.float32), self.y[index]

        batch_x = self.X[index * self.batch_size:(index + 1) *
        self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) *
        self.batch_size]

        # if index + 1 == self.len: # we hit the dataset boundary
        #     # print (f'[EMBER Dataloader] Dataset bounary. index+1 ({index+1}) equals self.len ({self.len})')
        #     if self.random_order: # if we have randomized samples, then randomize them again
        #         print ('Hit the dataset boundary! Randomizing!')
        #         self.randomize_data()

        if self.autoencoder_mode:
            return batch_x, batch_x

        return batch_x, batch_y
        # return torch.from_numpy(sample_X).float(), self.y[index] # torch.Tensor(self.y[index]).long()

    def __len__(self):
        return self.len 

    def _set_len(self, num) :
        self.len = num
        self.X = self.X[:num*self.batch_size]
        self.y = self.y[:num*self.batch_size]

    def set_autoencoder_mode(self, bool_val):
        self.autoencoder = bool_val
    
    def set_random_batch_mode(self, bool_val):
        self.random_order = bool_val
    
    def get_random_batch_mode(self):
        return self.random_order