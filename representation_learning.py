
from time import time
import umap
import sklearn
# from sklearn import manifold # import TSNE
from sklearn.decomposition import PCA
 #import TSNE
import os


#################
# Trains various data representations, with the following conditions:
# 1. the input data should already be preprocessed in such a way that is suitable for the methods included here
# 2. representation training should run as fast as possible (gpu-based, parallelized)
##################




def fix_gpu():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    return session



def autoencoder(dims, act='relu', init='glorot_uniform', zero_one_output = False):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    # fix_gpu() # adding it here just to try to see if it fixes the generator error (31.5.23)

    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    if not zero_one_output: # added 1.12.22
        y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    else:
        y = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


def pretrain(ae_model, x=None, y=None, training_generator = None, optimizer='adam', epochs=200, 
             batch_size=256, save_dir='results/temp', args=None, zero_one_output=False):
    import tensorflow as tf
    from tensorflow.keras import callbacks

    dataset = 'unknown'
    if args is not None:
        dataset = args.dataset
    print('...Pretraining...')
    if zero_one_output: 
        loss = 'binary_crossentropy' # not categorical crossentropy
    else:
        loss = 'mse'
    
    print (f'Loss function used for AE training: {loss}')
    ae_model.compile(optimizer=optimizer, loss=loss)

    csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')

    model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join(save_dir,'modelchkpt.keras'),
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)


    cb = [csv_logger, model_checkpoint_callback]

    # begin pretraining
    t0 = time()
    if training_generator is None:
        ae_model.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
    else:
        training_generator.set_autoencoder_mode(True)
        ae_model.fit_generator(generator=training_generator,
                validation_data=None,
                use_multiprocessing=True,
                workers=2)

    print('Pretraining time: %ds' % round(time() - t0))
    ae_model.save_weights(os.path.join(save_dir, 'ae.weights.h5'))
    # save the whole model as well
    save_path = os.path.join(save_dir, 'ae_model.h5')
    tf.keras.models.save_model(ae_model, save_path)

def load_ae_weights(model, weights):  # load weights of DEC model
    model.load_weights(weights)

def extract_features(self, x):
    return self.encoder.predict(x)


def train_pca(input_data, n_components, mode='cpu', random_state=None):
    # in this case, we should probably specify n_components as a float, which should keep the -float- value of variance in the data
    # another option, n_components='mle' --> a Minka’s MLE is used to guess the dimension
    if mode == 'cpu':
        pca = PCA(n_components=n_components, random_state = random_state)
    elif mode == 'gpu':
        import cuml
        pca = cuml.PCA(n_components=n_components, random_state = random_state)
    else:
        raise Exception(f'Specified an unsupported mode of the TSNE algorithm: {mode}')
    embedding = pca.fit_transform(input_data)
    return embedding, pca

def train_umap(input_data, n_components, n_neighbors, low_memory=False, mode='cpu', random_state=None):
    """"
    n_neighbors:  default value of n_neighbors for UMAP (as used above) is 15, 
        with n_neighbors=2 we see that UMAP merely glues together small chains, but due to the narrow/local view, fails to see how those connect together. 
    low_memory : works only for cpu mode.. cuml does not seem to have this feature implemented
    """
    if mode == 'cpu':
        init = 'spectral'
        if low_memory == True:
            init = 'random'
        umap_inst = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,  init=init, low_memory=low_memory, random_state=random_state) 
    elif mode == 'gpu':
        # import cuml
        from cuml.manifold import UMAP as cUMAP
        # from cuml import UMAP
        init = 'random'
        # if low_memory:
        #     init = 'random'
        umap_inst = cUMAP(n_neighbors=n_neighbors, n_components=n_components, init=init, random_state=random_state) 
    else:
        raise Exception(f'Specified an unsupported mode of the TSNE algorithm: {mode}')
    
    embedding = umap_inst.fit_transform(input_data)
    return embedding, umap_inst


def train_tsne(input_data, n_components, perplexity, init='pca', mode='cpu'):
    """
    init : sklearn version supports both PCA and random initialization. cuML supports only random initialization.
    """
    if mode == 'cpu':
        tsne = sklearn.manifold.TSNE(n_components=n_components, perplexity=perplexity, init=init)
    elif mode == 'gpu':
        import cuml
        tsne = cuml.TSNE(n_components=n_components, perplexity=perplexity, init=init)
    else:
        raise Exception(f'Specified an unsupported mode of the TSNE algorithm: {mode}')

    embedding = tsne.fit_transform(input_data)
    return embedding, tsne
