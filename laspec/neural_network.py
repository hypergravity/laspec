import os

import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from collections import Iterable

__all__ = ["NN", "optimizers"]


def create_nn_regressor(ninput=4, nhidden=10, noutput=1, 
                        activation_hidden="sigmoid", activation_output=None):
    """ An easy way of creating DNN with 2 dense layers (1 hidden layer)

    Parameters
    ----------
    ninput:
        input shape
    nhidden: tuple
        number of neurons in dense layers
    noutput:
        output shape
    activation_hidden:
        the activation function used in hidden layers
    activation_output:
        the activation function used in output layers

    Returns
    -------

    """
    model = Sequential()
    model.add(Input(shape=(ninput,),))
    # model.add(Dropout(dropout_rate))
    # model.add(BatchNormalization())
    if isinstance(nhidden, Iterable):
        assert len(nhidden) == len(activation_hidden)
        for i, _ in enumerate(nhidden):
            model.add(Dense(_, activation=activation_hidden[i]))
            # model.add(BatchNormalization())
    else:
        model.add(Dense(nhidden, activation=activation_hidden))
        # model.add(BatchNormalization())
    # model.add(Dropout(dropout_rate))
    model.add(Dense(noutput, activation=activation_output))
    return model


def test_create_nn_regressor():
    model = create_nn_regressor(4, 20, 1)
    model.build()
    model.summary()
    model(tf.ones((10, 4)))
    print(model.weights)


def create_c3nn2_classifier(ninput=100, nfilters=32, kernel_size=4, ndense=(128, 16), pool_size=2, dropout_rate=0.5,
                            noutput=1, activation_hidden="relu", activation_out="sigmoid"):
    """ An easy way of creating a CNN with 3 convolutional layers and 2 dense layers

    Parameters
    ----------
    ninput:
        input shape
    nfilters:
        number of filters
    kernel_size:
        kernel size
    ndense: tuple
        number of neurons in dense layers
    pool_size:
        pool size in MaxPooling
    dropout_rate:
        dropout rate
    noutput:
        output shape
    activation_hidden:
        the activation function used in hidden layers
    activation_out:
        the activation function used in output layers

    Returns
    -------

    """
    model = Sequential()
    model.add(
        Conv1D(filters=nfilters, kernel_size=kernel_size, strides=1, padding="valid", activation=activation_hidden,
               input_shape=(ninput, 1)))
    # ,data_format="channels_last"
    model.add(MaxPooling1D(pool_size, padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv1D(nfilters, kernel_size, padding="valid", activation=activation_hidden))
    model.add(MaxPooling1D(pool_size, padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv1D(nfilters, kernel_size, padding="valid", activation=activation_hidden))
    model.add(MaxPooling1D(pool_size, padding="valid"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(ndense[0], activation=activation_hidden, ))  # input_shape=(4000,)
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(ndense[1], activation=activation_hidden))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(Dense(noutput, activation=activation_out))
    return model


class NN:
    model = None
    filepath = ""
    callbacks_kwargs = {}
    callbacks_list = []
    history = None

    xtrain = None
    xtest = None
    ytrain = None
    ytest = None
    swtrain = None
    swtest = None

    def __init__(self, kind="c3nn2", ninput=100, *args, **kwargs):
        assert kind in ["nn", "c3nn2"]
        if kind == "c3nn2":
            # a fast way of creating c3nn2 classifier
            self.model = create_c3nn2_classifier(ninput=ninput, *args, **kwargs)
            print("@NN: generating fast c3nn2 with ninput={}".format(ninput))
        elif kind == "nn":
            # a fast way of creating nn regressor
            self.model = create_nn_regressor(ninput=ninput, *args, **kwargs)
            print("@NN: generating fast nn with ninput={}".format(ninput))
        else:
            raise ValueError("Bad value for *kind*!")

        # default device
        self.get_gpu()

        # default callbacks
        self.set_callbacks()
        return

    def set_callbacks(self, monitor_earlystopping="val_loss", patience_earlystopping=5,
                      monitor_modelcheckpoint="val_loss", filepath="",
                      monitor_reducelronplateau="val_loss", patience_reducelronplateau=2, factor_reducelronplateau=0.33,
                      ):
        """ set callbacks """
        self.filepath = filepath
        self.callbacks_kwargs = dict(monitor_earlystopping=monitor_earlystopping,
                                     patience_earlystopping=patience_earlystopping,
                                     monitor_modelcheckpoint=monitor_modelcheckpoint,
                                     filepath=filepath,
                                     monitor_reducelronplateau=monitor_reducelronplateau,
                                     patience_reducelronplateau=patience_reducelronplateau,
                                     factor_reducelronplateau=factor_reducelronplateau)
        self.callbacks_list = [
            # This callback will interrupt training when we have stopped improving
            keras.callbacks.EarlyStopping(
                # This callback will monitor the validation accuracy of the model
                monitor=monitor_earlystopping,
                # Training will be interrupted when the accuracy
                # has stopped improving for *more* than 1 epochs (i.e. 2 epochs)
                patience=patience_earlystopping,
            ),
            # This callback will save the current weights after every epoch
            keras.callbacks.ModelCheckpoint(
                filepath=filepath,  # Path to the destination model file
                # The two arguments below mean that we will not overwrite the
                # model file unless `val_loss` has improved, which
                # allows us to keep the best model every seen during training.
                monitor=monitor_modelcheckpoint,
                save_best_only=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                # This callback will monitor the validation loss of the model
                monitor=monitor_reducelronplateau,
                # It will divide the learning by 10 when it gets triggered
                factor=factor_reducelronplateau,
                # It will get triggered after the validation loss has stopped improving
                # for at least 10 epochs
                patience=patience_reducelronplateau,
            )
        ]
        return

    def train(self, x, y, sw, test_size=0.2, random_state=0, epochs=200, batch_size=256,
              optimizer=optimizers.Adam(lr=1e-5), loss="binary_crossentropy", metrics=['accuracy'],
              filepath=None):
        # a quick way to set filepath
        if filepath is not None:
            self.callbacks_kwargs.update({"filepath": filepath})
            self.set_callbacks(**self.callbacks_kwargs)

        # split sample
        xtrain, xtest, ytrain, ytest, swtrain, swtest = train_test_split(
            x, y, sw, test_size=test_size, random_state=random_state)
        print("@NN: Split data to training set [{}] and test set [{}]!".format(xtrain.shape[0], xtest.shape[0]))
        # store training and test data
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.swtrain = swtrain
        self.swtest = swtest

        # compile optimizer
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, )

        # train model
        self.history = self.model.fit(
            xtrain, ytrain, sample_weight=swtrain,
            batch_size=batch_size, epochs=epochs, callbacks=self.callbacks_list,
            validation_data=(xtest, ytest, swtest))

        # reload the best model
        self.model = keras.models.load_model(self.filepath)

        return self.history

    @staticmethod
    def set_gpu(device=0):
        """ set gpu device """
        old_device = NN.get_gpu(verbose=False)
        if isinstance(device, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = "{:d}".format(device)
        elif isinstance(device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        new_device = NN.get_gpu(verbose=False)
        print("Changing device {} to {}".format(old_device, new_device))
        return

    @staticmethod
    def get_gpu(verbose=True):
        """ get gpu device """
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            old_device = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            old_device = None
        if verbose:
            print("Current device is {} ".format(old_device))
        return old_device

    def save(self, filepath="", verbose=True):
        """ dump NN object """
        if not filepath == "":
            pass
        elif filepath == "" and not self.filepath == "":
            filepath = self.filepath
        else:
            raise ValueError("@NN: Bad value for filepath!")
        if verbose:
            print("@NN: save NN model to {}, save meta to {}...".format(filepath, filepath + ".nn"))
        self.model.save(filepath)
        self.model = None
        self.clear_dataset()
        self.callbacks_list = []
        self.history = None
        joblib.dump(self, filepath + ".nn")
        return

    @staticmethod
    def load(filepath):
        """ load NN object """
        assert os.path.exists(filepath)
        assert os.path.exists(filepath + ".nn")
        nn = joblib.load(filepath + ".nn")
        nn.model = load_model(filepath)
        nn.set_callbacks(**nn.callbacks_kwargs)
        return nn

    def predict(self, *args, **kwargs):
        """ an alias for model.predict """
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """ an alias for model.evaluate """
        return self.model.evaluate(*args, **kwargs)

    def summary(self, *args, **kwargs):
        """ an alias for model.summary """
        return self.model.summary(*args, **kwargs)

    @staticmethod
    def train_test_split(*arrays, train_size=None, test_size=None, random_state=None):
        return train_test_split(*arrays, train_size=train_size, test_size=test_size, random_state=random_state)

    def clear_dataset(self):
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.swtrain = None
        self.swtest = None
        return

    @staticmethod
    def set_memory(frac=0.1):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1,
                                          allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = frac
        sess = tf.compat.v1.Session(config=config)
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        return


#%%
def test():
#%%
    import numpy as np
    ntest = 1000
    ndim = 4
    x = np.random.uniform(0.2, 0.8, (ntest, ndim))
    y = x[:, 0] + x[:, 1] * 0.5
    y = np.sin(x[:, 0]*50) + x[:, 1] * 0.5
    y = (x[:,0]-0.5)**2
    y = (y-y.min())/(y.max()-y.min())*0.6+0.2 + np.random.randn(ntest)*0.1
    # y *=x[:,0]
    
    #%%
    import numpy as np
    ntest = 1000
    ndim = 4
    x = np.random.uniform(0.2, 0.8, (ntest, ndim))
    y = x[:, 0] + x[:, 1] * 0.5
    y = np.sin(x[:, 0]*20) + x[:, 1] * 0.5
    # y = (x[:,0]-0.5)**2
    y = (y-y.min())/(y.max()-y.min())*0.6+0.2 + np.random.randn(ntest)*0.02
    # y *=x[:,0]
    #%%
    # x = x.reshape(*x.shape, 1)
    # y = y.reshape(*y.shape, 1)

    nn = NN(kind="nn", ninput=1, nhidden=(100,50), noutput=1, activation_hidden=("relu", "relu"), activation_output="tanh")
    nn.set_callbacks(patience_earlystopping=100, patience_reducelronplateau=100, filepath="/tmp/cvsearch.h5")
    nn.train(x[:,0], y, y*0+1, test_size=0.1, epochs=10000, batch_size=np.int(x.shape[0]/20), 
             loss="mse", metrics="mae", optimizer=optimizers.Adam(lr=1e-1))
    nn.model = load_model(filepath="/tmp/cvsearch.h5")
    nn.predict(x[:,0])
    
    figure()
    plot(x[:,0], y,'.')
    plot(x[:,0], nn.predict(x[:,0]),'.')
#%%
from sklearn.neural_network import MLPRegressor
m = MLPRegressor(hidden_layer_sizes=(100,100), activation="logistic", solver="sgd",learning_rate="adaptive", learning_rate_init=1e-5,tol=0, max_iter=10000, verbose=True)
m.fit(x, y)
figure()
plot(x[:,0], y,'.')
plot(x[:,0], m.predict(x),'.')

#%%
from sklearn.svm import SVR
m = SVR(C=50, epsilon=0.05,gamma=100)
m.fit(x, y)
m.predict(x)
figure()
plot(x[:,0], y,'.')
plot(x[:,0], m.predict(x),'.')

#%%
from sklearn.gaussian_process import GaussianProcessRegressor

m = GaussianProcessRegressor(kernel=None,alpha=1e-10, n_restarts_optimizer=10)
m.fit(x, y)
figure()
plot(x[:,0], y,'.')
plot(x[:,0], m.predict(x),'.')

#%%
from sklearn.kernel_ridge import KernelRidge
k = KernelRidge(kernel="rbf",alpha=1e-2, gamma=100)
k.fit(x, y)
k.predict(x)
figure()
plot(x[:,0], y,'.')
plot(x[:,0], k.predict(x),'.')

#%%
figure()
plot(x[:,1], y,'.')
plot(x[:,1], nn.predict(x),'.')

#%%
def cvserach(x, y, sw=None, test_size=0.1,epochs=200, batch_size=50, 
             loss="mse", metrics="mae", optimizer=optimizers.Adam(lr=1e-3), 
             cv=5, nhidden_grid=[5, 10, 20]):
    
    import numpy as np
    x = np.random.uniform(-0.5, 0.5, (1000, 2))
    y = x[:, 0] + x[:, 1] * 0.5
    
    if sw is None:
        sw = np.ones_like(y, dtype=float)
        
    loss_values = np.zeros((len(nhidden_grid), cv))
    
    kf = KFold(n_splits=cv, shuffle=True)
    
    for i_split, (train_index, test_index) in enumerate(kf.split(x)):
        xtrain, xtest = x[train_index], x[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        swtrain, swtest = sw[train_index], sw[test_index]
        
        for i_nhidden, nhidden in enumerate(nhidden_grid):
            nn = NN(kind="nn", ninput=2, nhidden=nhidden, noutput=1)
            nn.train(xtrain, ytrain, ytrain*0+1, test_size=0.1,epochs=200, batch_size=50, 
                     loss="mse", metrics=None, optimizer=optimizers.SGD(lr=1e-2))
            loss_values[i_nhidden, i_split] = nn.evaluate(xtest, ytest)
            print(loss_values)
            
        
    
    
#%%
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Activation
# from keras.optimizers import adam, rmsprop, adadelta
# import numpy as np
# import matplotlib.pyplot as plt
#construct model

models = Sequential()
models.add(Dense(100, activation='relu' ,input_dim=1))
models.add(Dense(50, activation='relu'))
models.add(Dense(1,activation='tanh'))
models.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )

#train data
dataX = np.linspace(-2 * np.pi,2 * np.pi, 1000)
dataX = np.reshape(dataX, [dataX.__len__(), 1])
noise = np.random.rand(dataX.__len__(), 1) * 0.1
dataY = np.sin(dataX) + noise

models.fit(dataX, dataY, epochs=100, batch_size=10, shuffle=True, verbose = 1)
predictY = models.predict(dataX, batch_size=1)
score = models.evaluate(dataX, dataY, batch_size=10)

print(score)
#plot
fig, ax = plt.subplots()
ax.plot(dataX, dataY, 'b-')
ax.plot(dataX, predictY, 'r.',)

ax.set(xlabel="x", ylabel="y=f(x)", title="y = sin(x),red:predict data,bule:true data")
ax.grid(True)

plt.show()