import numpy
from multiprocessing import Process, Queue

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from keras.datasets import mnist
from keras.datasets import cifar10

from oct2py import octave

def run_in_separate_process(method, args):
    def queue_wrapper(q, params):
        r = method(*params)
        q.put(r)
    q = Queue()
    p = Process(target=queue_wrapper, args=(q, args))
    p.start()
    return_val = q.get()
    p.join()
    if type(return_val) is Exception:
        raise return_val
    return return_val

class MNIST(object):
    def __init__(self):
        self._dim = 4
        self._search_domain = numpy.array([[-6, 0], [16, 512], [50, 100], [0, 1]])
        self._num_init_pts = 1
        self._sample_var = 0.0
        self._min_value = 0.0
        self._num_fidelity = 0
        self._observations = []

    def train(self, x):
        #try:
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (self.X, self.Y), (self.X_valid, self.Y_valid) = mnist.load_data()

        self.X = self.X.reshape(60000, 784)
        self.X_valid = self.X_valid.reshape(10000, 784)

        print('x_train shape:', self.X.shape)
        print(self.X.shape[0], 'train samples')
        print(self.X_valid.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        num_classes = 10
        self.Y = keras.utils.to_categorical(self.Y, num_classes)
        self.Y_valid = keras.utils.to_categorical(self.Y_valid, num_classes)

        lr, batch_size, num_epochs, dropout = x
        lr = pow(10, lr)
        batch_size = int(batch_size)

        K.clear_session()
        sess = tf.Session()
        K.set_session(sess)
        graphr = K.get_session().graph
        with graphr.as_default():
            epochs = int(num_epochs)

            model = Sequential()
            model.add(Dropout(dropout))
            model.add(Dense(num_classes, activation='softmax'))

            # initiate RMSprop optimizer
            opt = keras.optimizers.Adam(lr=lr)

            # Let's train the model using RMSprop
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            self.X = self.X.astype('float32')
            self.X_valid = self.X_valid.astype('float32')
            self.X /= 255
            self.X_valid /= 255

            num_train = int(self.X.shape[0])
            print(num_train)
            indices = numpy.random.choice(self.X.shape[0], num_train)
            X = self.X[indices, :]
            Y = self.Y[indices, :]

            history = model.fit(X, Y,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(self.X_valid, self.Y_valid),
                                shuffle=True)
            val = history.history['val_acc']

            return [100*(1-validation) for validation in val]
            #except Exception as e:
            #    return e

    def evaluate_true(self, x):
        loss = run_in_separate_process(self.train, [x])
        if type(loss) is Exception:
            raise loss
        else:
            print(loss)
            return numpy.array(loss)

    def evaluate(self, x):
        return self.evaluate_true(x)
