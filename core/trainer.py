import _pickle as cPickle
from keras import Sequential, optimizers, callbacks
from keras.layers import Dense, regularizers
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

from core import project_constants
from model_arch.keras_arch import KerasModel
import matplotlib.pyplot as plt

__author__ = 'Arun Bhatia'

tb = callbacks.TensorBoard(histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=False,
                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


class HoloAshTrainer:

    def __init__(self, w2v_model, dims=300):
        self.w2v_model = w2v_model
        self.dims = dims
        self.keras_arch = KerasModel()


    def train(self, data_vector_list):
        """
        Train model using deep learning
        :param data:
        :return:
        """
        # with open(data_vector_path, 'rb+') as pickle_file:
        #     final_data_list = cPickle.load(pickle_file)

        data = np.array(data_vector_list, np.float32)
        X, y = data[:, 0:self.dims], data[:, self.dims]
        #self.__train_deep_learning(X, y)
        self.__train_deep_learning_keras(X, y)


    def __train_deep_learning_keras(self, data, cats):
        """
        Train data using deep learning
        :param data:
        :param cats:
        :return:
        """
        X_train, X_test, y_train, y_test = train_test_split(data, cats, test_size=0.40, random_state=42)
        y_train = to_categorical(y_train)
        print("unique_y:",np.unique(y_train))
        model = self.keras_arch.create_model()
        learning_rate = 0.0001
        epochs = 500
        decay_rate = learning_rate / epochs
        solver = optimizers.adam(lr=learning_rate,
                                 epsilon=1e-8,
                                 decay=decay_rate,
                                 beta_1=0.9,
                                 beta_2=0.999)
        print("summary: ", model.summary())
        model.compile(loss='categorical_crossentropy', optimizer=solver, metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_split=0.15, batch_size=32, shuffle=True, epochs=epochs, verbose=2, callbacks=[tb])

        # list all data in history
        # print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        model.save(project_constants.KERAS_MODEL_PATH)
        print("Saved classifier at: " + project_constants.KERAS_MODEL_PATH)
        #return model