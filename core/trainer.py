import _pickle as cPickle
from keras import Sequential, optimizers
from keras.layers import Dense, regularizers
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

__author__ = 'Arun Bhatia'

class HoloAshTrainer:

    def __init__(self, w2v_model, dims=300):
        self.w2v_model = w2v_model
        self.dims = dims


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
        model = Sequential()
        model.add(Dense(30, input_dim=300, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
        model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
        #model.add(Dense(350, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
        model.add(Dense(output_dim=4, activation='softmax', kernel_regularizer=regularizers.l2(0.05)))
        learning_rate = 0.0001
        epochs = 1000
        decay_rate = learning_rate / epochs
        solver = optimizers.adam(lr=learning_rate,
                                 epsilon=1e-8,
                                 decay=decay_rate,
                                 beta_1=0.9,
                                 beta_2=0.999)
        print("summary: ",model.summary())
        model.compile(loss='categorical_crossentropy',optimizer=solver,metrics=['accuracy'])
        model.fit(X_train,y_train, batch_size=3265, shuffle=True, epochs=epochs)
        return model