from keras import Sequential
from keras.layers import Dense, Dropout, regularizers, LSTM

from core import project_constants

__author__ = 'Arun Bhatia'



class KerasModel:

    def __init__(self):
        pass

    def create_model(self):
        """
        Create and Returns the model architecture
        :return:
        """
        model = Sequential()
        model.add(Dense(10, input_dim=project_constants.DIMS, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.3))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(256, activation='relu'))
        #model.add(Dropout(0.2))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        #model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(units=4, activation='sigmoid'))

        return model