import keras
import numpy as np


class housing_dataset():
    """ Housing dataset class to get the housing dataset and preprocess it."""

    def __init__(self):

        self.boston_housing = keras.datasets.boston_housing

    def get_housing_dataset(self):

        (x_train, y_train), (x_test, y_test) = self.boston_housing.load_data()
        order = np.argsort(np.random.random(y_train.shape))
        x_train = x_train[order]
        y_train = y_train[order]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        return x_train, y_train, x_test, y_test