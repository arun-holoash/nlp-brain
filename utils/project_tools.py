import csv
from tqdm import tqdm
from pandas import read_csv
import _pickle as cPickle
from core.project_constants import intent_dict
from core.sen2vec import Sentence2Vec
from core import project_constants
import numpy as np

__author__ = 'Arun Bhatia'

class ProjectTools:

    def __init__(self, w2v_model):
        self.s2v = Sentence2Vec(w2v_model)

    def get_all_data_features(self, training_data, cats, training_vector_path):
        """
        Convert training data file into list of vectors
        :param training_data_file:
        :return:
        """

        training_vectors = []
        for data, cat in tqdm(zip(training_data, cats)):
            single_datapoint = self.s2v.get_sen2vec(data)
            training_vectors.append(np.append(single_datapoint, self.get_class_index(cat)))

        with open(training_vector_path, 'wb+') as file_obj:
            cPickle.dump(training_vectors, file_obj)

        return training_vectors

    def get_training_data(self):
        """
        Loads training data from respective files and convert them into vectors
        :return:
        """
        training_data = []
        intent = []
        data = read_csv(project_constants.MAIN_SKELTON_QUERIES_PATH)
        training_data.extend(data[project_constants.QUERY].values)
        intent.extend(data[project_constants.INTENT].values)

        other_data = read_csv(project_constants.OTHER_SKELTON_QUERIES_PATH)
        training_data.extend(other_data[project_constants.QUERY].values)
        intent.extend(other_data[project_constants.INTENT].values)

        #Extra check
        assert len(training_data) == len(intent)

        return training_data, intent

    # def get_sen2vec(self, training_data, cats):
    #     """
    #     prepare sen2vec file from the training_data_file and save it as pkl file
    #     :param training_data:
    #     :param cats
    #     :return:
    #     """
    #     training_vectors = self.get_all_data_features(training_data, cats)
    #
    #     with open(project_constants.TRAINING_VECTOR_PATH, 'wb+') as file_obj:
    #         np.pickle.dump(training_vectors, file_obj)
    #
    #     return training_vectors

    def get_class_index(self, cat):
        """
        Given category name - Returns category index (for training purpose)
        @param cat:
        @return:
        """
        return intent_dict[cat]

    def get_class(self, cat_index):
        """
        Given category index - Returns actual category name
        @param cat_index:
        @return:
        """
        return intent_dict.keys()[intent_dict.values().index(cat_index)]