import json
import numpy as np
import time
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.nlp_tools import NlpTools
import os

__author__ = 'arun_bhatia'

"""
Class which converts a single sentence to vector using two key factors:
1. Smooth Inverse Frequency
2. Common component removal

"""

nlp_tools = NlpTools()

# To suppress warnings raised due to divide by zero or nan
np.seterr(divide='ignore', invalid='ignore')

class Sentence2Vec:

    def __init__(self, w2v_model, probabilty_data = "word_probabilities.json", alpha=1e-3, w2v_dimension = 300):
        self.w2v_model = w2v_model
        self.alpha = alpha
        self.w2v_dimension = w2v_dimension
        self._load_prob_words(probabilty_data)



    def _load_prob_words(self, data):
        """
        Load probability of all the words in computed on large corpus
        Note - Place all the probability JSON files on the /resources level only and just pass the dataset name
        with ".json"
        :return:
        """

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources/core_files/" + data))) as data_file:
            self.prob_words_dict = json.load(data_file)


    def get_sen2vec_optimized(self, list_of_sen, max_len=12):
        """
        Accepts list of sentences and convert all of them at once to vector form
        :param list_of_sen:
        :param max_len: Upper limit on max numbers of words per sentence. Ignores, the rest!
        :return:
        """

        total_inputs = len(list_of_sen)
        params = self.__get_params(list_of_sen, max_len, total_inputs)

        elem_wise_sum = np.zeros((total_inputs, self.w2v_dimension))
        for i in range(len(params)/2):
            elem_wise_sum += np.sum(np.multiply(params['W' + str(i)], params['beta' + str(i)]))

        raw_result = np.divide(elem_wise_sum, np.asarray([len(l) for l in list_of_sen]).reshape(total_inputs, 1))

        assert raw_result.shape[0] == total_inputs
        assert raw_result.shape[1] == self.w2v_dimension

        transpose_raw_result = raw_result.T

        U, S, V = np.linalg.svd(transpose_raw_result)

        result_vec_matrix = np.subtract(transpose_raw_result, np.dot(U, np.dot(U.T, transpose_raw_result)))

        #log.info('Shape of result sen2vector matrix: {0}'.format(result_vec_matrix.T.shape))
        return result_vec_matrix.T



    def __get_params(self, list_of_sen, max_len, total_inputs):
        """
        Calculate params needed for numpy operations
        :return:
        """
        t1 = time.time()
        list_of_list = [sen.split(' ') for sen in list_of_sen]
        prob_matrix = np.zeros((total_inputs, self.w2v_dimension))
        w2v_matrix = np.zeros((total_inputs, max_len * self.w2v_dimension))
        for sen_idx, sen_tokens in tqdm(enumerate(list_of_list)):
            appended_sen_vec = []
            total_words = len(sen_tokens)
            for token_idx in range(max_len):
                if token_idx < len(sen_tokens):
                    word = sen_tokens[token_idx]
                    appended_sen_vec.extend(self.w2v_model(word.decode('utf-8')).vector)
                    if word in self.prob_words_dict.keys():
                        prob_matrix[sen_idx][token_idx] = self.prob_words_dict[word]
                appended_sen_vec.extend([0.0] * ((max_len - total_words) * self.w2v_dimension))
            w2v_matrix[sen_idx] = appended_sen_vec

        #print('Shape of w2v matrix: ({0},{1})'.format(len(w2v_matrix), len(w2v_matrix[0])))

        #log.info("Total time taken to calculate matrix: {0} seconds".format(time.time() - t1))
        params = {}
        t1 = time.time()
        for word_idx in tqdm(range(max_len)):
            start_col = word_idx * self.w2v_dimension
            end_col = start_col + (self.w2v_dimension)
            params['W' + str(word_idx)] = w2v_matrix[:, start_col:end_col]
            params['beta' + str(word_idx)] = self.__calculate_beta_vector(prob_matrix[:, word_idx], total_inputs)


        #log.info("Total time taken to calculate params: {0} seconds".format(time.time() - t1))
        return params


    def __calculate_beta_vector(self, prob_word_vector, total_inputs):
        """
        Takes 1-D vector of p(word) per sentence
        Calculates beta = alpha / (alpha + prob_word)
        :param word:
        :return:
        """
        return np.divide(self.alpha, np.add(prob_word_vector, self.alpha)).reshape(total_inputs, 1)

    def get_sen2vec(self, sentence):
        """
        Given a sentence - performs two actions:
        1. Converts each word into vector and calculates SIF
        2. Use PCA to remove common components
        :param sentence:
        :return:

        Reference
        ----------
        Implements paper - A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
        See - https://openreview.net/pdf?id=SyK00v5xx
        """
        vs = np.zeros(self.w2v_dimension, dtype=np.float64)
        if len(sentence.strip()) == 0:
            return vs
        try:
            for word in nlp_tools.lower_caser(nlp_tools.remove_punc_alternative(sentence)).split(" "):
                if word not in self.prob_words_dict.keys():
                    pass
                else:
                    # smooth inverse frequency
                    sif = np.divide(self.alpha, np.add(self.alpha, self.prob_words_dict[word]))
                    vs = np.add(vs, np.multiply(sif,  self.w2v_model(word).vector))
                    #vs = np.add(vs, np.multiply(sif,  self.w2v_model.get(word)))
            vs_final = np.divide(vs, len(sentence.split(' ')))
            assert(len(vs_final) == self.w2v_dimension)
            # calculate PCA
            pca = PCA(n_components=self.w2v_dimension)
            pca.fit(vs_final.reshape(1, -1))
            # the PCA vector
            u = pca.components_[0]
            assert(len(u) == self.w2v_dimension)
            return np.subtract(vs_final, np.dot(u, np.dot(u.T, vs_final)))
        except Exception as ex:
            print("Exception while calculating sentence2vec: " + str(ex))
            return vs