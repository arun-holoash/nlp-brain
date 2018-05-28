import copy

import numpy as np

from core import project_constants
from sentiment_analyser.sentiment_analyzer import SentimentAnalyser
from utils.project_tools import ProjectTools

__author__ = 'Arun Bhatia'


class Analyzer:

    def __init__(self, w2v_model):

        self.w2v_model = w2v_model
        self.sentiment_analyser = SentimentAnalyser()
        self.tools = ProjectTools(w2v_model)


    def analyze_query(self, query, model_path):
        """
        Loads model from model_path and returns intent of the query
        :param model_path:
        :return:
        """

        result = copy.deepcopy(project_constants.QUERY_RESULT_SKELTON)

        result.update({project_constants.INTENT: self.predict_category(query, model_path=model_path)})
        result.update({project_constants.ENTITIES: self.extract_entities(query)})
        result.update({project_constants.SENTIMENT: self.sentiment_analyser.extract(query).sentiment})

        return result


    def extract_entities(self, query):
        """
        Fetch entities from query
        :param entity_path:
        :param query:
        :return:
        """
        entities = []
        ee = CogEntityExtractor(self.w2v_model)
        entities.append(ee.extract_entities(entity_path, query))
        return entities


    def predict_category(self, query, model_path):
        """
        Predicts the class for the given sentence using model loaded from model_path
        :param sentence:
        :return:
        """
        try:
            model = self.tools.load_model(model_path=model_path)
            response = self.tools.get_single_data_vector(query)
            predicted_intent = model.predict_classes(np.array([response]))
            print(predicted_intent[0])
            return self.tools.get_class(predicted_intent[0]) if predicted_intent is not None else None
        except Exception as err:
            print(err)
            return None
