__author__ = 'Arun Bhatia'

class CogEntityExtractor:

    def __init__(self, w2v_model):

        self.w2v_model = w2v_model


    def get_entities(self, query):
        """

        :param query:
        :return:
        """
