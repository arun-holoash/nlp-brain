import os

__author__ = 'Arun Bhatia'


###### INTENTS #######

SET_REMINDER = 'SET_REMINDER'
GET_REMINDER = 'GET_REMINDER'
MUSIC = 'MUSIC'
OTHER = 'OTHER'

intent_dict = {}

intent_dict[SET_REMINDER] = 0
intent_dict[GET_REMINDER] = 1
intent_dict[MUSIC] = 2
intent_dict[OTHER] = 3


####### Constants ############

QUERY = 'query'
INTENT = 'intent'



######## FILES LOC ##############
MAIN_SKELTON_QUERIES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                    "resources/training_files/skelton_queries.csv"))
DUP_SKELTON_QUERY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                    "resources/training_files/dup_skelton_queries.csv"))

OTHER_SKELTON_QUERIES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                    "resources/training_files/OTHER_intent.csv"))

TRAINING_VECTOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                    "resources/training_files/training_vectors.pkl"))