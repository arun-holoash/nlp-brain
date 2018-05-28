import en_core_web_md

from core import project_constants
from core.trainer import HoloAshTrainer
from utils.project_tools import ProjectTools
import _pickle as cPickle

__author__ = 'Arun Bhatia'

w2v_model = en_core_web_md.load()

trainer = HoloAshTrainer(w2v_model=w2v_model)
nlp_tools = ProjectTools(w2v_model=w2v_model)

RETRAIN = True

if not RETRAIN:
    data, cats = nlp_tools.get_training_data()
    data_vector_list = nlp_tools.get_all_data_features(data, cats,
                                                       project_constants.TRAINING_VECTOR_PATH)
else:
    with open(project_constants.TRAINING_VECTOR_PATH, 'rb+') as pickle_file:
        data_vector_list = cPickle.load(pickle_file)

trainer.train(data_vector_list)




