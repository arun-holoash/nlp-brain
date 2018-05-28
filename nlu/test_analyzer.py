from unittest import TestCase

import en_core_web_md

from core import project_constants
from nlu.analyzer import Analyzer

__author__ = 'Arun Bhatia'


model = en_core_web_md.load()
analyzer = Analyzer(model)

class TestAnalyzer(TestCase):

    def test_analyze_query(self):

        model_path = project_constants.KERAS_MODEL_PATH
        query = 'I have an appointment with Yoshua at 7pm'

        print(analyzer.analyze_query(query=query, model_path=model_path))
