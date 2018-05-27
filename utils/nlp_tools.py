import string
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class NlpTools:
    def __init__(self):
        pass

    def lower_caser(self, sentence):
        """
        Converts into lower case
        @param sentence:
        @return:
        """
        return sentence.strip().lower()

    def remove_punc_alternative(self, sentence):
        """
        Remove punctuations from the sentence - alternative
        @param sentence:
        @return:
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        return regex.sub('', sentence)

    def remove_punc_alternative_with_space(self, sentence):
        """
        Remove punctuations from the sentence and replaces it with a space
        @param sentence:
        @return:
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        return regex.sub(' ', sentence)

    def remove_stop_words(self, sentence):
        """
        Remove stop words
        @param sentence:
        @return:
        """
        stopword_list = set(stopwords.words('english'))  # currently, using stopword list from NLTK
        return ' '.join(str(w) for w in sentence.split() if w not in stopword_list)

    def lemmatize(self, sentence):
        """
        Lemmatize each word of the sentence
        @param sentence:
        @return:
        """
        porter_stemmer = PorterStemmer()
        return ' '.join(porter_stemmer.stem(str(w)) for w in sentence.lower().split())



