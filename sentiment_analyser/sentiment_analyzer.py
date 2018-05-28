from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sentiment_analyser.constants as SENTIMENT


class SentimentResult:
    def __init__(self, score):
        """
        Helper class to SentimentAnalyser
        :param score: float
        """
        self.score = score
        self.sentiment = self.classify

    @property
    def classify(self):
        """
        Two level ternary operator
        :param score: float
        :rtype: str
        """
        return SENTIMENT.POSITIVE \
            if self.score[SENTIMENT.ATTR_COMPOUND] > 0 else \
            SENTIMENT.NEGATIVE if self.score[SENTIMENT.ATTR_COMPOUND] < 0 else SENTIMENT.NEUTRAL

    def to_dict(self):
        """
        Converts object to dictionary for easy transport over HTTP
        :rtype: dict
        """
        return {
            SENTIMENT.ATTR_SCORE: self.score[SENTIMENT.ATTR_COMPOUND],
            SENTIMENT._: self.sentiment
        }


class SentimentAnalyser:
    """
    Implements Vader Sentiment Analyzer
    https://github.com/cjhutto/vaderSentiment
    """

    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()
        self.sentiments = None

    def extract(self, sentence):
        return SentimentResult(self.analyser.polarity_scores(sentence))

