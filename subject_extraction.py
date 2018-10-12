import os

from langdetect import detect
import nltk
from nltk.corpus import stopwords
import re


from datascience_utils.input_output \
    import load_df



NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']


GRAMMAR = (NOUNS, VERBS)



class _Description(object):

    def __init__(self, data):

        self.data = data

    stop = stopwords.words("english")



    def clean_document(self):

        document = re.sub('[^A-Za-z .-]+',
                          ' ', self.data)


        document = ' '.join(document.split())

        return document


    def tokenizer(self):

        sentences = nltk.sent_tokenize(self.data)
        words = [nltk.word_tokenize(sent) for sent in sentences]

        return sentences, words

    def grams(self, n):

        words = self.tokenizer(self.data)[1]

        ngram = nltk.ngrams(words, n)

        return ngram

    def word_freq_dist(self):

        """Returns a word count frequency distribution"""

        words = self.tokenizer(self.data)
        words = [word.lower() for word in words if word not in stop]
        fdist = nltk.FreqDist(words)

        return fdist


