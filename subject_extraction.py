import os
import numpy

from langdetect import detect
import nltk
from nltk.corpus import stopwords
import re

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']


GRAMMAR = (NOUNS, VERBS)


class _Description(object):

    def __init__(self, data):

        self.data = data

        self.stop = stopwords.words("english")

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

        words = self.tokenizer()[1]

        ngram = nltk.ngrams(words, n)

        return ngram

    def get_entities(self):

        """Returns Named Entities using NLTK Chunking"""
        entities = []
        sentences = self.tokenizer()[1]

        # Part of Speech Tagging (just english)
        # TODO Spanish Tagger
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        for tagged_sentence in sentences:
            for chunk in nltk.ne_chunk(tagged_sentence):
                if type(chunk) == nltk.tree.Tree:
                    entities.append(' '.join([c[0] for c in chunk]).lower())
        return sentences

    def word_freq_dist(self):

        """Returns a word count frequency distribution"""

        words = nltk.tokenize.word_tokenize(self.data)
        words = [word.lower() for word in words if word not in self.stop]
        fdist = nltk.FreqDist(words)

        return fdist

    def extract_subject(self):

        # Get most frequent Nouns
        fdist = self.word_freq_dist()
        most_freq_nouns = [w for w, c in fdist.most_common(10)
                           if nltk.pos_tag([w])[0][1] in NOUNS]

        # Get Top 10 entities
        entities = self.get_entities()[0]
        top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]

        # Get the subject noun by looking at the intersection of top 10 entities
        # and most frequent nouns. It takes the first element in the list
        subject_nouns = [entity for entity in top_10_entities
                         if entity[0] in most_freq_nouns]
        print(subject_nouns)
        return subject_nouns[0]







