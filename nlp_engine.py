import nltk
from nltk.stem.lancaster import LancasterStemmer # for english
from nltk.stem.snowball import FrenchStemmer # for french
from nltk.corpus import stopwords 
from textblob import TextBlob #to detect the language used => 'fr' or 'en'
from spellchecker import SpellChecker
import random
import numpy as np

"""
takes a sentence(a string) and a Vocabulary(list of word) and return the Bow of this sentence 
"""

class Nlp:
    def __init__(self, vocab=None):
        self.vocab=vocab

    def detect_language(self,sentence):
        if(len(sentence)>3):
            b = TextBlob(sentence)
            return b.detect_language()  
        else:
            return "en"

    def spell_correction(self,tokenized_sentence):
        spell = SpellChecker()
        misspelled=spell.unknown(tokenized_sentence)
        for i in tokenized_sentence:
            if i  in misspelled:
                tokenized_sentence[tokenized_sentence.index(i)]=spell.correction(i)
        return tokenized_sentence

    def tokenization(self,sentence):
        return nltk.word_tokenize(sentence)

    def stopwords_list(self):
        return list(set(stopwords.words('french')))+ list(set(stopwords.words('english')))+["?","!",".",";",","]

    def stemming(self,sentence):
        sentence=sentence.lower()
        """lang=detect_language(sentence)
        stemmer = LancasterStemmer() if lang=="en" else FrenchStemmer()"""
        stemmer=LancasterStemmer()
        return stemmer.stem(sentence)

    def get_ready_for_bow(self,sentence):
        stopwords=self.stopwords_list()
        sentence_tokenized=self.spell_correction(self.tokenization(sentence)) 
        tokenized_sentence_ready = [self.stemming(w) for w in sentence_tokenized if w not in stopwords]
        return tokenized_sentence_ready    

    def bag_of_words(self,sentence):
        bag=[]
        if(self.vocab==None):
            self.vocab = sorted(list(set(self.vocab)))
            vocabualry=list()
            for w in self.vocab:
                vocabualry.extend(self.get_ready_for_bow(w))
            self.vocab=vocabualry  

        tokenized_sentence_ready=self.get_ready_for_bow(sentence)
        for w in self.vocab :
            bag.append(tokenized_sentence_ready.count(w)) 
        
        return bag

