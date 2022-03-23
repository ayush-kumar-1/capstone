#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:55:09 2022

@author: elisekarinshak
"""

import pandas as pd
import numpy as np
import contractions
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')


def main(): 
    tweets = pd.read_csv("~/Desktop/STAT4990/Final Project/Provided information/cyberbullying_tweets.csv")

    #TEST----------------------------------------------------------------------

    text_sample = '''    I cant go   to the gYm today, I'm tired lol :( gr8'''
    out = to_lowercase(text_sample)  
    out = expand_contractions(out)
    out = remove_special(out)                                                                 
    out_words = word_tokenize(out)
    out_words = remove_stopwords(out_words)   
    out_words = lemmatization(out_words)      
    print(out_words)

    
    #APPLY TO DATASET ---------------------------------------------------------

    for i in range(0, len(tweets["tweet_text"])):
        tweets.iloc[i, 0] =  preprocess(tweets.iloc[i, 0])

#before tokenization:
#capitalization, contractions, symbols, white spaces
#after tokenizarion:
#stop words, stemming, lemmatization

#PREPROCESSING METHODS
#implementations of common NLP preprocessing methods


#METHODS BEFORE TOKENIZATION -----------------------------------------------

#Lower Case
def to_lowercase(text):
    return text.lower()

#Expand Contractions
def expand_contractions(text):
    return contractions.fix(text)

#Remove punctuation, symbols, numbers
special_chars = '''`~!@#$%^&*()-+-={}[]|\:;'<>,.?/"0123456789'''
def remove_special(text):
    for char in text:
        if char in special_chars:
            text = text.replace(char, "")

    return text

#METHODS AFTER TOKENIZATION -----------------------------------------------

#Remove Stopwords
def remove_stopwords(text_arr):
    stops = set(stopwords.words('english'))
    stops |= set(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's']) #additional stopwords
    
    filtered = []
    for word in text_arr:
        if word not in stops:
            filtered = np.append(filtered, word)

    return filtered

#Stemming 
ps = PorterStemmer()

def stemming(text_arr):
    word_stems = []
    for word in text_arr:
        word_stems.append(ps.stem(word))
    
    return(word_stems)

#Lemmatization
lm = WordNetLemmatizer()

def lemmatization(text_arr):
    word_lem = []
    for word in text_arr:
        word_lem.append(lm.lemmatize(word))
    
    return(word_lem)


#SUMMARY METHOD -----------------------------------------------------------

def preprocess(text):
    out = to_lowercase(text)  
    out = expand_contractions(out)
    out = remove_special(out)                                                                 
    out_words = word_tokenize(out)
    out_words = remove_stopwords(out_words)   
    out_words = lemmatization(out_words)      
    
    return out_words

def preprocess_all(tweets):
    return tweets.apply(preprocess)

if __name__ == "__main__":
    main()