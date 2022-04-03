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
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk_resources = ["punkt", "stopwords", "omw-1.4", "wordnet"]
for resource in nltk_resources:
    try:
        nltk.data.find("C:/Users/Ayush/AppData/Roaming/nltk_data/" + resource)
    except LookupError: 
        nltk.download(resource)

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
def to_lowercase(text_arr: pd.Series) -> pd.Series:
    """
    Converts all text to lowercase. 

    26.6 ms ± 772 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    return text_arr.str.lower()

#Remove all emojis
def remove_emoji(text_arr: pd.Series) -> pd.Series:
    """
    Removes all emojis from text.

    172 ms ± 2.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return text_arr.str.replace(emoji_pattern, "")

#Expand Contractions
def expand_contractions(text_arr: pd.Series) -> pd.Series:
    return text_arr.apply(contractions.fix)

def remove_ats(text_arr): 
    at_regex = r"@[A-Za-z0-9]+"
    return text_arr.str.replace(at_regex, "")


#Remove punctuation, symbols, numbers
def remove_special(text_arr: pd.Series) -> pd.Series:
    # special_chars = '''`~|!|@|#|\$%^&*\(\)-+-=\{\}[]\|\\:;'<>,.?/"0123456789'''
    special_chars = r"[^a-zA-Z\s]"
    return text_arr.str.replace(special_chars, "")

#METHODS AFTER TOKENIZATION -----------------------------------------------

#Remove Stopwords
def remove_stopwords(text_arr: pd.Series, stops: set = set(stopwords.words('english'))) -> pd.Series:
    #additional twitter stopwords
    stops |= set(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's']) #additional stopwords
    
    return text_arr.str.split().apply(lambda row: [word for word in row if word not in stops]).str.join(' ') 

#Stemming 
ps = PorterStemmer()

def stemming(text_arr):
    word_stems = []
    for word in text_arr:
        word_stems.append(ps.stem(word))
    
    return(word_stems)

def stem_list(lst, stemmer): 
    results = map(stemmer.stem, lst)
    return list(results)

def stem_series(text_arr: pd.Series, stemmer = PorterStemmer()) -> pd.Series:
    text_arr = text_arr.str.split()
    return text_arr.apply(lambda row: stem_list(row, stemmer)).str.join(' ')

#Lemmatization
lm = WordNetLemmatizer()

def lemmatization(text_arr):
    word_lem = []
    for word in text_arr:
        word_lem.append(lm.lemmatize(word))
    
    return(word_lem)

def lemmatize_list(lst, lemmatizer): 
    results = map(lemmatizer.lemmatize, lst)
    return list(results)

def lemmatize_series(text_arr: pd.Series, lemmatizer = WordNetLemmatizer()) -> pd.Series:
    text_arr = text_arr.str.split()
    return text_arr.apply(lambda row: lemmatize_list(row, lemmatizer)).str.join(' ')

#SUMMARY METHOD -----------------------------------------------------------

def preprocess(text):
    out = to_lowercase(text)  
    out = expand_contractions(out)
    out = remove_special(out)                                                                 
    out_words = word_tokenize(out)
    out_words = remove_stopwords(out_words)   
    out_words = lemmatization(out_words)      
    
    return out_words

def preprocess_all(text_arr: pd.Series) -> pd.Series:
    text_arr = remove_emoji(text_arr)
    text_arr = remove_special(text_arr)
    text_arr = expand_contractions(text_arr)
    text_arr = to_lowercase(text_arr)
    text_arr = remove_stopwords(text_arr)
    text_arr = lemmatize_series(text_arr)
    text_arr = stem_series(text_arr)
    return text_arr

if __name__ == "__main__":
    main()