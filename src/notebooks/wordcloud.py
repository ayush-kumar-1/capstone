#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:52:29 2022

@author: laurenwilkes
"""


import pandas as pd
import numpy as np
import contractions
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')


def main(): 
    tweets = pd.read_csv("~/Downloads/processed_tweets.csv")


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


def remove_ats(text_arr): 
    at_regex = r"@[A-Za-z0-9]+"
    return text_arr.str.replace(at_regex, "")


#Remove punctuation, symbols, numbers
def remove_special(text_arr: pd.Series) -> pd.Series:
    # special_chars = '''`~|!|@|#|\$%^&*\(\)-+-=\{\}[]\|\\:;'<>,.?/"0123456789'''
    special_chars = r"[^a-zA-Z\s]"
    return text_arr.str.replace(special_chars, "")




def preprocess(text):
    out = to_lowercase(text)  
    out = expand_contractions(out)                                                           
    out_words = word_tokenize(out)
    out_words = remove_stopwords(out_words)   
    out_words = lemmatization(out_words)
    
 
   
    
    return out_words

def preprocess_all(tweets):
    return tweets.apply(preprocess)

tweets = df.ascii
tweets = remove_ats(tweets)
tweets = remove_emoji(tweets)
tweets = remove_special(tweets)   

df['tweet_text'] = tweets


df = pd.read_csv("~/Downloads/processed_tweets.csv")

df["processed_text"] = preprocess_all(df.ascii)



import spacy



from wordcloud import WordCloud
from textwrap import wrap

def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()

df[df["cyberbullying_type"] == "not_cyberbullying"]["processed_text"]
notCyberList = df[df["cyberbullying_type"] == "not_cyberbullying"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(notCyberList)
notwordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)

import matplotlib.pyplot as plt

plt.imshow(notwordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




genderCyberList = df[df["cyberbullying_type"] == "gender"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(genderCyberList)
genderwordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(genderwordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


religionCyberList = df[df["cyberbullying_type"] == "religion"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(religionCyberList)
religionwordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(religionwordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

otherCyberList = df[df["cyberbullying_type"] == "other_cyberbullying"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(otherCyberList)
otherwordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(otherwordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


ageCyberList = df[df["cyberbullying_type"] == "age"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(ageCyberList)
agewordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(agewordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

ethnicityCyberList = df[df["cyberbullying_type"] == "ethnicity"]["processed_text"].apply(pd.Series).stack().unique()
text = " ".join(ethnicityCyberList)
ethnicitywordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(ethnicitywordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



