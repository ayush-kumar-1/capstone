import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import umap
import xgboost as xgb
import re
import wandb

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from tqdm.notebook import trange, tqdm
import emoji
import preprocessor as pre

from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece as WP
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.trainers import WordPieceTrainer

from sklearnex.svm import SVC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tfidf
from sklearn.preprocessing import MaxAbsScaler

import warnings #this my friends is overconfidence at it's finest
warnings.filterwarnings("ignore")

def train_test_val_split(X, y, test_size = 0.2, val_size = 0.2, verbose = False):
    """
    Split data into train, test, and validation sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42122, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42122, stratify=y_train)

    if verbose:
        print(f"Shape of raw data\n\nX: {X.shape}\nY: {y.shape}\n")
        print(f"Shape of split data\n\nTrain: {X_train.shape}\nValidation: {X_val.shape}\nTest: {X_test.shape}\n")


    return X_train, X_val, X_test, y_train, y_val, y_test

def model_report(model_name: str, model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)

    print(f"{model_name} Classification Report")
    print(f"Model Accuracy: {accuracy(y_test, y_pred):0.3f}")
    print(classification_report(y_test, y_pred), "\n")