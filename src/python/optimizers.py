import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import spacy 
import umap
import xgboost as xgb
import re
import wandb

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from tqdm.notebook import trange, tqdm
import emoji
import preprocessor as pre

from sklearnex import patch_sklearn
patch_sklearn()

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

class svm_optimizer():

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, study_name): 
        """
        Initialize Objectives with data. Allows for code reuse. 
        """
        self.models = [] #save all models tested
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.study = optuna.create_study(study_name = study_name, direction = "maximize")
    
    def objective(self, trial): 
        kernels = ["linear", "rbf", "sigmoid"]
        kernel = trial.suggest_categorical("kernel", kernels)
        regularization = trial.suggest_float("regularization", 0, 1)
        gamma = trial.suggest_loguniform("gamma", 10e-3, 10e3)

        model = SVC(max_iter = 10000, C = regularization, gamma = gamma,
                    kernel = kernel).fit(self.X_train, self.y_train)
        self.models.append(model) 

        y_pred = model.predict(self.X_val)
        return accuracy(self.y_val, y_pred)
    
    def optimize(self, n_trials = 20):
        """
        Optimizes the objective with the given study. Can be called multiple times 
        and will save the state. 
        """
        wandb_kwargs = {"project": "capstone"}
        wandbc = WeightsAndBiasesCallback(metric_name = "Validation Accuracy", wandb_kwargs=wandb_kwargs)
        self.study.optimize(self.objective, n_trials = n_trials, callbacks = [wandbc])
        
    def get_best_model(self):
        """
        Returns None if no optimization has occured. Else returns the best model 
        so far according to the study. 
        """
        if self.models is None: 
            return None
        
        return self.models[self.study.best_trial.number]

class xgb_optimizer(svm_optimizer): 
    
    def __init__(self, train, val, test, study_name): 
        self.models = []
        self.train = train
        self.val = val 
        self.test = test
        self.study = optuna.create_study(study_name = study_name, direction = "maximize")

    
    def objective(self, trial): 
        param = {
            'eta': trial.suggest_loguniform("eta", 10e-4, 1), 
            'max_depth': trial.suggest_int("max_depth", 2, 10),  
            'booster': trial.suggest_categorical('booster', ["gbtree"]),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'objective': 'multi:softprob',  
            'eval_metric': 'mlogloss',
            'num_class': 6} 

        steps = 150

        model = xgb.train(param, self.train, steps)
        self.models.append(model)
        
        y_pred_prob = model.predict(self.val)
        y_pred = np.argmax(y_pred_prob, axis = 1)

        return accuracy(y_val, y_pred)
    
    def optimize(self, n_trials = 50):
        """
        Optimizes the objective with the given study. Can be called multiple times 
        and will save the state. 
        """
        wandb_kwargs = {"project": "capstone"}
        wandbc = WeightsAndBiasesCallback(metric_name = "Validation Accuracy", wandb_kwargs=wandb_kwargs)
        self.study.optimize(self.objective, n_trials = n_trials, callbacks = [wandbc])