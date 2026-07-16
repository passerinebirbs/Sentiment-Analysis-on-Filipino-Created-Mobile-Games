# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 23:57:27 2025

@author: passe
"""

import pandas as pd
import numpy as np #for graphs
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

#MACHINE LEARNING LIBRARY 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score

#samplers tested will proceed with RandomOverSampler but jsut in case
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#to see if model is overfitting
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

import time
#timer to understand how long it would take to run the code when i make a mistake
start = time.perf_counter()

n=1 #dataset where i want to start
while n<5:#amount of datasets +1
    data = pd.read_json(f'preprocessed{n}.json')
    df= pd.DataFrame(data)
    
    # Split the entire DataFrame into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    #list of sentences p1 better less noice, also might prove to be a challenge
    preprocessed_train = train_df['preprocessed']
    all_sentences_train = [" ".join([" ".join(tokens) for tokens in lists]) for lists in preprocessed_train]
    
    #feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X_train = vectorizer.fit_transform(all_sentences_train)
    
    #labels for sentiment
    labels_train = train_df['score']
    le = LabelEncoder()
    y_train = le.fit_transform(labels_train)
    
    #ml model
    # model = LogisticRegression(max_iter=10000, random_state=42)
    # model = LinearSVC(C=1.0, dual = True)
    # model = RandomForestClassifier(random_state=42)
    model = XGBClassifier(random_state=42)
    
    # model.fit(X_train, y_train)
    
    # resampling 
    sampler = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    model.fit(X_train_resampled, y_train_resampled)
    
    #test data
    preprocessed_test = test_df['preprocessed']
    all_sentences_test = [" ".join([" ".join(tokens) for tokens in lists]) for lists in preprocessed_test]
        
    labels_test = test_df['score']
        
    X_test = vectorizer.transform(all_sentences_test)
    y_test = le.transform(labels_test)
        
    y_pred = model.predict(X_test)
    
        #accuracy, precision, recall, and f1 (add others?)
    print("Classification Report ")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    print("")
        
    #predicting train data for overfit checking
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"Train and Test Accuracy Difference: {abs(train_acc - test_acc)}")
    n+=1
    
end = time.perf_counter()
duration = end-start
print("Code duration: ", duration)