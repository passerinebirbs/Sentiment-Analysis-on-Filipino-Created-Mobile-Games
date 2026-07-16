# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 00:23:11 2025

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score

#samplers tested will proceed with RandomOverSampler but jsut in case
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#to see if model is overfitting
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

import WE_functions as wef
import time
#timer to understand how long it would take to run the code when i make a mistake
start = time.perf_counter()

embedding_dim=100
#read glove embedding
embeddings_index = wef.load_glove_embeddings("glove.2024.wikigiga.100d.txt", expected_dim=embedding_dim)

n=1 #dataset where i want to start
while n<4:#amount of datasets +1
    
    data = pd.read_json(f'preprocessed{n}.json')
    df= pd.DataFrame(data)
    
    # Split the entire DataFrame into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    #list of sentences p1 better less noice, also might prove to be a challenge
    preprocessed_train = train_df['preprocessed']
    all_sentences_train = [" ".join([" ".join(tokens) for tokens in lists]) for lists in preprocessed_train]
    
    #WE fro train data
    train_df["combined"] = train_df["preprocessed"].apply(wef.flatten_and_join)
    train_sentences = train_df["combined"].tolist()
    
    word_indexed = wef.build_vocab(train_sentences)
    
    embedding_matrix = np.zeros((len(word_indexed), embedding_dim))
    
    for word, idx in word_indexed.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    X_train = np.array([wef.sentence_to_embedding(sent, word_indexed, embedding_matrix) for sent in train_sentences])
    
    test_df = test_df.copy() 
    #WE fro train data
    test_df["combined"] = test_df["preprocessed"].apply(wef.flatten_and_join)
    test_sentences = test_df["combined"].tolist()
    
    word_indexed = wef.build_vocab(test_sentences)
    
    embedding_matrix = np.zeros((len(word_indexed), embedding_dim))
    
    for word, idx in word_indexed.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    X_test = np.array([wef.sentence_to_embedding(sent, word_indexed, embedding_matrix) for sent in test_sentences])
    
    #labels for sentiment
    labels_train = train_df['score']
    le = LabelEncoder()
    y_train = le.fit_transform(labels_train)
    
    labels_test = test_df['score']
    le = LabelEncoder()
    y_test = le.fit_transform(labels_test)
    
    #ml model
    # model = LogisticRegression(max_iter=10000, random_state=42)
    # model = LinearSVC(C=1.0, dual = True)
    # model = RandomForestClassifier(random_state=42)
    model = XGBClassifier(random_state=42)
    
    # #resampling bcs of imbalanced data
    sampler = SMOTE(random_state=42) #change RandomUnderSampler or RandomOverSampler or SMOTE 
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    #classifer
    # model.fit(X_train, y_train)
    model.fit(X_train_resampled, y_train_resampled)
    
    #predicting test data
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, zero_division=0, target_names=le.classes_, digits=4)
    print("Classification Report:\n", report)
    print('')
    
    #predicting train data for overfit checking
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"Train and Test Accuracy Difference: {abs(train_acc - test_acc)}")
    n+=1

end = time.perf_counter()
duration = end-start
print("Code duration: ", duration)