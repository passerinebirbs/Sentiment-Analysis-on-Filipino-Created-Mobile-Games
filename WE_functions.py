# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 03:03:47 2025

@author: passe
"""

import numpy as np
from collections import Counter

def load_glove_embeddings(glove_file_path, expected_dim):
    embeddings_index = {}

    with open(glove_file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            # Skip empty or malformed lines
            if len(values) != expected_dim + 1:  
                # word + expected_dim numbers
                continue

            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")

            embeddings_index[word] = vector

    print(f"Loaded {len(embeddings_index)} embeddings.")
    return embeddings_index

def build_vocab(tokenized_sentences, min_freq=1):
    counter = Counter()
    for s in tokenized_sentences:
        counter.update(s)

    word_to_index = {"<PAD>": 0, "<UNK>": 1}

    for word, freq in counter.items():
        if freq >= min_freq:
            word_to_index[word] = len(word_to_index)

    return word_to_index

def encode(sentence, word_indexed):
    return [word_indexed.get(w, 1) for w in sentence] 

def sentence_to_embedding(tokens, word_to_index, embedding_matrix): #words, built vocab, embedding matr
    vectors = []
    for word in tokens:
        idx = word_to_index.get(word, word_to_index["<UNK>"])
        vectors.append(embedding_matrix[idx])
    
    if len(vectors) == 0:
        # If empty sentence → return zeros
        return np.zeros(embedding_matrix.shape[1])
    
    # Average over all word embeddings
    return np.mean(vectors, axis=0)

def tfidf_glove_embedding(doc, feature_names, tfidf_vectorizer, glove_dict, embedding_dim):
    words = doc.split()
    tfidf_scores = tfidf_vectorizer.transform([doc])

    # Build dict: word → tfidf value
    scores = {feature_names[i]: tfidf_scores[0, i]
              for i in tfidf_scores.nonzero()[1]}

    weighted_vectors = []
    weights = []

    for w in words:
        if w in glove_dict and w in scores:
            weight = scores[w]
            weighted_vectors.append(glove_dict[w] * weight)
            weights.append(weight)

    if not weights:
        return np.zeros(embedding_dim)

    return np.sum(weighted_vectors, axis=0) / np.sum(weights)

def flatten_and_join(x):
    if isinstance(x, str):
        return x
    
    # If it's a list (maybe nested)
    if isinstance(x, list):
        flat = []
        for item in x:
            if isinstance(item, list):
                flat.extend(item)     # add inner list items
            elif isinstance(item, str):
                flat.append(item)
            else:
                continue
        return " ".join(flat)
    
    return ""   # anything else -> empty string
