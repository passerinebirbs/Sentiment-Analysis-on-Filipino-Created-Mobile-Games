# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 02:50:20 2026

@author: passe
"""

import matplotlib.pyplot as plt
import pandas as pd

n=1 #dataset where i want to start
while n<4:#amount of datasets +1
    
    data = pd.read_json(f'preprocessed{n}.json')
    df= pd.DataFrame(data)
    
    score_names = ["poor", "fair", "good", "very good", "excellent"]
    counts = df["score"].value_counts().reindex(score_names, fill_value=0)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="steelblue", edgecolor="black")
    
    plt.title(f"Review Rating Distribution of Dataset {n}")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    
    # Add count labels
    for i, value in enumerate(counts):
        plt.text(i, value + 0.05, str(value), ha="center")
    
    plt.tight_layout()
    plt.show()
    n+=1