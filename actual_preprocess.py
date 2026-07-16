     # -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 00:12:08 2025

@author: passe
"""

import preprocessing_txt as pt
import pandas as pd
from tqdm import tqdm
n=1 #dataset where i want to start
while n<3:#amount of datasets +1
    df = pd.read_json(f'dataset{n}.json')
    df= pd.DataFrame(df)
    df['ID'] = df.index + 1
    preprocessed= {}
    with_errors = []
    for i, row in tqdm(df.iterrows(), total=len(df)):    
        try:
            review = row['review']
            rev_id = row['ID']
            split_review = pt.split_paragraph(review)
            preprocessed_text = []
            for sentence in split_review:
                remove_links = pt.remove_links(sentence)
                remove_punctuations = pt.special_char_removal(remove_links)
                tokens = pt.tokenization(remove_punctuations)
                lowercase_words = pt.lowercase(tokens)
                corrected_words = pt.correct_misspellings(lowercase_words)
                lemmad =[]
                for word in corrected_words:
                    lemma_the_words = pt.lemmatization(word)
                    lemmad.append(lemma_the_words)
                stop_wrded= pt.stop_word_removal(lemmad)
                preprocessed_text.append(stop_wrded)
            preprocessed[rev_id] = preprocessed_text
        except Exception as e:
            print("")
            print(f"Code broke for {rev_id} with error {e}")
            with_errors.append(rev_id)
    
    print("The IDs with errors are: ", with_errors)
    
    p_df = pd.DataFrame(list(preprocessed.items()), columns=["ID", "preprocessed"])
    p_df = pd.merge(df, p_df, how = 'right')
    print(p_df)
    
    json_str = p_df.to_json(f'preprocessed{n}.json', orient = 'records', indent = 3)
    n+=1

