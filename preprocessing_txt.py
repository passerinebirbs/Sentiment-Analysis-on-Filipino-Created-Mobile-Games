# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 03:45:05 2025

@author: passe
"""

import nltk
import re
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
# Download the 'punkt' tokenizer if you haven't already


def split_paragraph(paragraph):#splits paragraphs into sentences        
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

# def special_char_removal(text): #must be a sentence. must change this to normalize special char removal
#     remove_special_char = str.maketrans('','', string.punctuation)
#     cleaned = text.translate(remove_special_char)
#     return cleaned

# def remove_punctuation(text): #only the unnecessary ones
#     cleaned = re.sub(r"[^\w\s!?]", "", text)
#     pattern = r"([{}])\1+".format(re.escape(string.punctuation))
#     new_text = re.sub(pattern, r"\1", cleaned) #replace consecutive punctuation with a single instance of that punctuation
#     return new_text

def remove_links(text):
    url_pattern = r'https?://\S+|www\.\S+|ftp://\S+'
    return re.sub(url_pattern, '', text) # Replace found URLs with an empty string

def special_char_removal(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def tokenization(text): 
    token = nltk.word_tokenize(text) #seperates 
    return token

def lowercase(text):
    lowercased = [word.lower() for word in text] #must be list of tokens
    return lowercased

# def correct_misspellings(text): #test must be tokenized otherwise per letter mababasa
#     spell = SpellChecker()
#     corrected = []
#     for word in text:
#         if word.isalpha(): #checks alphabet letters
#             corrected.append(spell.correction(word) or word)
#         else:
#             corrected.append('')
#     return corrected

def correct_misspellings(text): #test must be tokenized otherwise per letter mababasa
    spell = SpellChecker()
    corrected = []

    for word in text:
        if word.isalpha(): #checks alphabet letters
            if word in spell: #if word is already in the spellchecker dict
                corrected.append(word)
            elif spell.candidates(word): #choose most probable correction in list of candidates
                corrected.append(spell.correction(word))
            else: #return empty string if word cant be corrected
                corrected.append('')
        else:
            corrected.append(word)
    return corrected

def stop_word_removal(tokens): #must be a list of tokens of the sentence
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def pos_tagging(word): #done before lemmatization to understand base form
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # Default to noun if not found

def lemmatization(word): #last preprocess
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos_tagging(word))
    return lemma
