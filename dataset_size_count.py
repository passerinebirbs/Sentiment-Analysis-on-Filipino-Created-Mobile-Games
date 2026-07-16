# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:28:34 2026

@author: passe
"""

import pandas as pd
total=0

dataframes = ['alcatrazescape', 'manananggal', 'potionpunch', 'potionpunch2', 'prisonescape', 'sinag', 'sproutle', 'thrillerescape', 'vitafighters']
for names in dataframes:
    data = pd.read_json(f'{names}_review.json')
    df= pd.DataFrame(data)
    
    print (names, "Size: ", len(df))
    total +=len(df)
print(total)

steam = ['theme', 'created']

total2=0
for i in steam:
    data = pd.read_json(f'steam_reviews_for_fil_{i  }.json')
    df= pd.DataFrame(data)
    
    print (names, "Size: ", len(df))
    total2+=len(df)
print(total2)
    

#5 datasets? 
#1 for steam games 6.2k
#2 for potionpunch  24k + potionpunch2 = 29.5k
#3 for prisonescape 11.5k + alcatraz, thriller 3.4k = 14.9k
#5 for sinag, sproutle, vitafighters, manananggal = 4k