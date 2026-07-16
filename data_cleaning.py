# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:33:42 2026

@author: passe
"""
import pandas as pd
ppunchg=['potionpunch_review.json', 'potionpunch2_review.json']
escapeg=['alcatrazescape_review.json', 'prisonescape_review.json', 'thrillerescape_review.json']
miscg=['manananggal_review.json', 'sinag_review.json', 'sproutle_review.json', 'vitafighters_review.json']

sets = [ppunchg, escapeg, miscg]

n=1
for datasets in sets:
    data = [pd.read_json(file) for file in datasets]
    df = pd.concat(data, ignore_index=True)
    
    drop_df= df.drop(columns=['reviewId', 'userName', 'userImage', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion'])
    final_df = drop_df.rename(columns={"content" : "review"})
    final_df['score']=final_df['score'].replace({1:"poor", 2:"fair", 3:"good", 4:"very good", 5 :"excellent"})
    print(final_df)
    
    json_str = final_df.to_json(f'dataset{n}.json', orient = 'records', indent = 3)
    n+=1
    