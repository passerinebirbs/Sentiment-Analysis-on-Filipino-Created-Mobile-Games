# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:19:17 2026

@author: passe
"""

import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# overall accuracy scores
# data = pd.DataFrame({ #Acc
#     "LR":      [0.7764,	0.634,	0.7109],
#     "LSVC":    [0.7718,	0.617,	0.698],
#     "RF":      [0.7685,	0.6148,	0.7033],
#     "XGB":     [0.779,	0.6265,	0.698]
# })

# data = pd.DataFrame({ #Balanced
#     "LR":      [0.278,	0.3181,	0.2196],
#     "LSVC":    [0.3115,	0.3492,	0.264],
#     "RF":      [0.2641,	0.3023,	0.2107],
#     "XGB":     [0.2935,	0.3202,	0.2487]
# })

# data = pd.DataFrame({ #WPrec
#     "LR":      [0.6963,	0.5566,	0.6177],
#     "LSVC":    [0.7041,	0.554,	0.6249],
#     "RF":      [0.6689,	0.5258,	0.5561],
#     "XGB":     [0.7073,	0.5547,	0.6161]
# })


# data = pd.DataFrame({ #WF1
#     "LR":      [0.7084,	0.563,	0.6177],
#     "LSVC":    [0.7216,	0.5708,	0.6461],
#     "RF":      [0.6907,	0.5379,	0.6022],
#     "XGB":     [0.7137,	0.5638,	0.6386]
# })

# data = pd.DataFrame({ #MPrec
#     "LR":      [0.4197,	0.407,	0.3458],
#     "LSVC":    [0.4249,	0.4061,	0.3475],
#     "RF":      [0.3795,	0.3665,	0.247],
#     "XGB":     [0.4636,	0.4105,	0.3253]
# })

data = pd.DataFrame({ #MF1
    "LR":      [0.2964,	0.3212,	0.2086],
    "LSVC":    [0.3329,	0.3586,	0.2765],
    "RF":      [0.2734,	0.2993,	0.1924],
    "XGB":     [0.3146,	0.3313,	0.2566]
})



# Friedman test
stat, p = friedmanchisquare( #between models based on 
    data["LR"],
    data["LSVC"],
    data["RF"],
    data["XGB"]
)


print(f"Friedman statistic = {stat:.4f}")
print(f"P-value = {p:.4f}")

# Run Nemenyi test only if Friedman is significant
if p < 0.05:
    nemenyi = sp.posthoc_nemenyi_friedman(data)
    print("Nemenyi Post-hoc Test")
    print(nemenyi)
    
    