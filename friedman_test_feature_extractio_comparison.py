# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:41:53 2026

@author: passe
"""

import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


# data = pd.DataFrame({ #LR
#     "TFIDF":      [0.7764,	0.634,	0.7109],
#     "WE":         [0.7671,	0.5892,	0.7117],
#     "TFIDF-WE":   [0.7698,	0.6088,	0.7033]
# })

# data = pd.DataFrame({ #LR Macro
#     "TFIDF":      [0.278,	0.3181,	0.2196],
#     "WE":         [0.2071,	0.2051,	0.2014],
#     "TFIDF-WE":   [0.2528,	0.2872,	0.2291]
# })

# data = pd.DataFrame({ #LR WPrec 
#     "TFIDF":      [0.6963,	0.5566,	0.6177],
#     "WE":         [0.6048,	0.4413,	0.5782],
#     "TFIDF-WE":   [0.6662,	0.5129,	0.5729]
# })

# data = pd.DataFrame({ #LR WF1 
#     "TFIDF":      [0.7084,	0.563,	0.6177],
#     "WE":         [0.6696,	0.4434,	0.5954],
#     "TFIDF-WE":   [0.6836,	0.5191,	0.6154]
# })

# data = pd.DataFrame({ #LR MPrec 
#     "TFIDF":      [0.4197,	0.407,	0.3458],
#     "WE":         [0.2222,	0.2422,	0.3428],
#     "TFIDF-WE":   [0.376,	0.3601,	0.2611]
# })

# data = pd.DataFrame({ #LR MF1 
#     "TFIDF":      [0.2964,	0.3212,	0.2086],
#     "WE":         [0.1874,	0.1602,	0.1708],
#     "TFIDF-WE":   [0.2543,	0.2759,	0.2204]
# })



# data = pd.DataFrame({ #LSVC
#     "TFIDF":      [0.7718,	0.617,	0.698],
#     "WE":         [0.7678,	0.5896,	0.7147],
#     "TFIDF-WE":   [0.7683,	0.6104,	0.7056]
# })

# data = pd.DataFrame({ #LSVC Macro
#     "TFIDF":      [0.3115,	0.3492,	0.264],
#     "WE":         [0.2073,	0.2062,	0.2022],
#     "TFIDF-WE":   [0.2286,	0.2781,	0.2218]
# })

# data = pd.DataFrame({ #LSVC WPrec 
#     "TFIDF":      [0.7041,	0.554,	0.6249],
#     "WE":         [0.6058,	0.4402,	0.5784],
#     "TFIDF-WE":   [0.634,	0.4978,	0.5439]
# })

# data = pd.DataFrame({ #LSVC WF1 
#     "TFIDF":      [0.7216,	0.5708,	0.6461],
#     "WE":         [0.6698,	0.4469,	0.5966],
#     "TFIDF-WE":   [0.6768,	0.5082,	0.6064]
# })

# data = pd.DataFrame({ #LSVC MPrec 
#     "TFIDF":      [0.4249,	0.4061,	0.3475],
#     "WE":         [0.2276,	0.2413,	0.3429],
#     "TFIDF-WE":   [0.339,	0.3227,	0.2272]
# })

# data = pd.DataFrame({ #LSVC MF1 
#     "TFIDF":      [0.3329,	0.3586,	0.2765],
#     "WE":         [0.1876,	0.1634,	0.1711],
#     "TFIDF-WE":   [0.2241,	0.2587,	0.2063]
# })


# data = pd.DataFrame({ #RF
#     "TFIDF":      [0.7685,	0.6148,	0.7033],
#     "WE":         [0.7686,	0.5918,	0.7086],
#     "TFIDF-WE":   [0.7659,	0.6095,	0.7071]
# })

# data = pd.DataFrame({ #RF Macro
#     "TFIDF":      [0.2641,	0.3023,	0.2107],
#     "WE":         [0.2267,	0.2093,	0.1985],
#     "TFIDF-WE":   [0.2432,	0.28,	0.2101]
# })

# data = pd.DataFrame({ #RF WPrec 
#     "TFIDF":      [0.6689,	0.5258,	0.5561],
#     "WE":         [0.6592,	0.4379,	0.509],
#     "TFIDF-WE":   [0.663,	0.5161,	0.5564]
# })

# data = pd.DataFrame({ #RF WF1 
#     "TFIDF":      [0.6907,	0.5379,	0.6022],
#     "WE":         [0.6762,	0.4519,	0.5925],
#     "TFIDF-WE":   [0.6838,	0.52,	0.5999]
# })

# data = pd.DataFrame({ #RF MPrec 
#     "TFIDF":      [0.3795,	0.3665,	0.247],
#     "WE":         [0.3741,	0.2527,	0.1426],
#     "TFIDF-WE":   [0.3743,	0.3913,	0.2765]
# })

# data = pd.DataFrame({ #RF MF1 
#     "TFIDF":      [0.2734,	0.2993,	0.1924],
#     "WE":         [0.2215,	0.1694,	0.166],
#     "TFIDF-WE":   [0.2481,	0.2772,	0.19]
# })



# data = pd.DataFrame({ #XGB
#     "TFIDF":      [0.779,	0.6265,	0.698],
#     "WE":         [0.7676,	0.5912,	0.7064],
#     "TFIDF-WE":   [0.7677,	0.608,	0.7094]
# })

# data = pd.DataFrame({ #XGB Macro
#     "TFIDF":      [0.2935,	0.3202,	0.2487],
#     "WE":         [0.2269,	0.2134,	0.1979],
#     "TFIDF-WE":   [0.2714,	0.3067,	0.2374]
# })

# data = pd.DataFrame({ #XGB WPrec 
#     "TFIDF":      [0.7073,	0.5547,	0.6161],
#     "WE":         [0.6208,	0.4511,	0.5086],
#     "TFIDF-WE":   [0.6769,	0.5265,	0.6218]
# })

# data = pd.DataFrame({ #XGB WF1 
#     "TFIDF":      [0.7137,	0.5638,	0.6386],
#     "WE":         [0.6757,	0.4577,	0.5914],
#     "TFIDF-WE":   [0.6988,	0.5434,	0.6276]
# })

# data = pd.DataFrame({ #XGB MPrec 
#     "TFIDF":      [0.4636,	0.4105,	0.3253],
#     "WE":         [0.3099,	0.2588,	0.1425],
#     "TFIDF-WE":   [0.4105,	0.3773,	0.3788]
# })

data = pd.DataFrame({ #XGB MF1 
    "TFIDF":      [0.3146,	0.3313,	0.2566],
    "WE":         [0.2216,	0.1783,	0.1657],
    "TFIDF-WE":   [0.2876,	0.312,	0.2385]
})



stat, p = friedmanchisquare(
    data["TFIDF"],
    data["WE"],
    data["TFIDF-WE"]
)

print(f"Friedman statistic = {stat:.4f}")
print(f"P-value = {p:.4f}")

# Run Nemenyi test only if Friedman is significant
if p < 0.05:
    nemenyi = sp.posthoc_nemenyi_friedman(data)
    print("Nemenyi Post-hoc Test")
    print(nemenyi)