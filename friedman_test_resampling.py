# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:12:04 2026

@author: passe
"""

import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# data = pd.DataFrame({ #LR 
#     "OG":      [0.7764,	0.634,	0.7109],
#     "ROS":     [0.6174,	0.549,	0.5933],
#     "SMOTE":   [0.6066,	0.5255,	0.5835]
# })

# data = pd.DataFrame({ #LR macro
#     "OG":      [0.278,	0.3181,	0.2196],
#     "ROS":     [0.3789,	0.3881,	0.2995],
#     "SMOTE":   [0.3807,	0.3793,	0.2972]
# })

# data = pd.DataFrame({ #LR WPrec
#     "OG":      [0.6963,	0.5566,	0.6177],
#     "ROS":     [0.7147,	0.5669,	0.6277],
#     "SMOTE":   [0.7066,	0.5551,	0.614]
# })

# data = pd.DataFrame({ #LR WF1
#     "OG":      [0.7084,	0.563,	0.6177],
#     "ROS":     [0.6546,	0.5568,	0.6085],
#     "SMOTE":   [0.6459,	0.5348,	0.5964]
# })

# data = pd.DataFrame({ #LR MPrec
#     "OG":      [0.4197,	0.407,	0.3458],
#     "ROS":     [0.3369,	0.3628,	0.2827],
#     "SMOTE":   [0.2981,	0.3395,	0.2745]
# })

# data = pd.DataFrame({ #LR MF!
#     "OG":      [0.2964,	0.3212,	0.2086],
#     "ROS":     [0.3465,	0.3723,	0.289],
#     "SMOTE":   [0.3134,	0.3483,	0.2807]
# })


# data = pd.DataFrame({ #LSVC 
#     "OG":      [0.7718,	0.617,	0.698],
#     "ROS":     [0.6254,	0.5425,	0.591],
#     "SMOTE":   [0.587,	0.5173,	0.5713]
# })

# data = pd.DataFrame({ #LSVC macro
#     "OG":      [0.3115,	0.3492,	0.264],
#     "ROS":     [0.3411,	0.3631,	0.2828],
#     "SMOTE":   [0.3604,	0.3619,	0.2714]
# })

data = pd.DataFrame({ #LSVC WPrec
    "OG":      [0.7041,	0.554,	0.6249],
    "ROS":     [0.6918,	0.5444,	0.5984],
    "SMOTE":   [0.6915,	0.5321,	0.5903]
})

# data = pd.DataFrame({ #LSVC WF1
#     "OG":      [0.7216,	0.5708,	0.6461],
#     "ROS":     [0.6527,	0.543,	0.5942],
#     "SMOTE":   [0.628,	0.521,	0.5791]
# })

# data = pd.DataFrame({ #LSVC MPrec
#     "OG":      [0.4249,	0.4061,	0.3475],
#     "ROS":     [0.3286,	0.3496,	0.2788],
#     "SMOTE":   [0.2914,	0.3285,	0.25]
# })

# data = pd.DataFrame({ #LSVC MF1
#     "OG":      [0.3329,	0.3586,	0.2765],
#     "ROS":     [0.3275,	0.3548,	0.2782],
#     "SMOTE":   [0.3043,	0.3371,	0.2567]
# })


# data = pd.DataFrame({ #RF 
#     "OG":      [0.7685,	0.6148,	0.7033],
#     "ROS":     [0.6618,	0.5878,	0.6093],
#     "SMOTE":   [0.5934,	0.5432,	0.5751]
# })

# data = pd.DataFrame({ #RF macro
#     "OG":      [0.2641,	0.3023,	0.2107],
#     "ROS":     [0.2904,	0.353,	0.2454],
#     "SMOTE":   [0.3304,	0.3607,	0.2591]
# })

# data = pd.DataFrame({ #RF WPrec
#     "OG":      [0.6689,	0.5258,	0.5561],
#     "ROS":     [0.6548,	0.5442,	0.5533],
#     "SMOTE":   [0.673,	0.5493,	0.5657]
# })

# data = pd.DataFrame({ #RF WF1
#     "OG":      [0.6907,	0.5379,	0.6022],
#     "ROS":     [0.6579,	0.5592,	0.5739],
#     "SMOTE":   [0.626,	0.5452,	0.5659]
# })

# data = pd.DataFrame({ #RF MPrec
#     "OG":      [0.3795,	0.3665,	0.247],
#     "ROS":     [0.3103,	0.371,	0.2231],
#     "SMOTE":   [0.2681,	0.3481,	0.2281]
# })

# data = pd.DataFrame({ #RF MF!
#     "OG":      [0.2734,	0.2993,	0.1924],
#     "ROS":     [0.2983,	0.3561,	0.2246],
#     "SMOTE":   [0.2805,	0.3509,	0.2335]
# })


# data = pd.DataFrame({ #XGB 
#     "OG":      [0.779,	0.6265,	0.698],
#     "ROS":     [0.5645,	0.5136,	0.5592],
#     "SMOTE":   [0.603,	0.5352,	0.6199]
# })

# data = pd.DataFrame({ #XGB macro
#     "OG":      [0.2935,	0.3202,	0.2487],
#     "ROS":     [0.3915,	0.3763,	0.3213],
#     "SMOTE":   [0.3569,	0.3606,	0.2936]
# })

# data = pd.DataFrame({ #XGB WPrec
#     "OG":      [0.7073,	0.5547,	0.6161],
#     "ROS":     [0.7152,	0.5571,	0.6186],
#     "SMOTE":   [0.6889,	0.5446,	0.6047]
# })

# data = pd.DataFrame({ #XGB WF1
#     "OG":      [0.7137,	0.5638,	0.6386],
#     "ROS":     [0.6151,	0.5279,	0.5821],
#     "SMOTE":   [0.6356,	0.5336,	0.606]
# })

# data = pd.DataFrame({ #XGB MPrec
#     "OG":      [0.4636,	0.4105,	0.3253],
#     "ROS":     [0.3077,	0.3378,	0.291],
#     "SMOTE":   [0.2961,	0.339,	0.2787]
# })

# data = pd.DataFrame({ #XGB MF!
#     "OG":      [0.3146,	0.3313,	0.2566],
#     "ROS":     [0.3119,	0.3448,	0.2936],
#     "SMOTE":   [0.2969,	0.3371,	0.2708]
# })



stat, p = friedmanchisquare(
    data["OG"],
    data["ROS"],
    data["SMOTE"]
)

print(f"Friedman statistic = {stat:.4f}")
print(f"P-value = {p:.4f}")

# Run Nemenyi test only if Friedman is significant
if p < 0.05:
    nemenyi = sp.posthoc_nemenyi_friedman(data)
    print("Nemenyi Post-hoc Test")
    print(nemenyi)