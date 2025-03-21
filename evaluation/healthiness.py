import numpy as np 

def mean_in_mask(X, mask):
    return X.mean(where=mask.astype(bool))

def healthiness_score(X, mask, out_mask):
    return mean_in_mask(X, mask) / mean_in_mask(X, out_mask)

