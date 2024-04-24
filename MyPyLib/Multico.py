# -*- coding: utf-8 -*-
"""
多重線形性に関するモジュール
Created on Fri Dec  4 23:07:29 2020

@author: Goshi
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


def CalcVIF(X):
    if X.shape[1] == 1:
        return [1]
    else:
        vifs = [vif(X, i) for i in range(X.shape[1])]
    return vifs

def CheckMultico(X):
    MaxVIF = max(CalcVIF(X))
    #print("Maximum VIF: {}".format(int(MaxVIF)))
    if MaxVIF >= 10:
        return True
    else:
        return False