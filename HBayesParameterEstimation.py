 # -*- coding: utf-8 -*-
"""

"""

#%% import modules
import numpy as np
import pandas as pd
from pandas import DataFrame
import sys
import os
import glob

import pymc3 as pm
import MyPyLib.HBayesInf as HI

#%% select input

input_args = sys.argv
if len(input_args) != 1:
    tissue = input_args[1]
    stage = input_args[2]
    model_name = input_args[3]
    condition = input_args[4]
    out = "{}/{}_{}".format(input_args[5], condition,stage)
    fileout = "{}/{}".format(out, model_name)
    plot = False
    print(tissue)
else:
    tissue = "Samples"
    stage = "w165-"
    model_name = "A"
    condition = "20240424"
    out = "./output/{}/HBayes".format(stage)
    fileout = "./output/{}/HBayes/{}/".format(stage,model_name)
    plot = True

if not os.path.exists(fileout):
    os.makedirs(fileout)

sample_directory = "./Samples/"
stage_directory = "{}{}/".format(sample_directory,stage)
sample_list = glob.glob(stage_directory+"*/*/VDat*.dat")
sample_list = [i.replace("\\","/") for i in sample_list]

#%% Setting on estimation
ExcludeOutlier = True
ExcludeShortEdge = True
AreaNormalization = True
Artifitial = False

#%% Calculate matrix FF
FF_ALL = DataFrame()
FFlowNum = []
for sample in sample_list:
    data = sample.split("/")[-3].replace("VDat_","").replace(".dat","")    
    print(data)
    hosei = -HI.kakudohosei(sample_directory,data)
    print(hosei)
    if hosei!=np.nan:
        FF = HI.CalcFF(sample,AreaNormalization,ExcludeShortEdge,ExcludeOutlier,Artifitial,hosei)
        FFlowNum.append(FF.shape[0])
        FF["Sample"] = data
        FF_ALL = pd.concat([FF_ALL,FF])


#%% Calculate variables
y = -FF_ALL.iloc[:,0].values
X = FF_ALL.iloc[:,1:7].values
X_mean = X.mean(axis=0, keepdims=True)
X_centered = X - X_mean
X_nc = X
X = X_centered

sample_idxs, samples = pd.factorize(FF_ALL.Sample)

hierarchical_model = HI.InferenceModel(X,y,sample_idxs,samples,model_name)

#%% sampling&plotting

if __name__ == "__main__":
    with hierarchical_model:
        trace = pm.sample(10000,tune=10000,chains=6,cores=1,target_accept=0.99,return_inferencedata=True,progressbar=True)                    

summary, loo, waic = HI.samplesave(trace,fileout,stage,model_name,condition)

if not os.path.exists("{}/{}_{}_statistics.csv".format(out,condition,stage)):
    stats_names = ['condition','stage','model','waic','loo','divergences','r_hat','r_hat_para']
    empty_stats = pd.DataFrame(columns=stats_names)
    empty_stats.to_csv("{}/{}_{}_statistics.csv".format(out,condition,stage), header = True)
    
HI.statsave(trace,out,stage,model_name,summary,condition)



