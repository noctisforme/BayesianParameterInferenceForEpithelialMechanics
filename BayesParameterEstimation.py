#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:44:30 2024

@author: yan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import Series, DataFrame
import sys
import os
import copy
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import pymc3 as pm
import arviz as az
import math
import glob
import random
import scipy.stats
from scipy.stats import gaussian_kde
from scipy import optimize

import MyPyLib.ForceInf_lib
import MyPyLib.OgitaInf_NL as NOgi
import MyPyLib.GetMatrixParameterEstimation as GPE
import MyPyLib.ScaleConverter as SC
import MyPyLib.Outlier as Outlier
import MyPyLib.Multico as Multico
import MyPyLib.EB as EB
import MyPyLib.HalfVonMises_pymc3 as HVM
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

#%%　データ入力に関する情報の取得

input_args = sys.argv
if len(input_args) != 1:
    tissue = input_args[1]
    stage = input_args[2]
    filename = input_args[3]
    data = filename.split("/")[-3]
    fileout = "./output/{}/{}/".format(tissue,stage)
    fileout_sample = "{}/{}/".format(fileout,data)
    
else:
    # %%　ファイルを選ぶ
    filename = './Sample/sample/vertex/VDat_sample.dat'
    # %% データ入力に関する情報の取得（開発用）
    tissue = "Sample"
    stage = "sample"
    data = filename.split("/")[-3]
    parent = "./"+tissue
    fileout = "./output/{}/{}/".format(tissue, stage)
    fileout_sample = fileout+"{}/".format(data)
    
if not os.path.exists(fileout_sample):
    os.makedirs(fileout_sample)
    


    

        
# %% 推定に関する設定
ExcludeOutlier = True
ExcludeShortEdge = True

AreaNormalization = True
Artifitial = False


#%% ステージのフォルダ
sample_directory = "./"
tissue_directory = "{}{}/".format(sample_directory,tissue)
stage_directory = "{}{}/".format(tissue_directory,stage)
sample_list = glob.glob(stage_directory+"*/*/VDat*.dat")
sample_list = [i.replace("\\","/") for i in sample_list]

hosei = NOgi.kakudohosei(tissue_directory,data)
if hosei!=np.nan:
    hosei = -hosei #補正の方向は時計回転
else:
    pass
# %% datファイルの読み込み
[x, y, edge, cell, Rnd, CELL_NUMBER, E_NUM, V_NUM, INV_NUM, R_NUM, stl, title]\
    = MyPyLib.ForceInf_lib.loaddata(filename)  # OK
ERR_MAX = 2.0e-12
[MM, C_NUM, X_NUM, VEC_in, Rnd, INV_NUM] = GPE.GetMatrix_ParameterEstimation(
    x, y, edge, cell, E_NUM, CELL_NUMBER, R_NUM, INV_NUM, Rnd, ERR_MAX)  # OK
[V_in, E_in, C_in] = VEC_in
[RndJ, RndE, RndC] = Rnd


# %% 平均細胞を用いた無次元化

if AreaNormalization:
    non_dim = 1/np.sqrt(np.median([cell[i].area for i in C_in]))
    Data_t = SC.scale_converter(x, y, edge, cell, sc=non_dim, rotation=hosei)
    [x, y, edge, cell, sc] = Data_t.copy()
    [MM, C_NUM, X_NUM, VEC_in, Rnd, INV_NUM] = GPE.GetMatrix_ParameterEstimation(
        x, y, edge, cell, E_NUM, CELL_NUMBER, R_NUM, INV_NUM, Rnd, ERR_MAX)  # OK
else:
    Data_t = [x, y, edge, cell, 1]
    [V_in, E_in, C_in] = VEC_in
    [RndJ, RndE, RndC] = Rnd
    non_dim = 1

lmed = np.median([edge[i].dist for i in E_in])

# %% 短いedgeの除外
if ExcludeShortEdge:
    lmin = 0.05 if Artifitial else 3 * non_dim
    short_edges = [i for i in E_in if edge[i].dist < lmin]
    print("-Exclude too short edges-")
    print(short_edges)
    print("------------------------------")
    E_ex = short_edges
    [MM, C_NUM, X_NUM, VEC_in, Rnd, INV_NUM] = GPE.GetMatrix_ParameterEstimation(
        x, y, edge, cell, E_NUM, CELL_NUMBER, R_NUM, INV_NUM, Rnd, ERR_MAX, E_ex=short_edges)  # OK
    [V_in, E_in, C_in] = VEC_in
    [RndJ, RndE, RndC] = Rnd

# %% 形態特徴量の計算
l = np.array([edge[i].dist for i in E_in])
theta = np.array([edge[i].degree for i in E_in])
A = np.array([cell[i].area for i in C_in])

C_peri = np.zeros(CELL_NUMBER)
E_peri = np.zeros(E_NUM)
# obtain cell.edge & edge.ncell
for i in range(E_NUM):
    edge[i].ncell = []

for j in range(CELL_NUMBER):
    for i in range(E_NUM):
        if (edge[i].junc1 in cell[j].junc)*(edge[i].junc2 in cell[j].junc):
            cell[j].edge.append(i)  # the i-th edge is in the j-th cell
            # the j-th cell is in the cell set, one of whose edge is the i-th edge
            edge[i].ncell.append(j)
            C_peri[j] += edge[i].dist

for i in range(E_NUM):
    for j in edge[i].ncell:
        E_peri[i] += C_peri[j]
        edge[i].E_peri = E_peri[i]
E_peri = np.array([edge[i].E_peri for i in E_in])  # i番目の辺を含む細胞（二つ）の周長和
# %% 外れ値の除去
if ExcludeOutlier:
   
    OutlierC_in = Outlier.OutlierDetector2(A, 2.0)
    OutlierVertices = list(
        {ji for ci in C_in[OutlierC_in] for ji in cell[ci].junc})
    OutlierV_in = np.where([(ji in OutlierVertices) for ji in V_in])[0]
    OutlierObjVar = np.concatenate((OutlierV_in, OutlierV_in + len(V_in)))
    MM = np.delete(MM, OutlierObjVar, axis=0)



# %%目的変数Fの計算
y1_ = np.ones(theta.shape)
y2_ = np.zeros(A.shape)
y1 = y1_.reshape(-1, 1)
y2 = y2_.reshape(-1, 1)
y = np.vstack((y1, y2))
A = A.reshape(-1, 1)
theta = theta.reshape(-1, 1)
l = l.reshape(-1, 1)
E_peri = E_peri.reshape(-1, 1)
F = -np.dot(MM, y)

# %% Modeling

map_estimate_dict = {}
all_keys = ['ar', 'phi_a', 'b0', 'br', 'phi_b', 'gamma', 'k']
method = 'CG'
ans = {}
compare_dict = {}

def ModelA():
    exp = ['gamma']
    with pm.Model() as modelA:
        ar = pm.Uniform("ar", 0, 1)
        b0 = pm.Uniform("b0", 0, 1)
        br = pm.Uniform("br", 0, 1)
        k = pm.Uniform("k", 0, 1)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = HVM.PolarUniform("phi_b")
        gamma = 0
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = ar * pm.math.cos(2 * theta - 2*phi_a) - \
            b0 * (1 + br * pm.math.cos(2 * theta - 2 * phi_b)) * l
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)

    map_estimate = pm.find_MAP(model=modelA, method=method)

    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimate[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "A"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict["A"] = params_pd
    sampleandsave(modelA, "A")


def ModelA2():
    exp = []
    with pm.Model() as modelA2:
        ar = pm.Uniform("ar", 0, 1)
        b0 = pm.Uniform("b0", 0, 1)
        br = pm.Uniform("br", 0, 1)
        k = pm.Uniform("k", 0, 1)
        gamma = pm.HalfCauchy("gamma", 5)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = HVM.PolarUniform("phi_b")
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = ar * pm.math.cos(2 * theta - 2*phi_a) - \
            b0 * (1 + br * pm.math.cos(2 * theta - 2 * phi_b)) * \
            l + gamma * E_peri
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimate = pm.find_MAP(model=modelA2, method=method)

    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimate[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = 'A2'
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['A2'] = params_pd
    sampleandsave(modelA2, 'A2')

def ModelB():
    exp = ['br', 'phi_b', 'gamma']
    with pm.Model() as modelB:
        ar = pm.Uniform("ar", 0, 1)
        b0 = pm.Uniform("b0", 0, 1)
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = 0
        sigma = pm.HalfCauchy("sigma", 5)
        gamma = 0
        T_ = ar * pm.math.cos(2 * (theta - phi_a)) - b0*l
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateB = pm.find_MAP(model=modelB, method = method) 
    
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateB[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = 'B'
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['B'] = params_pd
    sampleandsave(modelB, 'B')
    
def ModelB2():
    exp = ['br', 'phi_b']
    with pm.Model() as modelB2:
        ar = pm.Uniform("ar", 0, 1)
        b0 = pm.Uniform("b0", 0, 1)
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = 0
        gamma = pm.HalfCauchy("gamma", 5)
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = ar * pm.math.cos(2 * (theta - phi_a)) - b0*l + gamma * E_peri
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateB2 = pm.find_MAP(model=modelB2, method=method)
    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateB2[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = 'B2'
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['B2'] = params_pd
    sampleandsave(modelB2, 'B2')

def ModelC():
    exp = ['ar', 'phi_a','br', 'phi_b' , 'gamma']
    with pm.Model() as modelC:
        ar = 0
        b0 = pm.Uniform("b0", 0, 1)
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = 0
        phi_b = 0
        gamma = 0
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = -b0*l
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)   
    map_estimateC = pm.find_MAP(model=modelC, method = method)
       
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateC[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "C"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['C'] = params_pd
    sampleandsave(modelC, 'C')
    
def ModelC2():
    exp = ['ar', 'phi_a', 'br', 'phi_b']
    with pm.Model() as modelC2:
        ar = 0
        b0 = pm.Uniform("b0", 0, 1)
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = 0
        phi_b = 0
        gamma = pm.HalfCauchy("gamma", 5)
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = -b0*l + gamma * E_peri
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateC2 = pm.find_MAP(model=modelC2, method=method)
    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateC2[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "C2"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['C2'] = params_pd
    sampleandsave(modelC2, 'C2')

def ModelD():
    exp = ['b0','br','phi_b', 'gamma']
    with pm.Model() as modelD:
        ar = pm.Uniform("ar", 0, 1)
        b0 = 0
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = 0
        gamma = 0
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = ar * pm.math.cos(2 * (theta - phi_a))
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateD = pm.find_MAP(model=modelD, method = method)
    print(map_estimateD)
       
    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateD[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "D"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['D'] = params_pd
    sampleandsave(modelD, 'D')
    
def ModelD2():
    exp = ['b0', 'br', 'phi_b']
    with pm.Model() as modelD2:
        ar = pm.Uniform("ar", 0, 1)
        b0 = 0
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = HVM.PolarUniform("phi_a")
        phi_b = 0
        gamma = pm.HalfCauchy("gamma", 5)
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = ar * pm.math.cos(2 * (theta - phi_a)) + gamma * E_peri
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateD2 = pm.find_MAP(model=modelD2, method=method)
    print(map_estimateD2)

    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateD2[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "D2"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['D2'] = params_pd
    sampleandsave(modelD2, 'D2')
    
def ModelE():
    exp = ['ar', 'phi_a', 'b0', 'br', 'phi_b', 'gamma']
    with pm.Model() as modelE:
        ar = 0
        b0 = 0
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = 0
        phi_b = 0
        gamma = 0
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = np.zeros(theta.shape).reshape(-1, 1)
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateE = pm.find_MAP(model=modelE, method = method)
    print(map_estimateE)
       
    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateE[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "E"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    sampleandsave(modelE, 'E')

def ModelE2():
    exp = ['ar', 'phi_a', 'b0', 'br', 'phi_b']
    with pm.Model() as modelE2:
        ar = 0
        b0 = 0
        br = 0
        k = pm.Uniform("k", 0, 1)
        phi_a = 0
        phi_b = 0
        gamma = pm.HalfCauchy("gamma", 5)
        sigma = pm.HalfCauchy("sigma", 5)
        T_ = gamma * E_peri
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)
    map_estimateE2 = pm.find_MAP(model=modelE2, method=method)
    print(map_estimateE2)

    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimateE2[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = "E2"
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict['E2'] = params_pd
    sampleandsave(modelE2, 'E2')

    
def get_divergences(t: az.InferenceData) -> np.ndarray:
    return t.sample_stats.diverging.values


def sampleandsave(model, model_name):
    count = 0
    with model:
        trace = pm.sample(10000, tune=5000, target_accept=0.99,
                          chains=2,cores=1, progressbar=True, return_inferencedata=True)
    rhat = pm.summary(trace).r_hat
    rhat_forcheck= max(rhat)
    divergences = get_divergences(trace).flatten()
    divergentnumber=divergences.nonzero()[0].size
    esspercent = az.ess(trace,method="mean",relative=True)
    count += 1
    
    while count<3:
        if  divergentnumber == 0 and rhat_forcheck<1.05 :
            break
        else:
            with model:
                trace = pm.sample(10000,tune=5000,target_accept=0.99,chains=2,cores=1,progressbar=True,return_inferencedata=True)
            rhat = pm.summary(trace).r_hat
            rhat_forcheck= max(rhat)
            divergences = get_divergences(trace).flatten()
            divergentnumber=divergences.nonzero()[0].size
            print('divergences =',divergentnumber)
            count +=1
    
    pm.plot_trace(trace)
    plt.tight_layout()
    plt.savefig(fileout_sample+data+"_trace"+model_name+".pdf",format="pdf")
    az.plot_posterior(trace, point_estimate='mode',hdi_prob=0.95)
    plt.savefig(fileout_sample+data+"_"+model_name+"posterior.pdf",format="pdf")
 
    trace.to_netcdf(fileout_sample+data+model_name+".nc")
    summary = az.summary(trace)
    summary.to_csv("{}/{}_summary.csv".format(fileout_sample,data),index=[model_name],mode='a')
    loo = az.loo(trace)
    # loo.to_csv("{}/{}_{}_loo.csv".format(fileout_sample,data,model_name))
    waic = az.waic(trace)
    # waic.to_csv("{}/{}_{}_waic.csv".format(fileout_sample,data,model_name))
    az.plot_trace(trace)
    rhat = pm.summary(trace).r_hat
    maxrhat = rhat.max()
    maxrhat_index = rhat.idxmax()
    divergences = get_divergences(trace).flatten()
    divergentnumber=divergences.nonzero()[0].size
    loo_value = loo.loc["loo"]
    waic_value = waic.loc["waic"]
    
    if not os.path.exists("{}/{}_statistics.csv".format(fileout_sample,data)):
        stats_names = ['stage','model','waic','loo','divergences','r_hat','r_hat_para']
        empty_stats = pd.DataFrame(columns=stats_names)
        empty_stats.to_csv("{}/{}_statistics.csv".format(fileout_sample,data))
        
    stat = pd.DataFrame([[stage,model_name,waic_value,loo_value,divergentnumber,maxrhat,maxrhat_index]],\
                        index=[model_name],columns=['stage','model','waic','loo','divergences','r_hat','r_hat_para'])
    stat.to_csv("{}/{}_statistics.csv".format(fileout_sample,data),header=False, index = True,mode="a")
# %% sampling
ModelA()
ModelA2()
ModelB()
ModelB2()
ModelC()
ModelC2()
ModelD()
ModelD2()
ModelE()
ModelE2()



# %%モデル選択と出力
compare_loo = az.compare(compare_dict, ic="loo")
compare_waic = az.compare(compare_dict, ic="waic")
compare_loo['tissue'] = tissue
compare_loo['stage'] = stage
compare_loo['data'] = data
compare_loo.to_csv(stage+'loo.csv', header=True, mode='a')
compare_waic['tissue'] = tissue
compare_waic['stage'] = stage
compare_waic['data'] = data
compare_waic.to_csv(stage+'waic.csv', header=True, mode='a')
best_name_loo = compare_loo.index.values.tolist()[0]
best_name_waic = compare_waic.index.values.tolist()[0]

#else
"""
statisticstable = pd.read_csv("{}/{}_statistics.csv".format(fileout_sample,data),index_col=0)
best_name_loo = statisticstable.loo.idxmax()
best_name_waic = statisticstable.waic.idxmax()
"""


MSmap_loo = map_estimate_dict[best_name_loo]
MSmap_waic = map_estimate_dict[best_name_waic]
MSmap_loo.to_csv('mapoutput_loo.csv', header=False, mode="a")
MSmap_waic.to_csv('mapoutput_waic.csv', header=False, mode="a")


