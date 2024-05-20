#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#%%import modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame
import sys
import os
import pymc3 as pm
import arviz as az
import glob

import MyPyLib.ForceInf_lib
import MyPyLib.OgitaInf_NL as NOgi
import MyPyLib.GetMatrixParameterEstimation as GPE
import MyPyLib.ScaleConverter as SC
import MyPyLib.Outlier as Outlier
import MyPyLib.HalfVonMises_pymc3 as HVM
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

#%%　select input

input_args = sys.argv
if len(input_args) != 1:
    tissue = input_args[1]
    stage = input_args[2]
    filename = input_args[3]
    data = filename.split("/")[-3]
    fileout = "./output/{}/{}/".format(tissue,stage)
    fileout_sample = "{}/{}/".format(fileout,data)
else:
    filename = './Samples/w165-/090826-t01_T01/vertex/VDat_090826-t01-T001.dat'
    tissue = "wing"
    stage = "w165-"
    data = filename.split("/")[-3]
    fileout = "./output/{}/Bayes".format(stage)
    fileout_sample = fileout+"{}/".format(data)
    
if not os.path.exists(fileout_sample):
    os.makedirs(fileout_sample)
    
sample_directory = "./Samples/"
stage_directory = "{}{}/".format(sample_directory,stage)
sample_list = glob.glob(stage_directory+"*/*/VDat*.dat")
sample_list = [i.replace("\\","/") for i in sample_list]

hosei = NOgi.kakudohosei(sample_directory,data)
if hosei!=np.nan:
    hosei = -hosei
else:
    pass

# %% Setting on estimation

ExcludeOutlier = True
ExcludeShortEdge = True
AreaNormalization = True
Artifitial = False

# %% dat file input

[x, y, edge, cell, Rnd, CELL_NUMBER, E_NUM, V_NUM, INV_NUM, R_NUM, stl, title]\
    = MyPyLib.ForceInf_lib.loaddata(filename)  # OK
ERR_MAX = 2.0e-12
[MM, C_NUM, X_NUM, VEC_in, Rnd, INV_NUM] = GPE.GetMatrix_ParameterEstimation(
    x, y, edge, cell, E_NUM, CELL_NUMBER, R_NUM, INV_NUM, Rnd, ERR_MAX)  # OK
[V_in, E_in, C_in] = VEC_in
[RndJ, RndE, RndC] = Rnd

# %% AreaNormalization

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

# %% ExcludeShortEdge

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

# %% Calculate cell shape features

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
            cell[j].edge.append(i)  
            edge[i].ncell.append(j)
            C_peri[j] += edge[i].dist

for i in range(E_NUM):
    for j in edge[i].ncell:
        E_peri[i] += C_peri[j]
        edge[i].E_peri = E_peri[i]
E_peri = np.array([edge[i].E_peri for i in E_in])

# %% ExcludeOutlier

if ExcludeOutlier:
    OutlierC_in = Outlier.OutlierDetector2(A, 2.0)
    OutlierVertices = list(
        {ji for ci in C_in[OutlierC_in] for ji in cell[ci].junc})
    OutlierV_in = np.where([(ji in OutlierVertices) for ji in V_in])[0]
    OutlierObjVar = np.concatenate((OutlierV_in, OutlierV_in + len(V_in)))
    MM = np.delete(MM, OutlierObjVar, axis=0)

# %%Calculate dependent variable F

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

# %% Modeling and sampling

map_estimate_dict = {}
all_keys = ['ar', 'phi_a', 'b0', 'br', 'phi_b', 'gamma', 'k']
method = 'CG'
ans = {}
compare_dict = {}

def build_model(model_name, exp, ar_exist, b0_exist, br_exist, gamma_exist, phi_a_exist, phi_b_exist, T, sampleandsave):
    with pm.Model() as model:
        ar = pm.Uniform("ar", 0, 1) if ar_exist==1 else 0
        b0 = pm.Uniform("b0", 0, 1) if b0_exist==1 else 0
        br = pm.Uniform("br", 0, 1) if br_exist==1 else 0
        k = pm.Uniform("k", 0, 1)
        gamma = pm.HalfCauchy("gamma", 5) if gamma_exist==1 else 0
        phi_a = HVM.PolarUniform("phi_a") if phi_a_exist==1 else 0 
        phi_b = HVM.PolarUniform("phi_b") if phi_b_exist==1 else 0
        sigma = pm.HalfCauchy("sigma", 5)

        T_ = T(ar, b0, br, phi_a, phi_b, gamma)
        P = -k * A
        X = pm.math.concatenate((T_, P))
        F_est = pm.math.dot(MM, X)
        F_like = pm.Normal("F_like", mu=F_est, sigma=sigma, observed=F)

    map_estimate = pm.find_MAP(model=model, method=method)
    ans = {}
    for key in all_keys:
        ans[key] = locals()[key] if key in exp else map_estimate[key]
    res = list(ans.values())
    ar, phi_a, b0, br, phi_b, gamma, k = res

    Parameter_names = ["ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"]
    params_pd = pd.DataFrame([res], index=[data], columns=Parameter_names)
    params_pd["tissue"] = tissue
    params_pd["stage"] = stage
    params_pd["model"] = model_name
    params_pd = params_pd.reindex(columns=["tissue", "stage", "model", "ar", "phi_a", "b0", "br", "phi_b", "gamma", "k"])
    map_estimate_dict[model_name] = params_pd
    sampleandsave(model, model_name)

# Define T expressions for different models
def T_A(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * theta - 2 * phi_a) - b0 * (1 + br * pm.math.cos(2 * theta - 2 * phi_b)) * l

def T_A2(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * theta - 2 * phi_a) - b0 * (1 + br * pm.math.cos(2 * theta - 2 * phi_b)) * l + gamma * E_peri

def T_B(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * (theta - phi_a)) - b0 * l

def T_B2(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * (theta - phi_a)) - b0 * l + gamma * E_peri

def T_C(ar, b0, br, phi_a, phi_b, gamma):
    return -b0*l

def T_C2(ar, b0, br, phi_a, phi_b, gamma):
    return -b0*l + gamma * E_peri

def T_D(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * (theta - phi_a))

def T_D2(ar, b0, br, phi_a, phi_b, gamma):
    return ar * pm.math.cos(2 * (theta - phi_a)) + gamma * E_peri

def T_E(ar, b0, br, phi_a, phi_b, gamma):
    return np.zeros(theta.shape).reshape(-1, 1)

def T_E2(ar, b0, br, phi_a, phi_b, gamma):
    return gamma * E_peri
    
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
    trace_dict = {model_name:trace}
    compare_dict.update(trace_dict)
    summary = az.summary(trace)
    summary.to_csv("{}/{}_summary.csv".format(fileout_sample,data),index=[model_name],mode='a')
    rhat = pm.summary(trace).r_hat
    maxrhat = rhat.max()
    maxrhat_index = rhat.idxmax()
    divergences = get_divergences(trace).flatten()
    divergentnumber=divergences.nonzero()[0].size
    
    loo_value = az.loo(trace).elpd_loo
    waic_value = az.waic(trace).elpd_waic
    
    if not os.path.exists("{}/{}_statistics.csv".format(fileout_sample,data)):
        stats_names = ['stage','model','waic','loo','divergences','r_hat','r_hat_para']
        empty_stats = pd.DataFrame(columns=stats_names)
        empty_stats.to_csv("{}/{}_statistics.csv".format(fileout_sample,data))
        
    stat = pd.DataFrame([[stage,model_name,waic_value,loo_value,divergentnumber,maxrhat,maxrhat_index]],\
                        index=[model_name],columns=['stage','model','waic','loo','divergences','r_hat','r_hat_para'])
    stat.to_csv("{}/{}_statistics.csv".format(fileout_sample,data),header=False, index = True,mode="a")
# %% sampling

build_model('A', ['gamma'], 1, 1, 1, 0, 1, 1, T_A, sampleandsave)
build_model('A2', [], 1, 1, 1, 1, 1, 1, T_A2, sampleandsave)
build_model('B', ['br', 'phi_b', 'gamma'], 1, 1, 0, 0, 1, 0, T_B, sampleandsave)
build_model('B2', ['br', 'phi_b'], 1, 1, 0, 1, 1, 0, T_B2, sampleandsave)
build_model('C', ['ar', 'phi_a','br', 'phi_b' , 'gamma'], 0, 1, 0, 0, 0, 0, T_C, sampleandsave)
build_model('C2', ['ar', 'phi_a','br', 'phi_b'], 0, 1, 0, 1, 0, 0, T_C2, sampleandsave)
build_model('D', ['b0','br','phi_b', 'gamma'], 1, 0, 0, 0, 1, 0, T_D, sampleandsave)
build_model('D2', ['b0','br','phi_b'], 1, 0, 0, 1, 1, 0, T_D2, sampleandsave)
build_model('E', ['ar', 'phi_a', 'b0', 'br', 'phi_b', 'gamma'], 0, 0, 0, 0, 0, 0, T_E, sampleandsave)
build_model('E2', ['ar', 'phi_a', 'b0', 'br', 'phi_b'], 0, 0, 0, 1, 0, 0, T_E2, sampleandsave)

# %% model selection and ouput
compare_loo = az.compare(compare_dict, ic="loo")
compare_waic = az.compare(compare_dict, ic="waic")
compare_loo['tissue'] = tissue
compare_loo['stage'] = stage
compare_loo['data'] = data
compare_loo.to_csv("{}/{}_loo.csv".format(fileout_sample,data), header=True, mode='a')
compare_waic['tissue'] = tissue
compare_waic['stage'] = stage
compare_waic['data'] = data
compare_waic.to_csv("{}/{}_waic.csv".format(fileout_sample,data), header=True, mode='a')
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
MSmap_loo.to_csv(fileout+'mapoutput_loo.csv',  mode="a")
MSmap_waic.to_csv(fileout+'mapoutput_waic.csv', header=True, mode="a")


