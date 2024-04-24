#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:02:28 2023

@author: yan
"""

import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

from MyPyLib import ForceInf_lib as FI
import MyPyLib.GetMatrixParameterEstimation as GPE
import MyPyLib.ScaleConverter as SC
import MyPyLib.Outlier as Outlier
import MyPyLib.HalfVonMises_pymc3 as HM


#%%形態特徴量関連の行列Lの計算
def calc_L(l,theta,A,Normalize=False):
        L = np.zeros([len(l) + len(A), 7]) 
        L[:len(l),0]  = 1
        L[:len(l),1] = np.sin(2 * theta)
        L[:len(l),2] = np.cos(2 * theta)
        L[:len(l),3] = -l
        L[:len(l),4] = -l * np.sin(2 * theta)
        L[:len(l),5] = -l * np.cos(2 * theta)
        L[len(l):,6] = -A
        return L 
    
#%%角度補正
def kakudohosei(parent,data):
    hosei = pd.read_csv(parent+"hosei.csv",engine = 'python')
    if(len(hosei[hosei.file == data]) == 0):
        print("***ERROR : NO HOSEI DATA:! ***")
        return np.nan
    else:
        rad_hosei = hosei[hosei.file == data]["hoseikakudo"]*np.pi/180
        return float(rad_hosei)

#%%FFの計算
def CalcFF(filename,AreaNormalization,ExcludeShortEdge,ExcludeOutlier,Artifitial,hosei=0):
    [x,y,edge,cell,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title]\
    = FI.loaddata( filename ) #OK
    ERR_MAX = 2.0e-12
    [MM,C_NUM,X_NUM,VEC_in,Rnd,INV_NUM] =  GPE.GetMatrix_ParameterEstimation\
                        (x,y,edge,cell,E_NUM,CELL_NUMBER,R_NUM,INV_NUM,Rnd,ERR_MAX) #OK
    [V_in,E_in,C_in] = VEC_in
    [RndJ,RndE,RndC] = Rnd
    
    
    #%% 平均細胞を用いた無次元化&角度補正
    if AreaNormalization:
        non_dim = 1/np.sqrt(np.median([cell[i].area for i in C_in]))
        Data_t = SC.scale_converter(x,y,edge,cell, sc=non_dim, rotation = hosei)
        [x,y,edge,cell,sc] = Data_t.copy()
        [MM,C_NUM,X_NUM,VEC_in,Rnd,INV_NUM] =  GPE.GetMatrix_ParameterEstimation\
                            (x,y,edge,cell,E_NUM,CELL_NUMBER,R_NUM,INV_NUM,Rnd,ERR_MAX) #OK
        
    else:
        Data_t = SC.scale_converter(x,y,edge,cell, sc=1, rotation = hosei)
        [V_in,E_in,C_in] = VEC_in
        [RndJ,RndE,RndC] = Rnd
        non_dim = 1    
    
    #%% 短いedgeの除外
    if ExcludeShortEdge:
        lmin = 0.05 if Artifitial else 3 * non_dim
        short_edges = [i for i in E_in if edge[i].dist <lmin]
        print("-Exclude too short edges-")
        print(short_edges)
        print("------------------------------")
        [MM,C_NUM,X_NUM,VEC_in,Rnd,INV_NUM] =  GPE.GetMatrix_ParameterEstimation\
                        (x,y,edge,cell,E_NUM,CELL_NUMBER,R_NUM,INV_NUM,Rnd,ERR_MAX,E_ex = short_edges) #OK
        [V_in,E_in,C_in] = VEC_in
        [RndJ,RndE,RndC] = Rnd
    
    #%% 形態特徴量の計算
    l = np.array([edge[i].dist for i in E_in])
    theta = np.array([edge[i].degree for i in E_in])          
    A = np.array([cell[i].area for i in C_in])
    
    #%% 係数行列Lと目的変数y,計画行列Xの計算
    L = calc_L(l,theta,A)
    FF = np.dot(MM,L)
    
    #%% 外れ値の除去
    if ExcludeOutlier:
        # 細胞面積の外れ値検出→目的変数の除外　（See 02DEC2020）
        OutlierC_in = Outlier.OutlierDetector2(A,2.0)
        OutlierVertices = list({ji for ci in C_in[OutlierC_in] for ji in cell[ci].junc} )
        OutlierV_in = np.where([(ji in OutlierVertices) for ji in V_in])[0]
        OutlierObjVar = np.concatenate((OutlierV_in,OutlierV_in + len(V_in)))
        FF = np.delete(FF, OutlierObjVar,axis=0)
    return pd.DataFrame(FF)

#%%事前分布とモデル宣言
def TransformBetaParameter(var_name,a,b):
    mu = pm.Deterministic(var_name+"_m",a/(a+b))
    sigma = pm.Deterministic(var_name+"_s",a * b /((a+b)*(a+b)*(a+b+1)))
    return mu,sigma

def Betaab(var_name):
    a = pm.HalfCauchy(var_name+"_a",5)
    b = pm.HalfCauchy(var_name+"_b",5)
    return a,b

def location_i(var_name,a,b):
    return pm.Beta(var_name,alpha=a,beta=b,dims="sample")

def InferenceModel(X,Y,sample_idxs,samples,model_name):
    coords = {
        "sample":samples,
        "obs_id":np.arange(len(sample_idxs))}
    with pm.Model(coords=coords) as hierarchical_model:
        sample_idx = pm.Data("sample_idx",sample_idxs,dims="obs_id") 

        k_a, k_b = Betaab("k0")
        k_m, k_s = TransformBetaParameter("k",k_a,k_b)
        k_i = location_i("k_i",k_a,k_b)
        y_est = k_i[sample_idx] * X[:,5]
        
        if model_name in ["A","B","C"]:
            b0_a, b0_b = Betaab("b0")
            b0_m, b0_s = TransformBetaParameter("b0",b0_a,b0_b)
            b0_i = location_i("b0_i",b0_a,b0_b)
            
            y_est += b0_i[sample_idx] * X[:,2]
            
        if model_name in ["A","B","D"]:
            ar_a, ar_b = Betaab("ar0")
            ar_m, ar_s = TransformBetaParameter("ar0",ar_a,ar_b)
            ar_i = location_i("ar_i",ar_a,ar_b)
            phiA_0 = HM.PolarUniform("phiA_0")
            phiA_sigma = pm.HalfCauchy("phiA_sigma",5)
            phiA_kappa = pm.Deterministic("phiA_kappa",1/phiA_sigma)
            phiA_i = HM.HalfVonMises("phiA_i", mu=phiA_0, kappa=1/phiA_sigma,dims="sample")
            
            a1_i = ar_i * pm.math.sin(2*phiA_i)
            a2_i = ar_i * pm.math.cos(2*phiA_i)
            
            y_est += a1_i[sample_idx] * X[:,0] + a2_i[sample_idx] * X[:,1]
            
        if model_name == "A":
            br_a, br_b = Betaab("br0")
            br_m, br_s = TransformBetaParameter("br0",br_a,br_b)
            br_i = location_i("br_i",br_a,br_b)
            phiB_0 = HM.PolarUniform("phiB_0")
            phiB_sigma = pm.HalfCauchy("phiB_sigma",5)
            phiB_kappa = pm.Deterministic("phiB_kappa",1/phiB_sigma)
            phiB_i = HM.HalfVonMises("phiB_i",mu=phiB_0,kappa=1/phiB_sigma,dims="sample")
            
            b1_i = b0_i * br_i * pm.math.sin(2*phiB_i)
            b2_i = b0_i * br_i * pm.math.cos(2*phiB_i)
            
            y_est += b1_i[sample_idx] * X[:,3] + b2_i[sample_idx] * X[:,4]
        

        eps = pm.HalfCauchy("eps", 5.0)
        y_like = pm.Normal("y_like", mu=y_est, sigma=eps, observed=Y,dims="obs_id")


    return hierarchical_model
#%% sampling結果の保存

def samplesave(trace,fileout,stage,model_name,condition):
    trace.to_netcdf("{}/{}_{}_{}_trace.nc".format(fileout,condition,stage,model_name))
    # compare_dict[model_name] = trace
    summary = az.summary(trace)
    summary.to_csv("{}/{}_{}_{}_summary.csv".format(fileout,condition,stage,model_name))
    loo = az.loo(trace)
    loo.to_csv("{}/{}_{}_{}_loo.csv".format(fileout,condition,stage,model_name))
    waic = az.waic(trace)
    waic.to_csv("{}/{}_{}_{}_waic.csv".format(fileout,condition,stage,model_name))
    az.plot_trace(trace)
    plt.savefig("{}/{}_{}_{}_trace.png".format(fileout,condition,stage,model_name))
    plt.close()
    return summary,loo,waic
    
    
def get_divergences(t: az.InferenceData) -> np.ndarray:
    return t.sample_stats.diverging.values


def statsave(trace,out,stage,model_name,summary,loo,waic,condition):
    rhat = pm.summary(trace).r_hat
    maxrhat = rhat.max()
    maxrhat_index = rhat.idxmax()
    divergences = get_divergences(trace).flatten()
    divergentnumber=divergences.nonzero()[0].size
    # esspercent = az.ess(trace,method="mean",relative=True)
    loo_value = loo.loc["loo"]
    waic_value = waic.loc["waic"]
    stat = pd.DataFrame([[condition,stage,model_name,waic_value,loo_value,divergentnumber,maxrhat,maxrhat_index]],\
                        index=[model_name],columns=['condition','stage','model','waic','loo','divergences','r_hat','r_hat_para'])
    stat.to_csv("{}/{}_{}_statistics.csv".format(out,condition,stage),header=False, index = True,mode="a")
    