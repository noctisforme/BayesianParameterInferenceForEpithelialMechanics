# -*- coding: utf-8 -*-
"""
外れ値の検出や可視化のためのモジュール
Created on Thu Dec  3 00:39:45 2020

@author: Goshi
"""
#%% Import modules
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.robust as robust
from MyPyLib.ForceInf_lib import EDGE, CELL 
#%% 外れ値の検出
def OutlierDetector(ListWOutlier):
    MAD = np.median(ListWOutlier)
    MADN = robust.mad(ListWOutlier)
    lower95, upper95 = MAD - 1.96 * MADN, MAD + 1.96 * MADN
    Lower = ListWOutlier < lower95
    Upper = upper95 < ListWOutlier
    print("Total: {}".format(len(ListWOutlier)))
    print("Lower Outleir: {}".format(sum(Lower)))
    print("Upper Outleir: {}".format(sum(Upper)))
    return  np.where(np.logical_or(Lower, Upper))[0]

def OutlierDetector2(ListWOutlier,maxmum_mag):
    MAD = np.median(ListWOutlier)
    Upper = (maxmum_mag) * MAD < ListWOutlier
    print("Total: {}".format(len(ListWOutlier)))
    print("Upper Outleir: {}".format(sum(Upper)))
    return  np.where( Upper  )

#%% 外れ値の可視化
def OutlierHistgram(ListWOutlier,data,fileout_sample=0,title="Cell Area"):
    plt.title(title+":"+data)
    MAD = np.median(ListWOutlier)
    MADN = robust.mad(ListWOutlier)
    lower_095, upper_095 = MAD - 1.96 * MADN, MAD + 1.96 * MADN
    count,_,_ = plt.hist(ListWOutlier)
    plt.vlines(lower_095,0,1.1*max(count),color="r",linestyle="--",label="95 %")
    plt.vlines(upper_095,0,1.1*max(count),color="r",linestyle="--",label="95 %")
    plt.legend()
    if fileout_sample:
        plt.savefig(fileout_sample+title.replace(" ","")+"_Outlier.png")
        


#%% 確認:どのvertexが除外されたか（面積外れ値）
def PlotOutlierVertices(Data_t,OutlierVertices,fileout_sample=0):
    [x,y,edge,cell,sc] = Data_t
    plt.figure()
    plt.axes().set_aspect('equal', 'datalim')
    for e in edge:
            tx = [x[e.junc1], x[e.junc2]]
            ty = [y[e.junc1], y[e.junc2]]
            plt.plot( tx, ty, '-k' )
    for ji in OutlierVertices:
        plt.plot(x[ji],y[ji],"xr")
    if fileout_sample:
        plt.savefig(fileout_sample+"Excluded_junc.png")
    
