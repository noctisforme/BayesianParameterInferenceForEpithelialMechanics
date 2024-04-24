# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:46:27 2018

@author: Goshi
"""
import numpy as np
from scipy.optimize import fmin
import os
import MyPyLib.ForceInf_lib
import MyPyLib.OgitaInf_NL as NOgi
import matplotlib.pyplot as plt
import re
import glob
import sys

def make_ConcatinateMatrix(MM,VV,V,G,smu,B):
    Sa = np.concatenate( [MM, VV], axis=1 )
    Sb = np.concatenate( [B, G.T], axis = 1 )
    S = np.concatenate( [Sa,smu*Sb], axis=0)
    return S



def get_ABIC( x, *args ):
    Mu = x
    MM, V, B, G, Parameter_Number = args[0],args[1],args[2],args[3],args[4]

    C_NUM = MM.shape[0] #  condition number
    X_NUM = MM.shape[1] # unknown vaiable number

    smu = np.sqrt(Mu)
    K = 1              # number of zero-eigen values of A
    UN = np.linalg.matrix_rank( np.transpose(B).dot(B) )  # rank of B'B
    M0 = X_NUM-UN;    # Number of zero-eigen values of B
    NKM = C_NUM+K-M0;

    S = make_ConcatinateMatrix(MM,V,B,G,smu,B)
    R = np.linalg.qr(S, mode='r')

    H = R[0:X_NUM,0:X_NUM]
    # h = R[0:,-1]
    F = R[X_NUM,X_NUM]
    F = F*F

    dh = np.diag(H)
    dh = np.delete( dh, -1 )
    dh = np.abs( dh )

    detlA = 2*np.sum( np.log(dh) )
    detlB = UN*np.log(Mu)

    ABIC = NKM+NKM*np.log(2.0*np.pi*F/(NKM))+detlA-detlB+2*Parameter_Number;

    print('%e %e' % (x, ABIC) )
    return ABIC



def load_true_PT( filename, E_NUM, CELL_NUMBER ):

    tep = np.zeros( E_NUM+CELL_NUMBER )
    for line in open(filename, 'r'):
        if '# T ' in line:
            itemList = line[:-1].split()
            eid = int( itemList[2] )
            tep[eid] = np.float( itemList[3] )

        if '# P ' in line:
            itemList = line[:-1].split()
            cid = int( itemList[2] )
            tep[E_NUM+cid] = np.float( itemList[3] )


    return tep


def EBayes(filename,fileout,data,MM,x,y,edge,cell,E_NUM,CELL_NUMBER,X_NUM,
           Trange=[0.4,1.6],Prange=[-0.04,0.04],display=1,omit_recal=0):
    
    filelist = glob.glob(fileout+"*")
    for i in filelist:     
        if re.match("^Eout",os.path.split(i)[1]) and omit_recal:
            [T,E] = NOgi.ForceReader(i)
            return [T,E]
    
    
    C_NUM = MM.shape[0] #  condition number
    X_NUM = MM.shape[1] # unknown vaiable number
    HParameter_Number = 2
    B = np.zeros( (X_NUM, X_NUM) )
    B[0:E_NUM,0:E_NUM] = np.identity(E_NUM)
    G = np.concatenate( ( np.ones(E_NUM), np.zeros(CELL_NUMBER)) ).reshape((1,X_NUM))
    V = np.zeros(MM.shape[0]*1).reshape(MM.shape[0],1) # 速度データは0
    
    
    
    
    
    
    print('\n#######################################')
    print('#          Minimization of ABIC        ')
    print('#######################################')
    
    ## change method for optimization ??
    mu = 1.0
    pargs = ( MM,V,B,G,HParameter_Number )
    #res = fmin( get_ABIC,mu,pargs,  xtol=1e-2, ftol=1e-2, maxiter=20, maxfun=100 )
    res = fmin( get_ABIC, mu,  args = pargs,  xtol=1e-2, ftol=1e-2, maxiter=20, maxfun=100 )
    
    
    print('\n#####################################')
    print('#           MAP estimation             ')
    print('#######################################')
    mu = res[0]
    smu = np.sqrt(mu)
    K=1              # number of zero-eigen values of A
    UN = np.linalg.matrix_rank( np.transpose(B).dot(B) )  # rank of B'B
    M0 = X_NUM-UN    # Number of zero-eigen values of B
    NKM = C_NUM+K-M0
    VV=np.zeros((C_NUM,1))
    S = make_ConcatinateMatrix(MM,VV,V,G,smu,B)
    R = np.linalg.qr(S, mode='r')
    H = R[0:X_NUM,0:X_NUM]
    h = R[0:X_NUM,-1]
    ep = np.linalg.pinv(H).dot(h)
    
    norm = np.mean( ep[0:E_NUM])
    ep = ep/norm
    
    
    
    
    
    print('\n#####################################')
    print('#           Show results               ')
    print('#######################################')
    
    T = ep[0:E_NUM]
    P = ep[E_NUM:X_NUM]
    
    #結果の出力
    f_out=open(fileout+"/Eout_"+data+".dat",'w')
    f_out.write('# %s \n' % (filename) )
    f_out.write('### type EBayes  \n')
    f_out.write('### E_NUM % d \n' % (E_NUM))
    f_out.write('### CELL_NUMBER % d \n' % (CELL_NUMBER))
                
    for (i, e) in enumerate(edge):
        f_out.write("E %5d % e\n" % (i,T[i]) )
    
    f_out.write("\n")
    
    for (i, c) in enumerate(cell):
        f_out.write("C %5d % e\n" % (i,P[i]) )
    
    f_out.write("\n")
    f_out.close()
    
    print('####    Draw tensions and pressures   ')
    MyPyLib.ForceInf_lib.Draw_Tension(x,y,T,edge,T_LINE_WIDTH = 2.0, tmin = 0.6, tmax = 1.5,savefile = fileout+data+"_ETmap.eps")
    if(display == 1.0):
        plt.pause(np.nan)    
    plt.close()
    MyPyLib.ForceInf_lib.Draw_Pressure(x,y,P,edge,cell, -0.02, 0.025, savefile = fileout+data+"_EPmap.eps")
    if(display == 1.0):
        plt.pause(np.nan)
    plt.close()

     
    
    NOgi.ForcePlot(data+"_EB",fileout,data,edge,cell,T,P,E_NUM,CELL_NUMBER,range(E_NUM),range(CELL_NUMBER),Trange,Prange,display)
    return [np.array(T),np.array(P)]