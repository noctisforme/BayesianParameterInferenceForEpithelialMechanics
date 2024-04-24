# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:40:35 2018

@author: Goshi
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def input_splitter(input_args):
    #Input File
    input_args[1]=input_args[1].replace("@{FullName=","")
    input_args[1]=input_args[1].replace("}","")
    filename = input_args[1]
    vertex,data=os.path.split(filename)
    non,vertex=os.path.split(vertex)
    non,data=os.path.split(non)
    parent,stage=os.path.split(non)
    
    #Output Directory
    fileout =input_args[2]+stage+"/" 
    if(os.path.exists(fileout)==False): 
        os.mkdir(fileout)
    fileout=fileout+data+"/"
    if(os.path.exists(fileout)==False):
        os.mkdir(fileout)
    stage=stage.replace("w","")
    stage=stage.replace("n","")
    stage=stage.replace("-","")
    stage=int(stage)
    return parent,filename,data,stage,fileout

def ForceReader(outitem):
    for line in open(outitem):
        if '#' in line:
            if 'type' in line:
                itemList = line[:-1].split()
            elif 'E_NUM' in line:
                itemList = line[:-1].split()
                E_NUM = int(itemList[2])
                T=list(range(E_NUM))
            elif 'CELL_NUMBER' in line:
                itemList = line[:-1].split()
                CELL_NUMBER = int(itemList[2])
                P=list(range(CELL_NUMBER))
        else:
            if 'E' in line:
                itemList = line[:-1].split()
                eid = int(itemList[1])
                T[eid]= np.float64(itemList[2])
            if 'C' in line:
                itemList = line[:-1].split()
                cid = int(itemList[1])
                P[cid]= np.float64(itemList[2])
    return [T,P]

def kakudohosei(parent,data):
    hosei = pd.read_csv(parent+"hosei.csv",engine = 'python')
    if(len(hosei[hosei.file == data]) == 0):
        print("***ERROR : NO HOSEI DATA:! ***")
        return np.nan
    else:
        rad_hosei = hosei[hosei.file == data]["hoseikakudo"]*np.pi/180
        return float(rad_hosei)


def CellEdge_inout(MM,edge,cell,E_NUM,CELL_NUMBER,V_NUM,R_NUM,INV_NUM,Rnd,E_ex = []):
    RndJ = Rnd[0]
    RndE = Rnd[1]
    RndC = Rnd[2]
    
    #C_in & C_out
    C_out = RndC
    C_in = set(range(CELL_NUMBER))-set(C_out)
    
    #V_in & V_out
    V_out = set()
    for i in C_out:
        V_out = V_out|set(cell[i].junc)
    #2019/06/05 Vall = V_out + VRnd + V_in
    V_in = set(range(V_NUM)) - V_out
    V_out = V_out - set(Rnd[0])
    if ( len(V_out) +  len(V_in) + len(Rnd[0]) != V_NUM ):
        print("***ERROR : V_out + V_in + Rnd[0] must be same as V_NUM!***" )
        print("           %d  +   %d  +  %d  = %d               %d" 
              %(len(V_out),len(V_in),len(Rnd[0]),len(V_out)+len(V_in)+len(Rnd[0]),V_NUM))
        sys.exit()
    if(E_ex != 0):        
        for i in E_ex:            
            if(edge[i].junc1 in V_in):
                V_in.remove(edge[i].junc1)
                V_out.add(edge[i].junc1)
            if(edge[i].junc2 in V_in):
                V_in.remove(edge[i].junc2)
                V_out.add(edge[i].junc2)
            
    #E_in & E_out
    E_out = [i for i in range(E_NUM) if (edge[i].junc1 in V_out|set(Rnd[0]))*(edge[i].junc2 in V_out|set(Rnd[0])) ==  1]
    E_in = set(range(E_NUM)) - set(E_out)
    #E_out = set(E_out) - set(Rnd[1])
    """#For check
    fi = plt.figure()
    for i in E_in:
        plt.plot([x[edge[i].junc1],x[edge[i].junc2]],[y[edge[i].junc1],y[edge[i].junc2]],'-g')
    for i in E_out:
        plt.plot([x[edge[i].junc1],x[edge[i].junc2]],[y[edge[i].junc1],y[edge[i].junc2]],'-b')
    plt.plot(x[list(V_out)],y[list(V_out)],'.')
    plt.plot(x[list(V_in)],y[list(V_in)],'ro')
    plt.pause(0.1)
    sys.exit()
    """
    #Exclude E_out & C_out
    ME=MM[:,:E_NUM]
    MC=MM[:,E_NUM:]
    ME_in = np.delete(ME,list(E_out),axis=1)
    MC_in = np.delete(MC,C_out,axis=1)
    MM_in = np.concatenate((ME_in,MC_in),axis=1)
    
    #temporal 
    V_out = [i - len(Rnd[0]) for i in V_out]
    
    #Exclude V_out
    MX = MM_in[:INV_NUM,:]
    MY = MM_in[INV_NUM:,:]
    MX_in = np.delete( MX, list(V_out), axis=0 )
    MY_in = np.delete( MY, list(V_out), axis=0 )
    MM_in = np.concatenate((MX_in,MY_in),0)
    
    
    return [MM_in,E_in,E_out,C_in,C_out]

def Nondimensionalize(cell,edge,CELL_NUMBER,E_NUM,C_in):
    Aave = np.mean([cell[i].area for i in C_in])
    for i in range(E_NUM):
        edge[i].dist = edge[i].dist/np.sqrt(Aave)
    for i in range(CELL_NUMBER):
        cell[i].area = cell[i].area/Aave
    return Aave

def tension():
    pass

def pressure():
    pass

def calc_tension(edge,para,edge_list,c):
    T = np.array([tension(edge[i].dist,edge[i].E_peri,edge[i].degree,para) for i in edge_list])
    return c*T

def calc_pressure(cell,para,cell_list,c):
    P = np.array([pressure(cell[i].area,para) for i in cell_list])
    P = c*P
    P = P - np.mean(P)
    return P

#calc residu 張力の平均が1縛り
def calc_residu(para,MM_in,edge,cell,edge_list,cell_list,OutputC_norm):
    T = calc_tension(edge,para,edge_list,1.0)
    P = calc_pressure(cell,para,cell_list,1.0)
    Tin_mean=np.mean(T)
    c_norm=1/Tin_mean
    T=c_norm*T
    P=c_norm*P
    P=P - np.mean(P)
    F = np.append(T,P)
    resi = np.dot(MM_in,F)
    if(OutputC_norm == 0):
        return resi
    else:
        return [c_norm,np.mean(T)]

def fitting(initial_value, bound, MM_in,edge,cell,edge_list,cell_list):

    kargs =  [MM_in,edge,cell,edge_list,cell_list,0]
    output = least_squares(calc_residu,initial_value,args = kargs,bounds = bound,verbose = 2)
    c_norm = calc_residu(output.x,MM_in,edge,cell,edge_list,cell_list,1)
    return [output.x,c_norm]

def write_output(SuiteiType,fileout,data,edge,cell,tension,pressure,E_in,C_in,CELL_NUMBER,E_NUM):
    f_out=open(fileout+"/"+SuiteiType+"out_"+data+".dat",'w')
    f_out.write('# %s \n' % (data) )
    f_out.write('### type '+SuiteiType+'  \n')
    f_out.write('### E_NUM % d \n' % (E_NUM))
    f_out.write('### CELL_NUMBER % d \n' % (CELL_NUMBER))
    for (i, e) in enumerate(edge):
        if(i in E_in):
            f_out.write("E %5d % e %e 1\n" % (i,tension[i],edge[i].degree) )
        else:
            f_out.write("E %5d % e %e 0\n" % (i,tension[i],edge[i].degree) )
    f_out.write("\n")
    for (i, c) in enumerate(cell):
        f_out.write("C %5d % e\n" % (i,pressure[i]) )
    
    f_out.write("\n")
    f_out.close()


def ForcePlot(SuiteiType,fileout,data,edge,cell,T,P,E_NUM,CELL_NUMBER,E_in,C_in,Trange,Prange,display):
    marker = list()
    for i in range(E_NUM):
        if edge[i].degree <= np.pi/8:
            marker.append('.r')
        elif edge[i].degree >= 7*np.pi/8:
            marker.append('.r')
        elif np.pi/8 < edge[i].degree <= 3*np.pi/8:
            marker.append('.g')
        elif 3*np.pi/8 < edge[i].degree <= 5*np.pi/8:
            marker.append('.b')
        else:
            marker.append('.m')
     
    
    plt.title(SuiteiType+":Edge length vs Tension\n"+data)
    plt.xlabel("edge length")
    plt.ylabel("Tension")
    plt.ylim([Trange[0],Trange[1]])
    for i in E_in:
        plt.plot(edge[i].dist,T[i],marker[i])
    
    plt.savefig(fileout+SuiteiType+"_DvsT.eps")
    if(display == 1.0):
        plt.pause(1.0e-10)
    plt.close()
    
    plt.title(SuiteiType+":Cell area vs Pressure\n"+data)
    plt.xlabel("cell area")
    plt.ylabel("Pressure")
    plt.ylim([Prange[0],Prange[1]])
    for i in list(C_in):
        plt.plot(cell[i].area,P[i],'.k')
    plt.savefig(fileout+SuiteiType+"_AvsP.eps")
    if(display == 1.0):
        plt.pause(1.0e-10)
    plt.close()    

def CompareForceEstimation(data,fileout,edge,T1,T2,P1,P2,type1,type2,E_NUM,E_in,C_in,Trange, Prange,display):
    marker = list()
    for i in range(E_NUM):
        if edge[i].degree <= np.pi/8:
            marker.append('.r')
        elif edge[i].degree >= 7*np.pi/8:
            marker.append('.r')
        elif np.pi/8 < edge[i].degree <= 3*np.pi/8:
            marker.append('.g')
        elif 3*np.pi/8 < edge[i].degree <= 5*np.pi/8:
            marker.append('.b')
        else:
            marker.append('.m')
    print("#marker = "+str(len(marker))+"\n#T1 = "+str(len(T1))+"\nT2 = "+str(len(T2)))
    #plot tension
    plt.ylim([Trange[0],Trange[1]])
    plt.xlim([Trange[0],Trange[1]])
    for i in E_in:
        plt.plot(T1[i],T2[i],marker[i])
    plt.title(type1+"vs"+type2+":tension\n"+data)
    plt.xlabel(type1)
    plt.ylabel(type2)
    plt.savefig(fileout+type1+"vs"+type2+"_T.png")
    if(display == 1.0):
        plt.pause(1.0e-10)
    plt.close()
    
    #plot pressure
    plt.ylim([Prange[0],Prange[1]])
    plt.xlim([Prange[0],Prange[1]])
    plt.plot(P1,P2,'.k')
    plt.title (type1+" vs "+type2+":Pressure\n"+data)
    plt.xlabel(type1)
    plt.ylabel(type2)
    plt.savefig(fileout+type1+"vs"+type2+"_P.png")
    if(display == 1.0):
        plt.pause(1.0e-10)
    plt.close()
    
    CT = np.corrcoef([T1[i] for i in E_in],[T2[i] for i in E_in])[0,1]
    CP = np.corrcoef([P1[i] for i in C_in],[P2[i] for i in C_in])[0,1]
    return  [CT,CP]
       

def calc_stress(edge,cell,T,P,edge_list,cell_list):
    AF=0
    for i in cell_list:
        AF += P[i]*cell[i].area
    AF = -1*AF*np.eye(2)
    
    LF = 0
    for i in edge_list:
        LF += (T[i]/edge[i].dist)*np.outer([edge[i].dx,edge[i].dy],[edge[i].dx,edge[i].dy])
    
    Stress = (AF + LF)/np.sum([cell[i].area for i in C_in])
    return Stress
    
#functions for test 


def residu2(para):
    resi = NOgi.calc_residu(para,MM_in,edge,cell,E_in,C_in,0)
    return np.linalg.norm(resi[0])
    
    def min_residu(hajime,owari,interval):
        para_hist = np.zeros([int((owari-hajime)/interval + 1),9])
        r_min = 100000000
        i_min = 0
        for i in np.arange(hajime,owari,interval):
            #initial_value = np.array([0.1,0,0.2,0.1,0,0.1,0.1,i,0.1])
            initial_value = np.full(9,i)
            initial_value[6] = 10
            para,fit_out = NOgi.fitting(initial_value,bound,MM_in,edge,cell,E_in,C_in)
            T = np.array(NOgi.calc_tension(edge,para,E_in,1))
            Tin_mean=np.mean(T)
            c_norm=1/Tin_mean
            r = residu2(para)/np.abs(c_norm)
            para_hist[int(i/interval)] = para
            if (r_min > r):
                r_min = r
                i_min = i
            plt.plot(i,residu2(para)/np.abs(c_norm),".b")
            #para = np.append(1,para)
        return [i_min,r_min,c_norm,para,para_hist]

#グラフの概形をプロット
def graph_shape(para,c_norm,lmax):
    l = np.linspace(0,lmax)
    Th = tension(l,0,para)
    Tv = tension(l,np.pi/2,para)
    plt.plot(l,Th,"r")
    plt.plot(l,Tv,"b")
    
