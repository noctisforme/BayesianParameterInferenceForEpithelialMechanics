# -*- coding: utf8 -*-
"""
Created on Wed Jun 12 17:51:26 2019

@author: Goshi

2023/11/21　座標系の回転を導入

"""
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import MyPyLib.ForceInf_lib as FI
import sys


def scale_converter(x,y,edge,cell,sc_mean = 1.0,sc = 0,flip="",rotation = 0):
    E_NUM = len(edge)
    CELL_NUMBER = len(cell)
    dist_error = 1.0e-10
    
    if(sc == 0):
        lmean = np.mean([edge[i].dist for i in range(E_NUM)])
        sc = sc_mean/lmean
    if(flip == "v"):
        xsc = -sc*x
        ysc = sc*y
    elif(flip == "h"):
        xsc = sc*x
        ysc = -sc*y
    else:
        xsc = sc*x
        ysc = sc*y
        
    # FI.DrawCells(xsc,ysc,edge,cell)
    if rotation:
        xscr = xsc * np.cos(rotation) - ysc * np.sin(rotation)
        yscr = xsc * np.sin(rotation) + ysc * np.cos(rotation)
        xsc = xscr
        ysc = yscr
        
        """
        FI.DrawCells(xsc,ysc,edge,cell)
        plt.title(rotation)
        plt.pause(0.01)
        """
    edgesc = cp.deepcopy(edge)
    for i in range(E_NUM):
        edgesc[i].x1 = xsc[edgesc[i].junc1]
        edgesc[i].y1 = ysc[edgesc[i].junc1]
        edgesc[i].x2 = xsc[edgesc[i].junc2]
        edgesc[i].y2 = ysc[edgesc[i].junc2]
        edgesc[i].set_distance(xsc,ysc)
        if( edgesc[i].dist - sc*edge[i].dist > dist_error):
            print("dist_error = %f > %f" %(edgesc[i].dist - sc*edge[i].dist,dist_error))
            
    
    # lerror = [sc*edge[i].dist - edgesc[i].dist for i in range(E_NUM)]
    # print("lerror min: {}".format(min(lerror)))
    # print("lerror max: {}".format(max(lerror)))
    # plt.title("length")
    # plt.plot([edgesc[i].dist - sc*edge[i].dist for i in range(E_NUM)],label="ato - mae")
    # plt.legend()
    # plt.ylim(-1,1)
    # # plt.pause(0.01)
    # plt.show()
    # derror = [edgesc[i].degree - edge[i].degree - rotation for i in range(E_NUM)]
    # for i in range(E_NUM):
    #     if derror[i] <= -np.pi:
    #         derror[i] = derror[i] + np.pi
    # print("derror min: {}".format(min(derror)))
    # print("derror max: {}".format(max(derror)))
    # plt.title("degree")
    # plt.plot([edgesc[i].degree - edge[i].degree for i in range(E_NUM)],label="ato - mae")
    # plt.hlines(rotation,0,E_NUM,label="rotation",color="gray")
    # plt.legend()
    # plt.ylim(-np.pi,np.pi)
    # # plt.pause(0.01)
    # plt.show()
    cellsc = cp.deepcopy(cell)
    for i in range(CELL_NUMBER):
        cellsc[i].area = sc*sc*cell[i].area
        
    return [xsc,ysc,edgesc,cellsc,sc]

#def force_scale_converter(x,y,edge,cell,T,P,sc_mean = 1.0):