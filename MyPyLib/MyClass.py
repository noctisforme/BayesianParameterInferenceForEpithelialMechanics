# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:43:23 2020

@author: Goshi
"""
from pandas import Series
import numpy as np

class Parameter:
    def __init__(self,Parameter_names):
        self.names = Parameter_names
        self.dim = len(self.names)
        self.min = Series(np.full(self.dim,-np.inf),index=self.names)
        self.max = Series(np.full(self.dim,np.inf),index=self.names)
        self.est = Series(np.zeros(self.dim),index=self.names)
   
    def bound(self):
       return (self.min.values,self.max.values)
   
    def change_est(self,estimated_value):
        self.est = Series(estimated_value,index=self.names)
    def set_ini(self):
        self.ini = (para.max + para.min)/2
        self.ini = self.ini.fillna(1)
        return self.ini.values
   
