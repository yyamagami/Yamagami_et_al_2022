# -*- coding: utf-8 -*-
import numpy as np
from numpy import sin,cos
from numpy import dtype
#import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset, date2index, num2date, date2num
import datetime
import sys
import os

class read_nc_each():

    def __init__(self,file):
        self.variables = None
        self.time = None
        self.lon = None
        self.lat = None
        self.lev = None
        self.calendar = None
        self.units = None
        self.var = None
        self.varc = None
        self.f = netCDF4.Dataset(file, 'r')
        
    def getdim(self):
        f = self.f
        self.lon = f.variables['lon'][:]
        self.lat = f.variables['lat'][:]
        self.lev = f.variables['lev'][:]
        self.time = f.variables['time'][:]
        self.units = f.variables['time'].units
        return self.lon,self.lat,self.lev,self.time

    def get_lonlat(self):
        f = self.f
        self.lon = f.variables['lon'][:]
        self.lat = f.variables['lat'][:]
        return self.lon,self.lat
    
    def getdim_each(self,var):
        f = self.f
        return f.variables[var][:]
    
    def getvar(self,varname):
        f = self.f
        return f.variables[varname][:,:,:,:]

    def getvar2d(self,varname,itime):
        f = self.f
        return f.variables[varname][itime,:,:]

    def getvar2dall(self,varname):
        f = self.f
        return f.variables[varname][:,:,:]    

    def getvar_time(self,varname,itime):
        f = self.f
        return f.variables[varname][itime,:,:,:]

    def getvar_eq(self,varname,ilat):
        f = self.f
        return f.variables[varname][:,:,ilat,:]
    
    def getvar_onegrid(self,varname,ilon,ilat,ilev):
        f = self.f
        return f.variables[varname][:,ilev,ilat,ilon]

    def close(self):
        self.f.close()

