# -*- coding: utf-8 -*-

import sys
import os
import subprocess

import matplotlib as mpl

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import simps
from scipy.integrate import cumtrapz
from scipy import interpolate
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon

from pylab import *
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans

import my_netcdf as netcdf# netcdf
from scipy.interpolate import griddata
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import copy
# ---------------------------------------------------- #
font_prop = mpl.font_manager.FontProperties()
undef = -999

# ---------------------------------------------------- #
#----functions
def draw_lonlat_box(area,m,ax,color='k',lw=2,linestyle='-'):
    wlon = area[0]
    elon = area[1]
    slat = area[2]
    nlat = area[3]
    #
    wlons = [wlon,wlon,wlon,elon]
    elons = [elon,elon,wlon,elon]
    slats = [slat,nlat,slat,slat]
    nlats = [slat,nlat,nlat,nlat]        
    for x1,x2,y1,y2 in zip(wlons,elons,slats,nlats):
        x = np.linspace(x1,x2,100)
        y = np.linspace(y1,y2,100)
        mx,my = m(x,y)
        m.plot(mx,my,linewidth=lw,color=color,linestyle=linestyle)

def get_data(filename,varname):
    rnc = netcdf.read_nc_each(filename)
    lon,lat = rnc.get_lonlat()
    lev = rnc.getdim_each('lev')
    tmpdata = rnc.getvar(varname)
    rnc.close()
    lon = lon.data
    lat = lat.data
    lev = lev.data
    tmpdata = tmpdata[:,0,:,:]
    tmpdata = np.ma.array(tmpdata.data, fill_value=-999., mask=tmpdata.data==-999.) #mask
    return tmpdata,lon,lat,lev


def plot_NA_each(utin,vtin,toin,tc,lons,lats,varname,fig,ax,plot_area=[-100,40,30,85],scale=1,veclen=1,fontsize=10,print_veclen=False,plot_pulse=False,vintlon=5,vintlat=3,headwidth=10,headlength=10,width=0.002):

    if plot_pulse:
        ut = utin
        vt = vtin
        to = toin
    else:
        #---- per decade
        ut = utin*10.
        vt = vtin*10.
        to = toin*10.
        sho = tc*10.

    #----plot settings
    cmap_sst = plt.cm.RdBu_r
    if plot_pulse:
        clevs_sic=np.linspace(-0.1,0.1,21)
        #clevs_sst=np.linspace(-3,3,21)
        clevs_sst=np.arange(-1.6,1.7,0.4)
    else:
        clevs_sic=np.linspace(-0.1,0.1,21)
        clevs_sst=np.linspace(-1.2,1.2,21)
        clevs_sho=np.arange(-100,100,2)

    lonos, latos = np.meshgrid(lons,lats) # make lats and lons

    #----make basemap
    m = Basemap(resolution='l',projection='lcc', lat_0=60, lon_0=-40,width=10000000, height=8000000,ax=ax)
    
    m.drawmapboundary(fill_color='white',color='black',linewidth=1.0)
    m.fillcontinents(color='gray',lake_color='gray')
    m.drawcoastlines()

    #----plot temperature
    mlons, mlats = m(lonos,latos)
    cs1 = m.contourf(mlons,mlats,to,cmap=cmap_sst,extend='both',levels=clevs_sst,alpha=1)#latlon=True)

    #----plot u, v
    q1 = m.quiver(mlons[::vintlat,::vintlon],mlats[::vintlat,::vintlon],ut[::vintlat,::vintlon],vt[::vintlat,::vintlon],angles='xy'
                  ,units='width',scale=scale,alpha=1,color='k',pivot='mid',headwidth=headwidth,headlength=headlength,width=width)#,latlon=True) # scale is just a scale for figsize and ua&va

    if print_veclen:
        font_prop.set_size(fontsize)
        if plot_pulse:
            qk = plt.quiverkey(q1, 1., 1.06, veclen, '{0} '.format(veclen)+'cm $s^{{-}1}$', labelpos='E') # 1 means 1 size of vector in m/s??
        else:
            qk = plt.quiverkey(q1, 0.8, 1.06, veclen, '{0} '.format(veclen)+'cm $s^{{-}1}$ decade$^{-1}$', labelpos='E') # 1 means1 size of vector in m/s??
    return m,cs1,q1

def plot_trend_obs(toin,lons,lats,fig,ax,plot_area=[-100,40,30,85],fontsize=10):

    to = toin*10.
    cmap_sst = plt.cm.RdBu_r    
    clevs_sst=np.arange(-0.5,0.6,0.1)

    lonos, latos = np.meshgrid(lons,lats) # make lats and lons
    m = Basemap(llcrnrlon=plot_area[0],llcrnrlat=plot_area[2],urcrnrlon=plot_area[1],urcrnrlat=plot_area[3],projection='cyl',resolution='c',ax=ax)
    m.drawmapboundary(fill_color='white',color='black',linewidth=1.0)
    m.fillcontinents(color='gray',lake_color='gray')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,99.,20.),labels=[True,False,False,False],fontsize=6)
    m.drawmeridians(np.arange(-180.,180.,40.),labels=[False,False,False,True],fontsize=6)
    
    #----plot temperature
    mlons, mlats = m(lonos,latos)
    cs1 = m.contourf(mlons,mlats,to,cmap=cmap_sst,extend='both',levels=clevs_sst,alpha=1,latlon=True)
    return m,cs1

#----main
if __name__ == '__main__':
    
    syear = 1970
    eyear = 1979
    loopnum = 12
    
    ylim = [-0.4,0.4]
    yticks = [-0.3,0,0.3]
    
    #---------------------------------------
    # read observation data
    pathh = "./data_nagac/"
    sst_trendc,lonc,latc,_ = get_data(pathh+'cobesst_sst_djf_trend_1970_2017.nc','trend')
    sst_trendc = sst_trendc[0]
    #area=[280,330,30,50] # nudging box
    area=[275,335,25,55] # nudging box
    lonidx1 = np.where((lonc >=area[0] ) & (lonc <=area[1] ))[0]
    latidx1 = np.where((latc >=area[2] ) & (latc <=area[3] ))[0]
    
    dummy = np.zeros(sst_trendc.shape)
    dummy[latidx1[0]:latidx1[-1]+1,lonidx1[0]:lonidx1[-1]+1] = 1.
    sst_trendc = np.ma.array(sst_trendc,fill_value=undef,mask=dummy==0)
    
    #---------------------------------------
    to_inte_all,lons,lats,_ = get_data("data_nagac/to_pace.nc",'to')
    uo_inte_all,_,_,_ = get_data("data_nagac/uo_pace.nc",'uo')
    vo_inte_all,_,_,_ = get_data("data_nagac/vo_pace.nc",'vo')

    tom_inte_all,_,_,_ = get_data("data_nagac/to_miroc6.nc",'to')
    uom_inte_all,_,_,_ = get_data("data_nagac/uo_miroc6.nc",'uo')
    vom_inte_all,_,_,_ = get_data("data_nagac/vo_miroc6.nc",'vo')    

    #----difference
    to_inte_all[:,:,:] -= tom_inte_all[:,:,:]
    uo_inte_all[:,:,:] -= uom_inte_all[:,:,:]
    vo_inte_all[:,:,:] -= vom_inte_all[:,:,:]    
    
    #----area averaged temperature
    df = pd.read_csv('data_nagac/temp_box_average.csv')
    to1s = df['box1_nagac'].values
    to2s = df['box2_nagac'].values
    to3s = df['box3_nagac'].values
    to4s = df['box4_nagac'].values

    to1m = df['box1_hist'].values
    to2m = df['box2_hist'].values
    to3m = df['box3_hist'].values
    to4m = df['box4_hist'].values        

    #----box
    box1=[280,330,30,50] # nudging box
    box2=[330,370,50,65] # northeastern atlantic
    box3=[340,380,65,80] # ノルウェー海
    box4=[380,430,65,85] # Barents-Kara Sea
    
    #--------------------------------------#
    #----plot
    plot_area=[-90,40,20,80]
    vintlon=4
    vintlat=4
    
    fontsize  = 10
    fontsize2 = 10
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #----make figure
    fig = plt.figure(figsize=(10,11))

    #----make gridspec & axis
    gs_master = GridSpec(nrows=8, ncols=22, height_ratios=[1,0.1,0.3,1,0.1,1,0.1,1],hspace=0.1,wspace=0.1)

    gs0 = GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=gs_master[0,0:7]) #
    ax0  = fig.add_subplot(gs0[:, :])
    gs0c = GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=gs_master[1,0:7]) #
    ax0c  = fig.add_subplot(gs0c[:, :])
    
    gs1 = GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=gs_master[0,8:15]) #
    ax1  = fig.add_subplot(gs1[:, :])
    gs2 = GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=gs_master[0,15:22]) #
    ax2  = fig.add_subplot(gs2[:, :])
    gs12c = GridSpecFromSubplotSpec(nrows=1,ncols=4,subplot_spec=gs_master[1,11:18])
    ax12c = fig.add_subplot(gs12c[:,:])
    
    gstext = GridSpecFromSubplotSpec(nrows=1, ncols=22, subplot_spec=gs_master[2,:]) #
    axtext = fig.add_subplot(gstext[:, :])        
    
    gs_p1 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,0:7]) #
    axp1 = fig.add_subplot(gs_p1[:, :])
    gs_p2 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,7:14])
    axp2 = fig.add_subplot(gs_p2[:, :])    
    gs_p3 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,14:21]) #
    axp3 = fig.add_subplot(gs_p3[:, :])
    gs_p4 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[5,0:7])
    axp4 = fig.add_subplot(gs_p4[:, :])
    gs_p5 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[5,7:14])
    axp5 = fig.add_subplot(gs_p5[:, :])
    gs_p6 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[5,14:21])
    axp6 = fig.add_subplot(gs_p6[:, :])
    gs_p7 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[7,0:7])
    axp7 = fig.add_subplot(gs_p7[:, :])
    gs_p8 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[7,7:14])
    axp8 = fig.add_subplot(gs_p8[:, :])
    gs_p9 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[7,14:21])
    axp9 = fig.add_subplot(gs_p9[:, :])            
    
    gs_pc = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[3:,-1]) #
    axpc = fig.add_subplot(gs_pc[:, :])
    
    #-----------------------------#
    #-----plot temp series
    years=np.arange(syear,eyear+1,1)
    toboxm = np.transpose(np.array([to1m,to2m,to3m,to4m]))
    toboxs = np.transpose(np.array([to1s,to2s,to3s,to4s]))
    tmpyears = np.arange(1970,1981,1)
    Y, X = np.meshgrid([0,1,2,3,4],tmpyears)
    
    cmap = plt.cm.coolwarm
    norm = BoundaryNorm(np.arange(-0.4,0.41,0.05), ncolors=cmap.N, clip=True)
    ccm1 = ax1.pcolormesh(X,Y,toboxm,cmap=cmap,norm=norm)
    _    = ax2.pcolormesh(X,Y,toboxs,cmap=cmap,norm=norm)
    
    cc12=plt.colorbar(ccm1,ax=ax12c, orientation="horizontal",fraction=0.4,ticks=[-0.4,-0.2,0,0.2,0.4])
    cc12.ax.tick_params(labelsize=6)
    cc12.set_label('[$^{\circ}$C]',fontsize=6)            
    ax12c.axis('off')
    
    ax1.set_ylim([0,4])
    ax2.set_ylim([0,4])
    ax1.set_xlim([1970,1980])
    ax2.set_xlim([1970,1980])
    
    ax1.set_xticks(np.arange(1970.5,1980,1),minor=True)
    ax2.set_xticks(np.arange(1970.5,1980,1),minor=True)
    
    ylim_range = np.arange(0.5,4,1)
    ax1.set_yticks(ylim_range,minor=True)
    ax2.set_yticks(ylim_range,minor=True)
    
    series_xticks=[1970.5,1972.5,1974.5,1976.5,1978.5]
    ax1.set_xticks(series_xticks)
    ax2.set_xticks(series_xticks)
    
    series_xticklabels=[1970,1972,1974,1976,1978]
    ax1.set_xticklabels(series_xticklabels,fontsize=7)
    ax2.set_xticklabels(series_xticklabels,fontsize=7)
    
    ax1.set_yticks([0.5,1.5,2.5,3.5])
    ax2.set_yticks([0.5,1.5,2.5,3.5])
    ax1.set_yticklabels(['Box1','Box2','Box3','Box4'],fontsize=7)
    ax2.set_yticklabels([],fontsize=7)
    
    ax1.grid(color='gray',linestyle='--',which='both', alpha=0.2)
    ax2.grid(color='gray',linestyle='--',which='both', alpha=0.2)
    
    fig.tight_layout()
    ax1.xaxis.set_label_coords(0.5, -0.18)    
    ax1.yaxis.set_label_coords(-0.2, 0.5)
    
    ax1.set_title("(b) Potential temperature anomaly (HIST)",fontsize=8, y=1,x=0.0,loc='left')
    ax2.set_title("(c) Potential temperature anomaly (NAGAc)",fontsize=8, y=1,x=0.0,loc='left')
    
    #---------------------------------------------#
    #---------------------------------------------#
    #---------------------------------------------#
    ax0.set_title('(a) Nudging SSTA (NAGAc)',fontsize=8, y=1,x=0.0,loc='left')    
    m0,cs0 = plot_trend_obs(sst_trendc,lonc,latc,fig,ax0,plot_area=[-100,0,0,70],fontsize=7)
    draw_lonlat_box([-80,-30,30,50],m0,ax0,color='k',lw=1)    
    draw_lonlat_box([-85,-25,25,55],m0,ax0,linestyle='dashed',color='black',lw=1)
    
    cc0=plt.colorbar(cs0,ax=ax0c, orientation="horizontal",fraction=0.4,ticks=[-0.4,-0.2,0,0.2,0.4])
    cc0.ax.tick_params(labelsize=6)
    cc0.set_label('[$^{\circ}$C]',fontsize=6)            
    ax0c.axis('off')
    #---------------------------------------------#
    #---------------------------------------------#
    #---------------------------------------------#    
    axtext.text(0,0.35,'Differences in annual velocity & potential temperature anomalies [0-198m] (NAGAc-HIST)',
                fontsize=10,horizontalalignment="left",verticalalignment="center")
    axtext.axis("off")

    #----plot
    scale = 30
    veclen = 3
        
    #----plot pulse experiment
    scale=65
    veclen=5        
    vintlon=4
    vintlat=3
        
    fontsize =9
    #----plot annual mean difference NAGA-HIST
    step=0 
    m_p1,cs_to_p1,qv_p1 = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp1,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=1
    m_p2,cs_to_p2,qv_p2 = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp2,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)

    step=2
    m_p3,cs_to_p3,qv_p3 = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp3,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=True,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=3
    m_p4,cs_to_p4,qv_p4 = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp4,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=4
    m_p5,_,_ = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp5,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=5
    m_p6,_,_ = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp6,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=6
    m_p7,_,_ = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp7,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)

    step=7
    m_p8,_,_ = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp8,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
    
    step=8
    m_p9,_,_ = plot_NA_each(uo_inte_all[step],vo_inte_all[step],to_inte_all[step],to_inte_all[step],lons,lats,'NAGAc-HIST',fig,axp9,plot_area=plot_area,scale=scale,veclen=veclen,fontsize=fontsize,vintlon=vintlon,vintlat=vintlat,print_veclen=False,plot_pulse=True,headwidth=14,headlength=14,width=0.003)
        
    #----colorbar
    cc_pc= plt.colorbar(cs_to_p1,ax=axpc, orientation="vertical",fraction=0.2,ticks=[-1.6,-0.8,0,0.8,1.6])        
    axpc.axis("off")
    cc_pc.set_label('[$^{\circ}$C]',fontsize=12)        
    
    #----draw box
    draw_lonlat_box([-80,-30,30,50],m_p1,axp1,color='m')
    draw_lonlat_box(box2,m_p1,axp1,color='m')
    draw_lonlat_box(box3,m_p1,axp1,color='m')
    draw_lonlat_box(box4,m_p1,axp1,color='m')
    
    #----draw title
    fontsize2 = 10
    axp1.set_title('(d) 1-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp2.set_title('(e) 2-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp3.set_title('(f) 3-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp4.set_title('(g) 4-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp5.set_title('(h) 5-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp6.set_title('(i) 6-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp7.set_title('(j) 7-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp8.set_title('(k) 8-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    axp9.set_title('(l) 9-yr',fontsize=fontsize2, y=1,x=0.0,loc='left')
    
    #plt.show()
    plt.savefig('fig5.pdf', bbox_inches="tight", pad_inches=0.2)
    subprocess.call('open fig5.pdf', shell=True)
    plt.close()
    
