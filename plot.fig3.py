# -*- coding: utf-8 -*-
import sys
import os
import subprocess

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import simps
from scipy.integrate import cumtrapz
from scipy import interpolate
import scipy.ndimage as ndimage
import scipy.signal as signal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon

from pylab import *
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans

import my_netcdf as netcdf # netcdf
from scipy.interpolate import griddata
import copy 
import matplotlib.ticker as mticker
import matplotlib.path as mpath

# ---------------------------------------------------- #
font_prop = mpl.font_manager.FontProperties()
undef = -999.
# ---------------------------------------------------- #
#----functions
def draw_2Dlev_box(xlonlat,zlev,ax,color='k',lw=2,linestyle='--'):
    wlon = xlonlat[0]
    elon = xlonlat[1]
    slat = zlev[0]
    nlat = zlev[1]
    wlons = [wlon,wlon,wlon,elon]
    elons = [elon,elon,wlon,elon]
    slats = [slat,nlat,slat,slat]
    nlats = [slat,nlat,nlat,nlat]        
    for x1,x2,y1,y2 in zip(wlons,elons,slats,nlats):
        x = np.linspace(x1,x2,100)
        y = np.linspace(y1,y2,100)
        ax.plot(x,y,linewidth=lw,color=color,linestyle=linestyle)
        
def get_data(filename,varname):
    rnc = netcdf.read_nc_each(filename)
    lon,lat = rnc.get_lonlat()
    lev = rnc.getdim_each('lev')
    tmpdata = rnc.getvar(varname)
    rnc.close()
    lon = lon.data
    lat = lat.data
    lev = lev.data
    tmpdata = tmpdata[0,:,:,:]
    
    #lon = np.concatenate([lon,lon[0:2]])
    #tmpdata = np.concatenate([tmpdata,tmpdata[:,:,0:2]],axis=2)
    loncopy = copy.copy(lon)
    lon = np.concatenate([lon,loncopy[0:40]+360.])
    tmpdata = np.concatenate([tmpdata,tmpdata[:,:,0:40]],axis=2)
    tmpdata = np.ma.array(tmpdata,fill_value=undef,mask=tmpdata==undef)
    return tmpdata,lon,lat,lev


def boxplot_each(ax,trend_listm,trend_listg,boxwidth=0.5):
    ax.boxplot( np.array(trend_listm)*10, vert=True, labels=['HIST'],positions=[1],showfliers=False,
                patch_artist=True,  # 細かい設定をできるようにする
                widths=boxwidth,  # boxの幅の設定
                boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                              edgecolor='black', linewidth=1),  # boxの枠線の設定
                #showmeans=True, # medianをmeanに変更
                medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                usermedians=[np.mean(trend_listm)*10], # medianをmeanに変更
                whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
               )
    ax.boxplot( np.array(trend_listg)*10, vert=True, labels=['NAGA'],positions=[2],showfliers=False,
                patch_artist=True,  # 細かい設定をできるようにする
                widths=boxwidth,  # boxの幅の設定
                boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                              edgecolor='black', linewidth=1),  # boxの枠線の設定
                #showmeans=True, # medianをmeanに変更
                medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                usermedians=[np.mean(trend_listg)*10], # medianをmeanに変更
                whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
               )
    
    xs1,xs2 = [],[]
    xs1.append(np.random.normal(1, 0.04, len(trend_listm)))
    xs2.append(np.random.normal(2, 0.04, len(trend_listg)))
    #xs1 = [1]*10
    #xs2 = [2]*10    
    ax.scatter(xs1,np.array(trend_listm)*10,edgecolor='cornflowerblue',facecolor='w',s=10,alpha=1,marker='o',linewidths=1)
    ax.scatter(xs2,np.array(trend_listg)*10,edgecolor='salmon',facecolor='w',s=10,alpha=1,marker='o',linewidths=1)
    
#----main
if __name__ == '__main__':
    #--------------------------------------#
    # read data

    data_path = "./data/trend/"
    #----NAGA
    tos,lons,lats,levs = get_data(data_path + 'pace/to_trend_pace.nc','to')
    uos,_,_,levs2 = get_data(data_path + 'pace/ut_DJFtrend_pace.nc','ut')
    uoss,_,_,_ = get_data(data_path + 'pace/ut_DJFtrend_pace_sig.nc','ut')    
    vos,_,_,_ = get_data(data_path + 'pace/vt_DJFtrend_pace.nc','vt')
    voss,_,_,_ = get_data(data_path + 'pace/vt_DJFtrend_pace_sig.nc','vt')    
    #----HIST
    tom,lonm,latm,levm = get_data(data_path + 'miroc6/to_trend_miroc6.nc','to')
    uom,_,_,levm2 = get_data(data_path + 'miroc6/ut_DJFtrend_miroc6.nc','ut')
    uoms,_,_,_ = get_data(data_path + 'miroc6/ut_DJFtrend_miroc6_sig.nc','ut')    
    vom,_,_,_ = get_data(data_path + 'miroc6/vt_DJFtrend_miroc6.nc','vt')
    voms,_,_,_ = get_data(data_path + 'miroc6/vt_DJFtrend_miroc6_sig.nc','vt')
    
    #---heat transport
    dfm_nbso = pd.read_csv('data_ht/ht_nbso_miroc6.csv')
    dfg_nbso = pd.read_csv('data_ht/ht_nbso_pace.csv')
    dfm_nor = pd.read_csv('data_ht/ht_nor_miroc6.csv')
    dfg_nor = pd.read_csv('data_ht/ht_nor_pace.csv')

    bnm_list = [ dfm_nbso[x].values[0] for x in dfm_nbso.columns ]
    bns_list = [ dfg_nbso[x].values[0] for x in dfg_nbso.columns ]
    nm_list = [ dfm_nor[x].values[0] for x in dfm_nor.columns ]
    ns_list = [ dfg_nor[x].values[0] for x in dfg_nor.columns ]        
    
    #--------------------------------------#
    #----trend per year -> per decade
    tos *= 10.
    uos *= 10.
    uoss *= 10.
    vos *= 10.
    voss *= 10.    

    tom *= 10.
    uom *= 10.
    uoms *= 10.    
    vom *= 10.
    voms *= 10.    
    
    cut_lev = 1000
    cut_lev2 = 500
    #-------------------------------------#
    # u,t at 20.5 E
    ilon = np.where(lons==20.5)[0]
    ilat = np.where((lats>69)&(lats<86))[0]
    ilev = np.where(levs2<cut_lev)[0]
    
    uos20E = uos[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]
    uom20E = uom[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]
    uos20Es = uoss[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]
    uom20Es = uoms[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]    

    tos20E = tos[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]
    tom20E = tom[ilev[0]:ilev[-1]+1,ilat[0]:ilat[-1]+1,ilon[0]]
    
    lat20E = lats[ilat[0]:ilat[-1]+1]
    lev20E = levs[ilev[0]:ilev[-1]+1]    
    
    #-------------------------------------#
    # v,t at 70.5N
    ilat = np.where(lats==70.5)[0]
    ilon = np.where((lons>310.)&(lons<380.))[0]
    ilev = np.where(levs<cut_lev)[0]
    
    vos65N = vos[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]
    vom65N = vom[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]
    vos65Ns = voss[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]
    vom65Ns = voms[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]    

    tos65N = tos[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]
    tom65N = tom[ilev[0]:ilev[-1]+1,ilat[0],ilon[0]:ilon[-1]+1]
    
    lon65N = lons[ilon[0]:ilon[-1]+1]-360.
    lev65N = levs[ilev[0]:ilev[-1]+1]
    
    #-----------------------------------------------------------#    
    #-----plot
    fontsize  = 10
    fontsize2 = 8
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #fig = plt.figure(figsize=(10,7))
    fig = plt.figure(figsize=(10,5))
    
    #----make gridspec & axis
    # [10,3,10,3,10,2]  -> 35 
    # [7,3,7,3,7,3,7,1] -> 38
    #gs_master = GridSpec(nrows=39, ncols=44, height_ratios=[1]*39,hspace=0.,wspace=0.)
    gs_master = GridSpec(nrows=26, ncols=44, height_ratios=[1]*26,hspace=0.,wspace=0.) 
    
    gs_1 = GridSpecFromSubplotSpec(nrows=10, ncols=10, subplot_spec=gs_master[0:10,0:10]) #
    ax1 = fig.add_subplot(gs_1[:, :])
    gs_2 = GridSpecFromSubplotSpec(nrows=10, ncols=10, subplot_spec=gs_master[0:10,13:23])
    ax2 = fig.add_subplot(gs_2[:, :])
    
    gs_3 = GridSpecFromSubplotSpec(nrows=7, ncols=10, subplot_spec=gs_master[0:10,26:34])
    ax3 = fig.add_subplot(gs_3[:, :])
    
    gs_5 = GridSpecFromSubplotSpec(nrows=10, ncols=10, subplot_spec=gs_master[13:23,0:10]) #
    ax5 = fig.add_subplot(gs_5[:, :])
    gs_6 = GridSpecFromSubplotSpec(nrows=10, ncols=10, subplot_spec=gs_master[13:23,13:23])
    ax6 = fig.add_subplot(gs_6[:, :])
    
    gs_7 = GridSpecFromSubplotSpec(nrows=7, ncols=10, subplot_spec=gs_master[13:23,26:34])
    ax7 = fig.add_subplot(gs_7[:, :])

    gs_12 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[-1,0:23]) #
    ax12 = fig.add_subplot(gs_12[:, :])    
    
    #----ylim
    ax1.set_ylim([cut_lev2,0])
    ax2.set_ylim([cut_lev2,0])
    ax5.set_ylim([cut_lev,0])
    ax6.set_ylim([cut_lev,0])
    ax3.set_ylim([-1,3])
    ax7.set_ylim([-10,10])
    
    #----xlim
    ax1.set_xlim([70,80])
    ax2.set_xlim([70,80])
    ax5.set_xlim([-25,25])
    ax6.set_xlim([-25,25])

    #----plot
    cmap = plt.cm.seismic
    cmap = plt.cm.coolwarm
    
    clevs=np.linspace(-1,1,21)
    clevs2=np.linspace(-1,1,21)    
    clevs3=np.linspace(-1,1,21)
    
    mpl.rcParams['hatch.linewidth'] = 0.2   # hatch linewidth
    mpl.rcParams['hatch.color'] = 'black' # hatch color
    
    #################################################
    #----BSO
    cf1 = ax1.contourf(lat20E,lev20E,uos20E,cmap=cmap,extend='both',levels=clevs)
    cs_hatch1 = ax1.contourf(lat20E,lev20E,uos20Es,colors='none',extend='both',levels=clevs,alpha=0,hatches=['/////'])
    
    cf2 = ax2.contourf(lat20E,lev20E,uos20E-uom20E,cmap=cmap,extend='both',levels=clevs)
    cs_hatch2 = ax2.contourf(lat20E,lev20E,uos20Es,colors='none',extend='both',levels=clevs,alpha=0,hatches=['/////'])    
    
    cc1 = ax1.contour(lat20E,lev20E,tos20E,colors='gray',extend='both',levels=clevs2,linewidths=1)
    cc2 = ax2.contour(lat20E,lev20E,tos20E-tom20E,colors='gray',extend='both',levels=clevs2,linewidths=1)
    ax1.clabel(cc1, fontsize=6, fmt='%1.1f')
    ax2.clabel(cc2, fontsize=6, fmt='%1.1f')    
    
    # boxplots    
    boxplot_each(ax3,bnm_list,bns_list)
    
    #################################################    
    #----
    cf5 = ax5.contourf(lon65N,lev65N,vos65N,cmap=cmap,extend='both',levels=clevs3)
    cs_hatch5 = ax5.contourf(lon65N,lev65N,vos65Ns,colors='none',extend='both',levels=clevs3,alpha=0,hatches=['/////'])    
    
    cf6 = ax6.contourf(lon65N,lev65N,vos65N-vom65N,cmap=cmap,extend='both',levels=clevs3)
    cs_hatch6 = ax6.contourf(lon65N,lev65N,vos65Ns,colors='none',extend='both',levels=clevs3,alpha=0,hatches=['/////'])    
    
    cc5 = ax5.contour(lon65N,lev65N,tos65N,colors='gray',extend='both',levels=clevs2,linewidths=1)
    cc6 = ax6.contour(lon65N,lev65N,tos65N-tom65N,colors='gray',extend='both',levels=clevs2,linewidths=1)
    ax5.clabel(cc5, fontsize=6, fmt='%1.1f')
    ax6.clabel(cc6, fontsize=6, fmt='%1.1f')
    
    # boxplots    
    boxplot_each(ax7,nm_list,ns_list)
    
    #################################################    
    #----xlabel
    xlat      = [70,75,80,85]
    xlat_label = ["70N","75N","80N","85N"]
    ax1.set_xticks(xlat)
    ax1.set_xticklabels(xlat_label, rotation=0, fontsize=fontsize2)
    ax2.set_xticks(xlat)
    ax2.set_xticklabels(xlat_label, rotation=0, fontsize=fontsize2)

    xlat2      = [-40,-20,0,20,40]
    xlat_label2 = ["40W","20W","0","20E","40E"]
    ax5.set_xticks(xlat2)
    ax5.set_xticklabels(xlat_label2, rotation=0, fontsize=fontsize2)
    ax6.set_xticks(xlat2)
    ax6.set_xticklabels(xlat_label2, rotation=0, fontsize=fontsize2)
    
    ax1.set_xlim([70,80])
    ax2.set_xlim([70,80])
    ax5.set_xlim([-25,25])
    ax6.set_xlim([-25,25])
    
    #----colorbar
    cb12 = plt.colorbar(cf1,ax=ax12, orientation="horizontal",fraction=0.8)#,ticks=[0,20,40,60,80,100])
    ax12.axis("off")                
    cb12.set_label('[K cm $s^{-1}$ decade$^{-1}$]',fontsize=10)
    
    #----draw box
    xlonlat=[74.,79]
    zlev=[0,200]
    draw_2Dlev_box(xlonlat,zlev,ax1,color='k',lw=2,linestyle=':')
    draw_2Dlev_box(xlonlat,zlev,ax2,color='k',lw=2,linestyle=':')
    
    xlonlat=[-12,8]
    zlev=[0,435]
    draw_2Dlev_box(xlonlat,zlev,ax5,color='k',lw=2,linestyle=':')
    draw_2Dlev_box(xlonlat,zlev,ax6,color='k',lw=2,linestyle=':')
    
    #----title    
    ax1.set_title('(a) 20E (NAGA) ',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax2.set_title('(b) 20E (NAGA-HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax3.set_title('(c) HT$_{BSO}$ trend [TW decade$^{-1}$]',fontsize=fontsize2, y=1.,x=0.0,loc='left')        
    ax5.set_title('(d) 70N (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax6.set_title('(e) 70N (NAGA-HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax7.set_title('(f) HT$_{Norway}$ trend [TW decade$^{-1}$]',fontsize=fontsize2, y=1.,x=0.0,loc='left')    

    # save
    #plt.show()
    #sys.exit()
    plt.savefig('fig3.pdf', bbox_inches="tight", pad_inches=0)
    subprocess.call('open fig3.pdf',shell=True)
    plt.close()    
    
