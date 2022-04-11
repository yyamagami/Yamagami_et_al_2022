# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd

import scipy as sp
from scipy.integrate import simps
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.stats as stats
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon

import datetime as dt
import copy
import glob
import subprocess

import decimal
from decimal import *

import matplotlib.cbook as cbook

# ---------------------------------------------------- #
#----make seasonal mean (like JJA mean)
def ten_trend_MonteCarlo(inlist,sampling_num,rng):
    outlist=[]
    for i in range(sampling_num):
        # choise 10-members by MonteCarlo method
        tmp_list = rng.choice(inlist, size=(10), replace=False,shuffle=False) 
        outlist.append(tmp_list)
    return outlist

def calc_mean_stats(tmp_stats,loopnum=1000):
    """
    bxpstats : list of dict
    A list of dictionaries containing the results for each column of data. Keys of each dictionary are the following:
    
    Key	Value Description
    label	tick label for the boxplot
    mean	arithemetic mean value
    med	        50th percentile
    q1	        first quartile (25th percentile)
    q3	        third quartile (75th percentile)
    cilo	lower notch around the median
    cihi	upper notch around the median
    whislo	end of the lower whisker
    whishi	end of the upper whisker
    fliers	outliers
    """
    stats_keys = ['mean','iqr','cilo','cihi','whishi','whislo','fliers','q1','med','q3']
    sic_listm_stats_MC={}
    for key in stats_keys:
        if key == 'fliers':
            sic_listm_stats_MC[key] = np.array([])
        else:
            tmp_list = [ tmp_stats[x][key] for x in range(loopnum) ] 
            sic_listm_stats_MC[key] = np.mean( tmp_list )
        #print(key,tmp_list,sic_listm_stats_MC[key])
    return [sic_listm_stats_MC]

def calc_mean_trend_std(tmp_trend_list,loopnum=1000):
    mean_list = [ np.mean(x) for x in tmp_trend_list ] 
    std_list  = [ np.std(x) for x in tmp_trend_list ] 
    return np.mean(mean_list),np.mean(std_list)

def plot_series_model_en(dfin,varname,color,label,ax,alpha=0.1,spread=True,trend_check=True,spread_alpha=0.05,percent=False,anom=True,trend_SST=False,facecolor=None):
    df = copy.copy(dfin)
    #----calculate trend of ensemble mean
    ytrend = df[varname+'_en'].values - signal.detrend(df[varname+'_en'].values)
    years = df.year.values
    trend_per_year = (ytrend[-1]-ytrend[0])/(years[-1]-years[0]+1)
    mean = np.mean(df[varname+'_en'].values)
    if anom:
        df[varname+'_en'] = df[varname+'_en'].values - mean
        ytrend = ytrend - mean
    
    #----plot ensemble spread
    if spread:
        ax.fill_between(df.year.values,df[varname+'_en'].values-df[varname+'_std'].values,df[varname+'_en'].values+df[varname+'_std'].values,alpha=spread_alpha,facecolor=facecolor)
    #----plot ensemble mean
    if trend_check:
        if trend_SST:
            ax.plot(df.year,df[varname+'_en'],alpha=0.8,color=color,linestyle='-',linewidth=0.2)
            ax.plot([],[],alpha=1,color=color,label='{0}({1:.2f})'.format(label,trend_per_year*10.),linestyle='--',linewidth=1)
        else:
            ax.plot(df.year,df[varname+'_en'],alpha=0.8,color=color,linestyle='-',linewidth=0.2)
            ax.plot([],[],alpha=1,color=color,label='{0}({1:.1f})'.format(label,trend_per_year*1.e3),linestyle='--',linewidth=1)
    else:
        ax.plot(df.year,df[varname+'_en'],alpha=alpha,color=color,label='{0}'.format(label),linewidth=1.5)
    #----plot trend
    if trend_check:
        ax.plot(df.year,ytrend,alpha=0.8,color=color,linestyle='--',linewidth=1.5)

    
def plot_series_obs(dfin,varname,color,label,ax,trend_check=True,anom=True,trend_SST=False):
    #----calculate trend of ensemble mean
    df = copy.copy(dfin)
    ytrend = df[varname].values - signal.detrend(df[varname].values)
    years = df.year.values
    trend_per_year = (ytrend[-1]-ytrend[0])/(years[-1]-years[0]+1)
    #----remove mean
    mean = np.mean(df[varname].values)
    if anom:    
        df[varname] = df[varname].values - mean
        ytrend = ytrend - mean
    #----plot
    if trend_check:
        if trend_SST:
            ax.plot(df.year,df[varname],alpha=0.8,color=color,linestyle='-',linewidth=0.15)
            ax.plot([],[],alpha=1,color=color,label='{0}({1:.2f})'.format(label,trend_per_year*10.),linestyle='--',linewidth=1)
        else:
            ax.plot(df.year,df[varname],alpha=0.8,color=color,linestyle='-',linewidth=0.15)
            ax.plot([],[],alpha=1,color=color,label='{0}({1:.1f})'.format(label,trend_per_year*1.e3),linestyle='--',linewidth=1)           
    else:
        ax.plot(df.year,df[varname],alpha=1,color=color,label='{0}'.format(label),linewidth=1.5)
    #----plot trend
    if trend_check:    
        ax.plot(df.year,ytrend,alpha=0.8,color=color,linestyle='--',linewidth=0.9)
        
#-------------------------------------------------------------#
def make_fig1_MonteCarlo(sic_listm,sic_listg,sic_list,
                         sst_gu_listm,sst_gu_listg,sst_gu_list,
                         sic_listh,sic_listc,
                         sst_gu_listh,sst_gu_listc,
                         sst_amv_listh,sst_amv_listc,
                         model_names_list_new,
                         le_name='HIST',
                         fig_name='scatters_HIST_and_CMIP6.pdf',
                         plot_naga=False,
                         plot_LE=False,
                         only_internal=False,
                         only_cmip=True,                 
                         le_members=None,
                         dfm_en=None,
                         dfg_en=None,
                         dfh_djf=None,
                         dfc_djf=None,
                         dfm_sst_djf_gu_en=None,
                         dfg_sst_djf_gu_en=None,
                         dfh_sst_djf_gu=None,
                         dfc_sst_djf_gu=None,
                         plot_cmip=False,
                         sic_listm_stats_MC=None,
                         sst_gu_listm_stats_MC=None,
                         plot_anom = True
                         ):
    #---------------------------------
    #----plot
    #plot_anom = True
    plot_main = True   # main fig.1
    plot_long = False #SI fig.**
    only_hist = False
    
    if plot_main:
        multimodel = False
    else:
        multimodel = True
        
    if multimodel:
        alpha = 0.5
        alpha = 1
    else:
        alpha = 0.5
        alpha = 1

    linewidth=1.5
    linewidth2=2    
    
    fontsize=8
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #spread_alpha = 0.05
    bc = 'None' # back ground color

    plot_end_year = 2017
    
    #----grid
    major_ticksx = np.arange(-10, 10, 0.1)
    minor_ticksx = np.arange(-10, 10, 0.05)
    major_ticksy3 = np.arange(-10,10, 2)
    minor_ticksy3 = np.arange(-10,10, 1)    
    
    yticklabels = [ "{0:.0f}%".format(x*100) for x in major_ticksy3]

    major_ticks_hist = np.arange(0, 30, 5)
    minor_ticks_hist = np.arange(0, 30, 1)
    
    #-----------------------------------------------------------------------------------#    
    #-----------------------------------------------------------------------------------#    
    #----make figure
    x1=5
    x2=6
    x2w=3
    
    fig = plt.figure(figsize=(7,5))
    
    #----make gridspec & axis
    gs_master = GridSpec(nrows=5, ncols=x2+x2w+2, height_ratios=[5,1,1,5,1],hspace=0,wspace=0)
    
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,0:x1]) #
    ax3 = fig.add_subplot(gs_1[:, :])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=5, subplot_spec=gs_master[0,x2:x2+x2w])
    ax2 = fig.add_subplot(gs_2[:, :])

    gs_2b = GridSpecFromSubplotSpec(nrows=1, ncols=5, subplot_spec=gs_master[1,x2:x2+x2w])
    ax2b = fig.add_subplot(gs_2b[:, :])    
    
    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,0:x1]) #
    ax1 = fig.add_subplot(gs_3[:, :])
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=5, subplot_spec=gs_master[3,x2:x2+x2w])
    ax4 = fig.add_subplot(gs_4[:, :])

    gs_4b = GridSpecFromSubplotSpec(nrows=1, ncols=5, subplot_spec=gs_master[4,x2:x2+x2w])
    ax4b = fig.add_subplot(gs_4b[:, :])    

    ax2b.axis('off')
    ax4b.axis('off')    
    #-----------------------------------------------------------------------------------#    
    # plot histogram part
    #-----------------------------------------------------------------------------------#    
    
    #----set grid
    ax2.set_xticks(major_ticksy3)
    ax2.set_xticks(minor_ticksy3, minor=True)
    ax2.set_yticks(major_ticksx)
    ax2.set_yticks(minor_ticksx, minor=True)    
    
    ax4.set_xticks(major_ticksy3)
    ax4.set_xticks(minor_ticksy3, minor=True)
    
    ax2.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax2.grid(color='gray',linestyle='--',which='minor', alpha=0.2)
    ax4.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax4.grid(color='gray',linestyle='--',which='minor', alpha=0.2)

    ax4.set_ylim([-8,1.5])
    ax4.set_xlim([0.5,4.5])    
    ax2.set_ylim([0.,0.4])
    ax2.set_xlim([0.5,4.5])    
    
    #----plot histograms
    alpha = 0.6

    label1m=le_name+' ($\sigma={0:.1f}$)'.format(np.std(np.array(sic_listm)*1.e3))
    label1g='NAGA ($\sigma={0:.1f}$)'.format(np.std(np.array(sic_listg)*1.e3))
    label3='CMIP6 ($\sigma={0:.1f}$)'.format(np.std(np.array(sic_list)*1.e3))
    label5m=le_name+' ($\sigma={0:.3f}$)'.format(np.std(np.array(sst_gu_listm)*10.))
    label5g='NAGA ($\sigma={0:.3f}$)'.format(np.std(np.array(sst_gu_listg)*10.))
    label6='CMIP6 ($\sigma={0:.3f}$)'.format(np.std(np.array(sst_gu_list)*10.))        
    
    #----boxplot of sic
    boxwidth=0.5
    sic_listm_stats_MC[0]['med']=sic_listm_stats_MC[0]['mean'] # medianをmeanに変更
    ax4.bxp( sic_listm_stats_MC, vert=True, positions=[2],showfliers=False,
             patch_artist=True,  # 細かい設定をできるようにする
             widths=boxwidth,  # boxの幅の設定
             boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                           edgecolor='black', linewidth=1),  # boxの枠線の設定
             #showmeans=True, # medianをmeanに変更
             medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
             whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
             capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
             flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
             #usermedians=[sic_listm_stats_MC[0]['mean']], # medianをmeanに変更
            )
    ax4.boxplot( np.array(sic_listg)*1.e3, vert=True, labels=['NAGA'],positions=[3],showfliers=False,
                 patch_artist=True,  # 細かい設定をできるようにする
                 widths=boxwidth,  # boxの幅の設定
                 boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                               edgecolor='black', linewidth=1),  # boxの枠線の設定
                 #showmeans=True, # medianをmeanに変更
                 medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                 whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                 capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                 usermedians=[np.mean(sic_listg)*1.e3], # medianをmeanに変更
                 whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
                )
    ax4.boxplot( np.array(sic_list)*1.e3, vert=True, labels=['CMIP6'],positions=[4],showfliers=False,
                 patch_artist=True,  # 細かい設定をできるようにする
                 widths=boxwidth,  # boxの幅の設定
                 boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                               edgecolor='black', linewidth=1),  # boxの枠線の設定
                 #showmeans=True, # medianをmeanに変更
                 medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                 whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                 capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                 usermedians=[np.mean(sic_list)*1.e3], # medianをmeanに変更
                 whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
                )
    
    xs1,xs2,xs3 = [],[],[]
    xs1.append(np.random.normal(2, 0.04, len(sic_listm)))
    xs2.append(np.random.normal(3, 0.04, len(sic_listg)))
    xs3.append(np.random.normal(4, 0.04, len(sic_list)))
    ax4.scatter(xs1,np.array(sic_listm)*1.e3,edgecolor='cornflowerblue',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)
    ax4.scatter(xs2,np.array(sic_listg)*1.e3,edgecolor='salmon',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)
    ax4.scatter(xs3,np.array(sic_list)*1.e3,edgecolor='green',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)        
    ax4.scatter([1],[-4.9],color='black',marker=',',s=20,alpha=1)
    ax4.scatter([1],[-5.3],color='gray',marker='d',s=20,alpha=1)
    ax4.annotate("HadISST2", xy = (0.6, -4.3), size = 5, color = "black")
    ax4.annotate("COBE-SST2", xy = (0.6, -6.1), size = 5, color = "dimgrey")    
    
    #----histogram of gulf sst    
    sst_gu_listm_stats_MC[0]['med']=sst_gu_listm_stats_MC[0]['mean'] # medianをmeanに変更    
    ax2.bxp( sst_gu_listm_stats_MC, vert=True, positions=[2],showfliers=False,
             patch_artist=True,  # 細かい設定をできるようにする
             widths=boxwidth,  # boxの幅の設定
             boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                           edgecolor='black', linewidth=1),  # boxの枠線の設定
             #showmeans=True, # medianをmeanに変更
             medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
             whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
             capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
             flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
             #usermedians=[sst_gu_listm_stats_MC[0]['mean']], # medianをmeanに変更
            )
    ax2.boxplot( np.array(sst_gu_listg)*10, vert=True, labels=['NAGA'],positions=[3],showfliers=False,
                 patch_artist=True,  # 細かい設定をできるようにする
                 widths=boxwidth,  # boxの幅の設定
                 boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                               edgecolor='black', linewidth=1),  # boxの枠線の設定
                 #showmeans=True, # medianをmeanに変更
                 medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                 whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                 capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                 usermedians=[np.mean(sst_gu_listg)*10], # medianをmeanに変更
                 whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
                )
    ax2.boxplot( np.array(sst_gu_list)*10, vert=True, labels=['CMIP6'],positions=[4],showfliers=False,
                 patch_artist=True,  # 細かい設定をできるようにする
                 widths=boxwidth,  # boxの幅の設定
                 boxprops=dict(facecolor='None',alpha=1,  # boxの塗りつぶし色の設定
                               edgecolor='black', linewidth=1),  # boxの枠線の設定
                 #showmeans=True, # medianをmeanに変更
                 medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                 whiskerprops=dict(color='black', linewidth=1),  # ヒゲの線の設定
                 capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),  # 外れ値の設定
                 usermedians=[np.mean(sst_gu_list)*10], # medianをmeanに変更
                 whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定
                )
    ax2.scatter(xs1,np.array(sst_gu_listm)*10,edgecolor='cornflowerblue',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)
    ax2.scatter(xs2,np.array(sst_gu_listg)*10,edgecolor='salmon',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)
    ax2.scatter(xs3,np.array(sst_gu_list)*10,edgecolor='green',facecolor='w',s=8,alpha=1,marker='o',linewidths=0.5)        
    
    ax2.scatter(1,0.24,color='black',marker=',',s=20,alpha=1)
    ax2.scatter(1,0.21,color='gray',marker='d',s=20,alpha=1)
    ax2.annotate("HadISST2", xy = (0.6, 0.27), size = 5, color = "black")
    ax2.annotate("COBE-SST2", xy = (0.6, 0.17), size = 5, color = "dimgrey")    
    
    ax4.set_xticks([1,2,3,4])
    ax4.set_xticklabels(["Obs.","HIST","NAGA","CMIP6"], rotation=0, fontsize=8)    
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["Obs.","HIST","NAGA","CMIP6"], rotation=0, fontsize=8)
        
    linestyle='-'
    linewidth2=0.5
    linewidth=0.5
    linealpha=1

    fontsize=8
    
    #---title
    ax2.set_title('(c) Gulf Stream DJF SST trend',fontsize=fontsize, y=1,x=0.0,loc='left')
    ax4.set_title('(d) Barents-Kara DJF SIC trend',fontsize=fontsize, y=1,x=0.0,loc='left')

    #----set axis label
    ax2.set_ylabel('SST trend [$^{\circ}$C decade$^{-1}$]',fontsize=fontsize)
    ax4.set_ylabel('SIC trend [% decade$^{-1}$]',fontsize=fontsize)    

    #-----------------------------------------------------------------#
    # plot  time series
    #-----------------------------------------------------------------#
    alpha = 0.5
    linewidth=1.5
    linewidth2=2    
    spread_alpha = 0.3
    bc = 'None' # back ground color
    plot_end_year = 2017
    
    #----grid
    major_ticksx = np.arange(1900, 2020, 10)
    minor_ticksx = np.arange(1900, 2020, 1)
    
    major_ticksy1 = np.arange(-1.,1.2, 0.1)
    minor_ticksy1 = np.arange(-1.,1.1, 0.05)
    yticklabels = [ "{0:.0f}%".format(x*100) for x in major_ticksy1]
    
    major_ticksy2 = np.arange(-4, 4, 1)
    minor_ticksy2 = np.arange(-4, 4, 0.2)
    
    major_ticksy3 = np.arange(-20,20, 0.5)
    minor_ticksy3 = np.arange(-20,20, 0.1)

    #-----------------------------------------------------------------------------------#    
    #----background color
    ax1.set_facecolor(bc)
    ax3.set_facecolor(bc)
    
    #----set grid
    ax1.set_xticks(major_ticksx)
    ax1.set_xticks(minor_ticksx, minor=True)
    ax1.set_yticks(major_ticksy1)
    ax1.set_yticks(minor_ticksy1, minor=True)
    ax1.grid(color='gray',linestyle='--',which='major', alpha=0.5)
    ax1.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax1.set_yticklabels(yticklabels)
    
    ax3.set_xticks(major_ticksx)
    ax3.set_xticks(minor_ticksx, minor=True)
    ax3.set_yticks(major_ticksy3)
    ax3.set_yticks(minor_ticksy3, minor=True)
    ax3.grid(color='gray',linestyle='--',which='major', alpha=0.5)
    ax3.grid(color='gray',linestyle='--',which='minor', alpha=0.)

    #----set x,ylim
    if plot_long:
        ax1.set_xlim([1900,plot_end_year])
        ax3.set_xlim([1900,plot_end_year])
    else:
        ax1.set_xlim([1970,plot_end_year])
        ax3.set_xlim([1970,plot_end_year])
    
    if plot_main:
        if plot_anom:
            ax1.set_ylim([-0.25,0.25])
            ax3.set_ylim([-1,1])
        else:
            ax1.set_ylim([0.25,0.85])
            ax3.set_ylim([14,16])
    else:
        if plot_anom:        
            ax1.set_ylim([-0.3,0.3])
            ax3.set_ylim([-1.5,1.5])
        else:
            ax1.set_ylim([0.25,0.9])
            ax3.set_ylim([13.5,16.5])
            
    #----set title
    fontsize=8    
    ax1.set_title('(b) Barents-Kara DJF SIC anomalies',fontsize=fontsize, y=1.,x=0.0,loc='left')
    ax3.set_title('(a) Gulf Stream DJF SST anomalies',fontsize=fontsize, y=1.,x=0.0,loc='left')        
    
    #----set axis label    
    ax1.set_ylabel('SIC anomalies [%]',fontsize=fontsize)
    ax3.set_ylabel('SST anomalies [$^{\circ}$C]',fontsize=fontsize)
    ax1.set_xlabel('year',fontsize=fontsize)
    ax3.set_xlabel('year',fontsize=fontsize)    
    
    #----------------------------------------#
    #----plot ax1 (DJF SIC Barents-Kara)
    plot_series_model_en(dfm_en,'sic','blue','HIST',ax1,alpha=0.5,trend_check=True,spread_alpha=spread_alpha,anom=plot_anom,facecolor='cornflowerblue')
    if only_hist:
        pass
    else:
        plot_series_model_en(dfg_en,'sic','red','NAGA',ax1,alpha=0.5,trend_check=True,spread_alpha=spread_alpha,anom=plot_anom,facecolor='salmon')
        
    plot_series_obs(dfh_djf,'sic','black','HadISST2',ax1,trend_check=True,anom=plot_anom)
    plot_series_obs(dfc_djf,'sic','gray','COBE-SST2',ax1,trend_check=True,anom=plot_anom)    
    
    ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', borderaxespad=0, fontsize=5) # legend    
    #----------------------------------------#    
    #----plot ax3 (DJF SST Gulf)
    plot_series_model_en(dfm_sst_djf_gu_en,'sic','blue','HIST',ax3,alpha=1,trend_check=True,spread_alpha=spread_alpha,anom=plot_anom,trend_SST=True,facecolor='cornflowerblue')
    if only_hist:
        pass
    else:    
        plot_series_model_en(dfg_sst_djf_gu_en,'sic','red','NAGA',ax3,alpha=1,trend_check=True,spread_alpha=spread_alpha,anom=plot_anom,trend_SST=True,facecolor='salmon')
        
    plot_series_obs(dfh_sst_djf_gu,'sst','black','HadISST2',ax3,trend_check=True,anom=plot_anom,trend_SST=True)
    plot_series_obs(dfc_sst_djf_gu,'sst','gray','COBE-SST2',ax3,trend_check=True,anom=plot_anom,trend_SST=True)
    ax3.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0, fontsize=5) # legend
    
    fig.tight_layout()
    ax1.xaxis.set_label_coords(0.5, -0.18)
    ax1.yaxis.set_label_coords(-0.2, 0.5)
    ax3.xaxis.set_label_coords(0.5, -0.18)
    ax3.yaxis.set_label_coords(-0.2, 0.5)

    ax2.xaxis.set_label_coords(0.5, -0.18)
    ax2.yaxis.set_label_coords(-0.2, 0.5)
    ax4.xaxis.set_label_coords(0.5, -0.18)
    ax4.yaxis.set_label_coords(-0.2, 0.5)    

    #plt.show()
    #sys.exit()
    plt.savefig('fig1.pdf', bbox_inches="tight", pad_inches=0.1)
    subprocess.call('open fig1.pdf',shell=True)        
    plt.close()

#----main
if __name__ == '__main__':
    rng = np.random.default_rng()
    
    #-------------------------------------------------#
    cutyear = 1970
    endyear = 2017
    csvpath = "./data_fig1/"    

    #-------------------------------------------------#
    # Barents-Kara SIC
    #-------------------------------------------------#
    #----HadISST 
    dfh_djf = pd.read_csv(csvpath+'sic_BK_had_djf.csv')
    #----COBE-SST2
    dfc_djf = pd.read_csv(csvpath+'sic_BK_cob_djf.csv')
    #----HIST
    dfm_en_Monte = pd.read_csv(csvpath+'sic_BK_miroc6_djf_enmean.csv')    
    #----NAGA
    dfg_en = pd.read_csv(csvpath+'sic_BK_pace_djf_enmean.csv')        
    #----------------------------------------------------#
    # Gulf Stream SST
    #----------------------------------------------------#
    #---HIST
    dfm_sst_djf_gu_en_Monte = pd.read_csv(csvpath+'sst_gu_miroc6_djf_enmean.csv')    
    #----NAGA
    dfg_sst_djf_gu_en = pd.read_csv(csvpath+'sst_gu_pace_djf_enmean.csv')        
    #----HadISST
    dfh_sst_djf_gu = pd.read_csv(csvpath+'sst_gu_had_djf.csv')            
    #----cobesst
    dfc_sst_djf_gu = pd.read_csv(csvpath+'sst_gu_cob_djf.csv')
    
    ###-----------------------------------------------------###
    # trends
    dfh_sic_trend = pd.read_csv(csvpath+'sic_BK_had_djf_trend.csv')
    dfc_sic_trend = pd.read_csv(csvpath+'sic_BK_cob_djf_trend.csv')
    dfm_sic_trend = pd.read_csv(csvpath+'sic_BK_miroc6_djf_trend.csv')
    dfg_sic_trend = pd.read_csv(csvpath+'sic_BK_pace_djf_trend.csv')
    df_sic_trend = pd.read_csv(csvpath+'sic_BK_cmip6_djf_trend.csv')    

    dfh_sst_gu_trend = pd.read_csv(csvpath+'sst_gu_had_djf_trend.csv')
    dfc_sst_gu_trend = pd.read_csv(csvpath+'sst_gu_cob_djf_trend.csv')
    dfm_sst_gu_trend = pd.read_csv(csvpath+'sst_gu_miroc6_djf_trend.csv')
    dfg_sst_gu_trend = pd.read_csv(csvpath+'sst_gu_pace_djf_trend.csv')
    df_sst_gu_trend = pd.read_csv(csvpath+'sst_gu_cmip6_djf_trend.csv')            
    
    ###-----------------------------------------------------###    
    sic_listm = []
    sic_listg = []
    sic_listh = []
    sic_listc = []

    sst_gu_listm = []
    sst_gu_listg = []
    sst_gu_listh = []
    sst_gu_listc = []
    
    #----------------------------------#
    for run in np.arange(1,51):
        sic_listm.append( dfm_sic_trend['sic_run{0:02}'.format(run)].values[0]    )
        sst_gu_listm.append( dfm_sst_gu_trend['sst_run{0:02}'.format(run)].values[0]    )
    for run in np.arange(1,11):
        sic_listg.append( dfg_sic_trend['sic_run{0:02}'.format(run)].values[0]    )
        sst_gu_listg.append( dfg_sst_gu_trend['sst_run{0:02}'.format(run)].values[0]    )
        
    sst_gu_listh.append( dfh_sst_gu_trend['sst'].values[0] )
    sst_gu_listc.append( dfc_sst_gu_trend['sst'].values[0] )    
    sic_listh.append( dfh_sic_trend['sic'.format(run)].values[0]    )
    sic_listc.append( dfc_sic_trend['sic'.format(run)].values[0]    )    
    
    #---cmip6 trend list
    sic_list    = [ df_sic_trend[x].values[0]  for x in df_sic_trend.columns ]
    sst_gu_list = [ df_sst_gu_trend[x].values[0]  for x in df_sst_gu_trend.columns ]
    
    #--------------------------------------------------------------#    
    # ax.boxplot(sic_listm) は
    # stats = cbook.boxplot_stats(sic_listm)とax.bxp(stats)と等しい
    # つまりstatsを1000回繰り返しコンポジットを作れば良い
    #--------------------------------------------------------------#
    #----sic_listm,sst_gu_listm, sst_amv_listmからランダムに10個を選ぶのを1000回行いアウトプットに保存
    monte_num=1000
    #monte_num=10 # test
    sic_listm_MonteCarlo     = ten_trend_MonteCarlo(np.array(sic_listm)*1.e3,monte_num,rng)
    sst_gu_listm_MonteCarlo  = ten_trend_MonteCarlo(np.array(sst_gu_listm)*10,monte_num,rng)    
    sic_mean, sic_std       = calc_mean_trend_std(sic_listm_MonteCarlo)
    sst_gu_mean, sst_gu_std = calc_mean_trend_std(sst_gu_listm_MonteCarlo)    
    
    #--------------------------------------------------------------#
    # compute the boxplot stats
    tmp_stats = cbook.boxplot_stats(sic_listm_MonteCarlo,bootstrap=10000,whis=[5,95])
    sic_listm_stats_MC = calc_mean_stats(tmp_stats,loopnum=monte_num)
    sic_listm_stats_MC[0]['label'] = 'HIST'
    
    tmp_stats = cbook.boxplot_stats(sst_gu_listm_MonteCarlo,bootstrap=10000,whis=[5,95])
    sst_gu_listm_stats_MC = calc_mean_stats(tmp_stats,loopnum=monte_num)
    sst_gu_listm_stats_MC[0]['label'] = 'HIST'    
    
    #sys.exit()
    #--------------------------------------------------------------#
    #--------------------------------------------------------------#        
    #figfun.make_fig1(sic_listm,sic_listg,[],
    #sys.exit()
    make_fig1_MonteCarlo(sic_listm,sic_listg,sic_list,        
                         sst_gu_listm,sst_gu_listg,sst_gu_list,
                         sic_listh,sic_listc,
                         sst_gu_listh,sst_gu_listc,
                         [],[],
                         [],
                         fig_name='scatters_allCMIP6.pdf',
                         le_name="HIST",plot_naga=True,
                         only_cmip=False,
                         dfm_en=dfm_en_Monte,
                         dfg_en=dfg_en,
                         dfh_djf=dfh_djf,
                         dfc_djf=dfc_djf,
                         dfm_sst_djf_gu_en=dfm_sst_djf_gu_en_Monte,
                         dfg_sst_djf_gu_en=dfg_sst_djf_gu_en,
                         dfh_sst_djf_gu=dfh_sst_djf_gu,
                         dfc_sst_djf_gu=dfc_sst_djf_gu,
                         sic_listm_stats_MC=sic_listm_stats_MC,
                         sst_gu_listm_stats_MC=sst_gu_listm_stats_MC,
                         plot_anom=True
                         )
    
