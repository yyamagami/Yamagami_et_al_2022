# -*- coding: utf-8 -*-
#
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

# ---------------------------------------------------- #
if __name__ == '__main__':
    
    #----------------------------------------------------#    
    #----model names    
    le_models = ['ACCESS-ESM1-5','CESM2','CanESM5',
                 'CNRM-CM6-1','EC-Earth3',
                 'GISS-E2-1-H',
                 'INM-CM5-0','IPSL-CM6A-LR','MIROC-ES2L',
                 'MPI-ESM1-2-HR','MPI-ESM1-2-LR', 'NorCPM1',
                 'UKESM1-0-LL']
    
    le_members = {'ACCESS-ESM1-5':30,'CESM2':11,'CanESM5':65,
                  'CNRM-CM6-1':21,
                  'EC-Earth3':25,
                  'GISS-E2-1-H':25,
                  'INM-CM5-0':10,'IPSL-CM6A-LR':32,'MIROC-ES2L':11,
                  'MPI-ESM1-2-HR':10,'MPI-ESM1-2-LR':10, 'NorCPM1':30,
                  'UKESM1-0-LL':17}
    
    #----------------------------------------------------#    
    #----read trend data
    #----------------------------------------------------#
    #----CMIP6 LE models
    df_trend_le = pd.read_csv('csv_trends/CMIP6_LE_trends.csv')
    
    #----CMIP6 run01 models
    df_trend_r1 = pd.read_csv('csv_trends/CMIP6_run01_trends.csv')
    
    #----HIST internal
    df_trend_hist = pd.read_csv('csv_trends/HIST_trends.csv')
    
    #----NAGA internal
    df_trend_naga = pd.read_csv('csv_trends/NAGA_trends.csv')
    
    #----COBE-SST2
    df_trend_cobe = pd.read_csv('csv_trends/COBE-SST2_trends.csv')

    #----HadISST2
    df_trend_hadi = pd.read_csv('csv_trends/HadISST2_trends.csv')    
    
    #----------------------------------------------------#    
    #----list
    sic_list     = list(df_trend_le.sic.values)
    sst_gu_list  = list(df_trend_le.sst.values)
    model_le     = list(df_trend_le.model.values)
    
    sic_listr1    = list(df_trend_r1.sic.values)
    sst_gu_listr1 = list(df_trend_r1.sst.values)
    model_r1      = list(df_trend_r1.model.values)
    
    sic_listm     = list(df_trend_hist.sic.values)
    sst_gu_listm  = list(df_trend_hist.sst.values)
    model_hist    = list(df_trend_hist.model.values)
    
    sic_listg     = list(df_trend_naga.sic.values)
    sst_gu_listg  = list(df_trend_naga.sst.values)
    model_naga    = list(df_trend_naga.model.values)

    sic_listc     = list(df_trend_cobe.sic.values)
    sst_gu_listc  = list(df_trend_cobe.sst.values)
    model_cobe    = list(df_trend_cobe.model.values)

    sic_listh     = list(df_trend_hadi.sic.values)
    sst_gu_listh  = list(df_trend_hadi.sst.values)
    model_hadi    = list(df_trend_hadi.model.values)    
        
    #---------------------------------
    #----plot
    fontsize=10
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #----grid
    major_ticksx = np.arange(-10, 10, 0.2)
    minor_ticksx = np.arange(-10, 10, 0.1)
    major_ticksy3 = np.arange(-10,10, 2)
    minor_ticksy3 = np.arange(-10,10, 1)    
    #
    yticklabels = [ "{0:.0f}%".format(x*100) for x in major_ticksy3]
    #
    major_ticks_hist = np.arange(0, 30, 5)
    minor_ticks_hist = np.arange(0, 30, 1)
    
    #-----------------------------------------------------------------------------------#    
    #-----------------------------------------------------------------------------------#    
    #----make figure
    fig = plt.figure(figsize=(10,8))
    
    #----make gridspec & axis
    gs_master = GridSpec(nrows=5, ncols=28, height_ratios=[1,10,2,1,10],hspace=0,wspace=0)

    # CMIP6 run01 all models
    gs_8 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1,10:11]) #
    ax8 = fig.add_subplot(gs_8[:, :])
    gs_9 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[1,0:10])
    ax9 = fig.add_subplot(gs_9[:, :])
    gs_10 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,0:10]) #
    ax10 = fig.add_subplot(gs_10[:, :])
    gs_11 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1,12:17]) #
    ax11 = fig.add_subplot(gs_11[:, :])
    gs_12 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1,17:22]) #
    ax12 = fig.add_subplot(gs_12[:, :])
    gs_13 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1,22:24]) #
    ax13 = fig.add_subplot(gs_13[:, :])                            
    
    # HIST, NAGA
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[4,10:11]) #
    ax1 = fig.add_subplot(gs_1[:, :])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,0:10])
    ax2 = fig.add_subplot(gs_2[:, :])
    gs_5 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,0:10]) #
    ax5 = fig.add_subplot(gs_5[:, :])

    # CMIP6 large ensemble    
    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[4,22:23]) #
    ax3 = fig.add_subplot(gs_3[:, :])
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,12:22]) #
    ax4 = fig.add_subplot(gs_4[:, :])
    gs_6 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[3,12:22]) #
    ax6 = fig.add_subplot(gs_6[:, :])
    
    gs_7 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[4,23:24]) #
    ax7 = fig.add_subplot(gs_7[:, :])                
    
    #-----------------------------------------------------------------------------------#    
    #-----------------------------------------------------------------------------------#    
    ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    ax3.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    ax5.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    ax6.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)

    ax8.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    ax10.tick_params(labelbottom=False,
                     labelleft=False,
                     labelright=False,
                     labeltop=False)
    #----set grid
    ax1.set_yticks(major_ticksy3)
    ax1.set_yticks(minor_ticksy3, minor=True)
    ax1.set_xticks(major_ticks_hist)
    ax1.set_xticks(minor_ticks_hist, minor=True)    
    ax1.set_xticklabels([0,5,10])
    
    ax2.set_xticks(major_ticksx)
    ax2.set_xticks(minor_ticksx, minor=True)
    ax2.set_yticks(major_ticksy3)
    ax2.set_yticks(minor_ticksy3, minor=True)

    ax3.set_yticks(major_ticksy3)
    ax3.set_yticks(minor_ticksy3, minor=True)    
    ax3.set_xticks(major_ticks_hist)
    ax3.set_xticks(minor_ticks_hist, minor=True)
    ax3.set_xticklabels([0,5,10])    
    
    ax4.set_xticks(major_ticksx)
    ax4.set_xticks(minor_ticksx, minor=True)
    ax4.set_yticks(major_ticksy3)
    ax4.set_yticks(minor_ticksy3, minor=True)

    ax5.set_xticks(major_ticksx)
    ax5.set_xticks(minor_ticksx, minor=True)
    ax5.set_yticks(major_ticks_hist)
    ax5.set_yticks(minor_ticks_hist, minor=True)
    ax5.set_yticklabels([0,5,10,15])        
    
    ax6.set_xticks(major_ticksx)
    ax6.set_xticks(minor_ticksx, minor=True)
    ax6.set_yticks(major_ticks_hist)
    ax6.set_yticks(minor_ticks_hist, minor=True)
    ax6.set_yticklabels([0,5,10,15])            

    ax8.set_xticks(major_ticksx)
    ax8.set_xticks(minor_ticksx, minor=True)
    ax8.set_yticks(major_ticks_hist)
    ax8.set_yticks(minor_ticks_hist, minor=True)
    ax8.set_yticklabels([0,5,10,15])        
    
    ax9.set_xticks(major_ticksx)
    ax9.set_xticks(minor_ticksx, minor=True)
    ax9.set_yticks(major_ticksy3)
    ax9.set_yticks(minor_ticksy3, minor=True)

    ax10.set_yticks(major_ticksx)
    ax10.set_yticks(minor_ticksx, minor=True)
    ax10.set_xticks(major_ticks_hist)
    ax10.set_xticks(minor_ticks_hist, minor=True)
    ax10.set_xticklabels([0,5,10,15])            
    
    ax1.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax1.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax2.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax2.grid(color='gray',linestyle='--',which='minor', alpha=0.2)
    ax3.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax3.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax4.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax4.grid(color='gray',linestyle='--',which='minor', alpha=0.2)
    ax5.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax5.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax6.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax6.grid(color='gray',linestyle='--',which='minor', alpha=0.)

    ax8.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax8.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax9.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax9.grid(color='gray',linestyle='--',which='minor', alpha=0.2)
    ax10.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax10.grid(color='gray',linestyle='--',which='minor', alpha=0.)
    ax11.grid(color='gray',linestyle='--',which='major', alpha=0.2)
    ax11.grid(color='gray',linestyle='--',which='minor', alpha=0.)            

    ax1.axis("off")
    ax3.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    ax10.axis("off")
    ax11.axis("off")
    ax12.axis("off")    
    ax13.axis("off")    
    
    ax1.set_ylim([-8,2])
    ax3.set_ylim([-8,2])
    ax8.set_ylim([-8,2])
    
    ax5.set_xlim([-0.0,0.45])
    ax6.set_xlim([-0.0,0.45])
    ax10.set_xlim([-0.0,0.45])
    
    ax1.set_xlim([0.5,2.5])
    ax3.set_xlim([0.5,2.5])
    ax8.set_xlim([0.5,2.5])            
    
    ax5.set_ylim([0.5,2.5])        
    ax6.set_ylim([0.5,2.5])
    ax10.set_ylim([0.5,2.5])    
    
    #----plot histograms
    alpha = 0.5

    #------boxplot
    # boxplot settings
    boxwidth=0.5
    ax1.boxplot( [np.array(sic_listm)*1.e3], labels=["HIST"], vert=True, positions=[1],showfliers=False,
                 patch_artist=True, 
                 widths=boxwidth,  
                 boxprops=dict(facecolor='blue',alpha=0.75, 
                               color='black', linewidth=1), 
                 medianprops=dict(color='black', linewidth=1), 
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),
                 usermedians=[np.mean(sic_listm)*1.e3], 
                 whis=[5,95] 
                )
    
    ax1.boxplot( [np.array(sic_listg)*1.e3], labels=["NAGA"], vert=True, positions=[2],showfliers=False,
                 patch_artist=True,
                 widths=boxwidth, 
                 boxprops=dict(facecolor='red',alpha=0.75, 
                               color='black', linewidth=1),
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1), 
                 capprops=dict(color='black', linewidth=1), 
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),
                 usermedians=[np.mean(sic_listg)*1.e3], 
                 whis=[5,95] 
                )

    ax3.boxplot( [np.array(sic_list)*1.e3], labels=["cmip6"], vert=True, positions=[1],showfliers=False,
                 patch_artist=True, 
                 widths=boxwidth,  
                 boxprops=dict(facecolor='green',alpha=0.75,  
                               color='black', linewidth=1),  
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),
                 usermedians=[np.mean(sic_list)*1.e3], 
                 whis=[5,95]  
                )

    ax8.boxplot( [np.array(sic_listr1)*1.e3], labels=["run01"], vert=True, positions=[1],showfliers=False,
                 patch_artist=True, 
                 widths=boxwidth,  
                 boxprops=dict(facecolor='green',alpha=0.75, 
                               color='black', linewidth=1), 
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),
                 usermedians=[np.mean(sic_listr1)*1.e3],
                 #usermedians=[-2], # medianをmeanに変更 #debug
                 whis=[5,95]  # ヒゲのパーセンタイルを5~95に指定                 
                )
    
    ax5.boxplot( [np.array(sst_gu_listm)*10], labels=["HIST"], vert=False, positions=[1],showfliers=False,
                 patch_artist=True,  
                 widths=boxwidth,  
                 boxprops=dict(facecolor='blue',alpha=0.75,
                               color='black', linewidth=1),
                 medianprops=dict(color='black', linewidth=1), 
                 whiskerprops=dict(color='black', linewidth=1), 
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1), 
                 usermedians=[np.mean(sst_gu_listm)*10],
                 whis=[5,95] 
                )
    
    ax5.boxplot( [np.array(sst_gu_listg)*10], labels=["NAGA"], vert=False, positions=[2],showfliers=False,
                 patch_artist=True,
                 widths=boxwidth, 
                 boxprops=dict(facecolor='red',alpha=0.75,  
                               color='black', linewidth=1), 
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1), 
                 usermedians=[np.mean(sst_gu_listg)*10], 
                 whis=[5,95] 
                )
    
    ax6.boxplot( [np.array(sst_gu_list)*10], labels=["cmip6"], vert=False, positions=[1],showfliers=False,
                 patch_artist=True,  
                 widths=boxwidth,  
                 boxprops=dict(facecolor='green',alpha=0.75,  
                               color='black', linewidth=1),  
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),
                 usermedians=[np.mean(sst_gu_list)*10], 
                 whis=[5,95]
                )

    ax10.boxplot( [np.array(sst_gu_listr1)*10], labels=["run01"], vert=False, positions=[1],showfliers=False,
                 patch_artist=True, 
                 widths=boxwidth,  
                 boxprops=dict(facecolor='green',alpha=0.75,  
                               color='black', linewidth=1),  
                 medianprops=dict(color='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),  
                 flierprops=dict(markeredgecolor='black', markeredgewidth=1),  
                 usermedians=[np.mean(sst_gu_listr1)*10],
                 whis=[5,95] 
                )
        
    alpha = 1
    
    linestyle='-'
    linewidth2=0.5
    linewidth=0.5
    linealpha=0.5
    
    ax4.hlines(np.mean(sic_list)*1.e3,-100,100,colors='green',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax4.vlines(np.mean(sst_gu_list)*10,-100,100,colors='green',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax4.hlines(sic_listh[0]*1.e3,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax4.vlines(sst_gu_listh[0]*10,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax4.hlines(sic_listc[0]*1.e3,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax4.vlines(sst_gu_listc[0]*10,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    
    ax2.hlines(np.mean(sic_listm)*1.e3,-100,100,colors='blue',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax2.hlines(np.mean(sic_listg)*1.e3,-100,100,colors='red',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax2.vlines(np.mean(sst_gu_listm)*10,-100,100,colors='blue',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax2.vlines(np.mean(sst_gu_listg)*10,-100,100,colors='red',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax2.hlines(sic_listh[0]*1.e3,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax2.vlines(sst_gu_listh[0]*10,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax2.hlines(sic_listc[0]*1.e3,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax2.vlines(sst_gu_listc[0]*10,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)

    ax9.hlines(np.mean(sic_listr1)*1.e3,-100,100,colors='green',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax9.vlines(np.mean(sst_gu_listr1)*10,-100,100,colors='green',linestyle=linestyle,linewidth=linewidth2,alpha=linealpha)
    ax9.hlines(sic_listh[0]*1.e3,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax9.vlines(sst_gu_listh[0]*10,-100,100,colors='black',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax9.hlines(sic_listc[0]*1.e3,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    ax9.vlines(sst_gu_listc[0]*10,-100,100,colors='gray',linestyle=linestyle,linewidth=linewidth,alpha=linealpha)
    
    linewidth2=2
    linewidth=2    
    linestyle='-'
    
    #-----------------------------------------------------------------#
    # scatter plot
    #-----------------------------------------------------------------#
    #
    #----color & marker list
    model_names_list_new = copy.copy(model_le)
    tmp_len = len(model_names_list_new)
    #
    color_list = []
    loop_num = 1
    #
    for i in range(tmp_len):
        tmp_num = np.mod(i*loop_num/tmp_len, 1)
        #print(i,tmp_num,i*loop_num/tmp_len,tmp_len)
        color_list.append(cm.tab20(tmp_num))
    
    marker_list = []
    for i in  model_names_list_new :   
        marker_list.append(i[0])
    #sys.exit()
    #

    ax4.set_xlim([-0.0,0.45])    
    ax4.set_ylim([-8,2])
    ax2.set_xlim([-0.0,0.45])    
    ax2.set_ylim([-8,2])
    
    #----set title
    ax5.text(0,2.5,'(b) HIST, NAGA',size=fontsize,horizontalalignment="left")
    ax6.text(0,2.5,'(c) CMIP6 (ensemble means)',size=fontsize,horizontalalignment="left")
    
    #----set scale
    sic_listm = 1.e3 * np.array(sic_listm) # %/decade
    sic_list = 1.e3 * np.array(sic_list) # %/decade
    sic_listg = 1.e3 * np.array(sic_listg) # %/decade
    sic_listh = 1.e3 * np.array(sic_listh) # %/decade
    sic_listc = 1.e3 * np.array(sic_listc) # %/decade        

    sst_gu_listm = 10 * np.array(sst_gu_listm) # K/decade
    sst_gu_list = 10 * np.array(sst_gu_list) # K/decade
    sst_gu_listg = 10 * np.array(sst_gu_listg) # K/decade
    sst_gu_listh = 10 * np.array(sst_gu_listh) # K/decade
    sst_gu_listc = 10 * np.array(sst_gu_listc) # K/decade    
    
    #
    alpha = 1
    sic_hist_mean = np.mean(sic_listh)
    sst_hist_mean = np.mean(sst_gu_listh)
    
    #------------------------------------------------------------------------------------#
    # fitting line 

    x = np.concatenate([sst_gu_list,[sst_hist_mean]])
    y = np.concatenate([sic_list,[sic_hist_mean]])

    #print(x,sst_gu_list,sst_hist_mean)
    #print(y,sic_list,sic_hist_mean)    
    
    a,b = np.polyfit(x, y, 1)
    x2=np.arange(-1,1,0.01)
    y2 = a * x2 + b
    ax4.plot(x2, y2,color='gray',linewidth=0.4,linestyle='solid')
    
    tmp_r, tmp_p = pearsonr(x,y) # pearson correlation
    #print("----------------")
    #print("r=", tmp_r)
    #print("p=", tmp_p)
    
    a_str = str( Decimal(str(a)).quantize(Decimal("1"),rounding=ROUND_HALF_UP) )
    b_str = str( Decimal(str(b)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP) )
    tmp_text = 'SIC='+a_str+'$*$SST'+b_str
    #print(a,a_str)
    #print(b,b_str)
    print(tmp_text)
    ax4.text(0.98,0.94,tmp_text,size=10,horizontalalignment="right",transform=ax4.transAxes)
    ax4.text(0.98,0.88,'r={0:.2f}'.format(np.corrcoef(x,y)[0,1]),size=10,horizontalalignment="right",transform=ax4.transAxes,color='red') 
    #
    ax7.scatter([],[],color='black',label='HadISST2',marker=',',s=20)
    ax7.scatter([],[],color='gray',label='COBE-SST2',marker='d',s=20)
    ax7.scatter([],[],color='blue',label='HIST (50)',marker='o',s=20)
    ax7.scatter([],[],color='red',label='NAGA (10)',marker='^',s=20)
    ax7.scatter([],[],color='white',label='  ',alpha=0,s=20)
    ax7.scatter([],[],color='green',label='Multi-model mean',marker='*',s=25)
    #
    for i in range( len(model_names_list_new) ):
        color = color_list[i]
        marker = marker_list[i]
        label = "{0} ({1})".format(model_names_list_new[i],le_members[model_names_list_new[i]])
        ax4.scatter(sst_gu_list[i],sic_list[i],color=color,marker='${0}$'.format(marker),s=50,alpha=1)
        ax7.scatter([],[],color=color,label=label,marker='${0}$'.format(marker),s=20,alpha=1)        
    #
    ax4.scatter(np.mean(sst_gu_list),np.mean(sic_list),color='green',marker='*',s=80)
    ax4.scatter(sst_gu_listh,sic_listh,color='black',marker=',',s=80)
    ax4.scatter(sst_gu_listc,sic_listc,color='gray',marker='d',s=80)    
    #
    ax7.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=7)
    
    #------------------------------------------------------------------------------------#
    # fitting line 
    x = sst_gu_listm
    y = sic_listm
    a,b = np.polyfit(x, y, 1)
    x2=np.arange(-1,1,0.01)    
    y2 = a * x2 + b
    ax2.plot(x2, y2,color='gray',linewidth=0.4,linestyle='solid')
    
    tmp_r, tmp_p = pearsonr(x,y) # pearson correlation

    #print("r & p of HIST")
    #print("r=", tmp_r)
    #print("p=", tmp_p)
    
    a_str = str( Decimal(str(a)).quantize(Decimal("1"),rounding=ROUND_HALF_UP) )
    #b_str = str( Decimal(str(b)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP) )
    b_str = str( Decimal(str(b)).quantize(Decimal("0.1"),rounding=ROUND_HALF_UP) )
    tmp_text = 'SIC='+a_str+'$*$SST'+b_str
    #print(a,a_str)
    #print(b,b_str)
    print(tmp_text)
    #print("----------------")
    #print()
    
    ax2.text(0.98,0.94,tmp_text,size=10,horizontalalignment="right",transform=ax2.transAxes)
    ax2.text(0.98,0.88,'r={0:.2f}'.format(np.corrcoef(sst_gu_listm,sic_listm)[0,1]),size=10,horizontalalignment="right",transform=ax2.transAxes)
    #
    ax2.scatter(sst_gu_listm,sic_listm,color='blue',marker='o',s=10,alpha=0.5)
    ax2.scatter(sst_gu_listg,sic_listg,color='red',marker='^',s=10,alpha=0.5)
    #
    ax2.scatter(np.mean(sst_gu_listm),np.mean(sic_listm),color='blue',label='HIST',marker='o',s=80)
    ax4.scatter(np.mean(sst_gu_listm),np.mean(sic_listm),color='blue',label='HIST',marker='o',s=80)    
    ax2.scatter(np.mean(sst_gu_listg),np.mean(sic_listg),color='red',label='NAGA',marker='^',s=80)
    ax4.scatter(np.mean(sst_gu_listg),np.mean(sic_listg),color='red',label='NAGA',marker='^',s=80)
    ax2.scatter(sst_gu_listh,sic_listh,color='black',label='HadISST2',marker=',',s=80)
    ax2.scatter(sst_gu_listc,sic_listc,color='gray',label='COBE-SST2',marker='d',s=80)    

    #-----------------------------------------------------------------------------------------    
    # scatter for CMIP6 run01
    #
    #----color & marker list
    model_names_list_new = copy.copy(model_r1)
    tmp_len = len(model_names_list_new)
    #
    color_list = []
    loop_num = 1
    #
    for i in range(tmp_len):
        tmp_num = np.mod(i*loop_num/tmp_len, 1)
        color_list.append(cm.tab20(tmp_num))
    
    marker_list = []
    for i in  model_names_list_new :   
        marker_list.append(i[0])
    #
    ax9.set_xlim([-0.0,0.45])    
    ax9.set_ylim([-8,2])
    
    #----set title
    ax10.text(0,2.5,'(a) CMIP6 (one member)',size=fontsize,horizontalalignment="left")
    
    #----set scale
    sic_listr1 = 1.e3 * np.array(sic_listr1) # %/decade
    sst_gu_listr1 = 10 * np.array(sst_gu_listr1) # K/decade
    alpha = 1
    
    #---------
    # fitting line & scatters 
    x = sst_gu_listr1
    y = sic_listr1
    #print(x,sst_gu_listr1)
    #print(y,sic_listr1)    
    
    a,b = np.polyfit(x, y, 1)
    x2=np.arange(-1,1,0.01)
    y2 = a * x2 + b
    ax9.plot(x2, y2,color='gray',linewidth=0.4,linestyle='solid')
    
    tmp_r, tmp_p = pearsonr(x,y) # pearson correlation
    #print("----------------")
    #print("r=", tmp_r)
    #print("p=", tmp_p)
    
    a_str = str( Decimal(str(a)).quantize(Decimal("1"),rounding=ROUND_HALF_UP) )
    b_str = str( Decimal(str(b)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP) )
    tmp_text = 'SIC='+a_str+'$*$SST'+b_str
    #print(a,a_str)
    #print(b,b_str)
    print(tmp_text)
    ax9.text(0.98,0.94,tmp_text,size=10,horizontalalignment="right",transform=ax9.transAxes)
    ax9.text(0.98,0.88,'r={0:.2f}'.format(np.corrcoef(x,y)[0,1]),size=10,horizontalalignment="right",transform=ax9.transAxes,color='red') 
    #
    ax11.scatter([],[],color='black',label='HadISST2',marker=',',s=20)
    ax11.scatter([],[],color='gray',label='COBE-SST2',marker='d',s=20)
    ax11.scatter([],[],color='blue',label='HIST (50)',marker='o',s=20)
    ax11.scatter([],[],color='red',label='NAGA (10)',marker='^',s=20)
    ax11.scatter([],[],color='white',label='  ',alpha=0,s=20)
    ax11.scatter([],[],color='green',label='Multi-model mean',marker='*',s=25)
    #
    legend_counter = 0
    for i in range( len(model_names_list_new) ):
        color = color_list[i]
        marker = marker_list[i]
        label = model_names_list_new[i]
        ax9.scatter(sst_gu_listr1[i],sic_listr1[i],color=color,marker='${0}$'.format(marker),s=50,alpha=1)
        #
        if label=='CanESM5' or label=='MPI-ESM1-2-HR':
            legend_counter += 1
        #
        if legend_counter==0:
            ax11.scatter([],[],color=color,label=label,marker='${0}$'.format(marker),s=20,alpha=1)
        elif legend_counter==1:
            ax12.scatter([],[],color=color,label=label,marker='${0}$'.format(marker),s=20,alpha=1)
        elif legend_counter==2:
            ax13.scatter([],[],color=color,label=label,marker='${0}$'.format(marker),s=20,alpha=1)                                    
    #
    ax9.scatter(np.mean(sst_gu_listr1),np.mean(sic_listr1),color='green',marker='*',s=80)
    ax9.scatter(sst_gu_listh,sic_listh,color='black',marker=',',s=80)
    ax9.scatter(sst_gu_listc,sic_listc,color='gray',marker='d',s=80)
    ax9.scatter(np.mean(sst_gu_listm),np.mean(sic_listm),color='blue',label='HIST',marker='o',s=80)
    ax9.scatter(np.mean(sst_gu_listg),np.mean(sic_listg),color='red',label='NAGA',marker='^',s=80)    
    #
    ax11.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=7)
    ax12.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=7)
    ax13.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=7)        
    
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #----set x&yaxis label
    fontsize=12
    fig.text(0.075,0.50,"DJF Barents-Kara SIC [$\%$ decade$^{-1}$]",
             rotation=90,fontsize=fontsize,
             horizontalalignment="center",verticalalignment="center")
    fig.text(0.50,0.05,"DJF Gulf Stream SST [$^{\circ}$C decade$^{-1}$]",
             fontsize=fontsize,
             horizontalalignment="center",verticalalignment="center")
    #
    #plt.show()
    #plt.close()
    #sys.exit()
    #
    plt.savefig("fig6.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    subprocess.call('open fig6.pdf',shell=True)    
    
