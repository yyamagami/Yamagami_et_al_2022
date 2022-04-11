# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import matplotlib as mpl
from matplotlib import colors
from matplotlib.cbook import contiguous_regions
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
import my_netcdf as netcdf # netcdf
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
import matplotlib.path as mpath

# ---------------------------------------------------- #
font_prop = mpl.font_manager.FontProperties()
undef = -999

#---------------------------------------------------- #
#----functions
def draw_lonlat_box(area,m,ax,color='k',lw=1,linestyle='-'):
    wlon = area[0]
    elon = area[1]
    slat = area[2]
    nlat = area[3]
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
    tmpdata = tmpdata[0,:,:,:]
    
    lon = np.concatenate([lon,lon[0:2]])
    tmpdata = np.concatenate([tmpdata,tmpdata[:,:,0:2]],axis=2)
    tmpdata = np.ma.array(tmpdata.data, fill_value=-999., mask=tmpdata==-999.) #mask

    #----mask nan
    nan_mask = np.isnan(tmpdata.data)
    mask_candidate = np.array([nan_mask,tmpdata.mask])
    new_mask = np.any(mask_candidate,axis=0) 
    #print(mask_candidate.shape,nan_mask.shape,tmpdata.mask.shape,new_mask.shape)        
    tmpdata = np.ma.array(tmpdata.data, fill_value=-999., mask=new_mask)
    
    return tmpdata,lon,lat,lev

def plot_BK_each(uin,vin,shadein,contourin,inlon,inlat,fig,ax,
                 plot_area=[-100,40,30,85],scale=1,veclen=1,fontsize=10,
                 print_veclen=False,
                 cmap = plt.cm.Reds,
                 sclevs=np.linspace(-1,1,11),
                 cclevs=np.linspace(-1,1,11),
                 plot_contour=True,
                 plot_atm=False,
                 ua=None,
                 va=None,
                 shadea=None,
                 contoura=None,
                 lona=None,
                 lata=None,
                 lonlat=False,
                 vecunit="m$s^{{-}1}$ decade$^{-1}$",
                 shadein_sig=None,
                 usig=None,
                 vsig=None):

    #---- per decade
    uino = uin*10.
    vino = vin*10.
    sino = shadein*10.    
    cino = contourin*10.
    if (usig is not None)&(vsig is not None):
        usig = usig*10.
        vsig = vsig*10.

    lonos, latos = np.meshgrid(inlon,inlat) # make lats and lons
    
    #----chose lat interval for vector plot
    uino.mask[:,:] = True
    print(uino.shape)
        
    step_list=[]
    for y in range(len(inlat)):
        if plot_atm:
            step =  int(np.round(inlat[y]/20))
        else:
            step =  int(np.round(inlat[y]/15))
        #
        if step==0:
            step = 1
        step=np.abs(step) #

        if plot_atm:
            step_index = np.arange(0,258,step)
        else:
            step_index = np.arange(0,362,step)
        #
        uino.mask[y,step_index] = False
    vino = np.ma.array(vino,fill_value=-999,mask=uino.mask)
    #
    if (usig is not None)&(vsig is not None):
        # uino.maskでFalseかつ,usig,vsigがともにFalseのグリッドのみFalse,あとはTrueにする
        tmpwhere =  usig.mask | vsig.mask | uino.mask 
        usig = np.ma.array(usig,fill_value=-999,mask=tmpwhere)
        vsig = np.ma.array(vsig,fill_value=-999,mask=tmpwhere)

        # uino.maskでFalseかつ,usigがTrueまたはvsigがTrueのグリッドのみFalse,あとはTrueにする        
        tmpwhere =  uino.mask & (~vsig.mask | ~usig.mask)
        uino = np.ma.array(uino,fill_value=-999,mask=tmpwhere)
        vino = np.ma.array(vino,fill_value=-999,mask=tmpwhere)        
        
    #----make basemap
    if lonlat:
        m = Basemap(llcrnrlon=15,llcrnrlat=60,urcrnrlon=70,urcrnrlat=85,projection='cyl',resolution='c',ax=ax)
    else:
        m = Basemap(resolution='l',projection='lcc', lat_0=75, lon_0=30,width=3600000, height=3000000,ax=ax)
        
    m.drawmapboundary(fill_color='white',color='black',linewidth=1.0)
    if not plot_atm:
        m.fillcontinents(color='gray',lake_color='gray')    
    else:
        m.drawcoastlines(linewidth=1.0)
    m.drawparallels(np.arange(-90.,99.,10.),labels=[True,False,False,False],fontsize=fontsize,linewidth=0.8)
    m.drawmeridians(np.arange(-180.,180.,30.),labels=[False,False,False,True],fontsize=fontsize,linewidth=0.8)
    mlons, mlats = m(lonos,latos)
    
    #----shade
    cs1 = m.contourf(mlons,mlats,sino,cmap=cmap,extend='both',levels=sclevs,alpha=1,latlon=lonlat)
    if not shadein_sig is None:
        sino_sig = shadein_sig*10.                
        mpl.rcParams['hatch.linewidth'] = 0.2   # hatch linewidth
        mpl.rcParams['hatch.color'] = 'black' # hatch color
        cs_hatch = m.contourf(mlons,mlats,sino_sig,colors='none',extend='both',
                              levels=sclevs,alpha=0,latlon=lonlat,hatches=['/////'])
    #----contour
    if plot_contour:
        cc1 = m.contour(mlons,mlats,cino,levels=cclevs,alpha=1,latlon=lonlat,colors='k',linewidths=1)
        plt.clabel(cc1, fontsize=8, fmt='%1.1f')
    #----vector
    vintlon = 1 #vector interval
    vintlat = 1 #vector interval
    #####
    if (usig is not None)&(vsig is not None):
        q1 = m.quiver(mlons[::vintlat,::vintlon],mlats[::vintlat,::vintlon],
                      uino[::vintlat,::vintlon],vino[::vintlat,::vintlon],
                      angles='xy',units='width',scale=scale,alpha=1,color='dimgray',
                      pivot='mid',headwidth=10,headlength=10,latlon=lonlat) # scale is just a scale for figsize and ua&va
        q2 = m.quiver(mlons[::vintlat,::vintlon],mlats[::vintlat,::vintlon],
                      usig[::vintlat,::vintlon],vsig[::vintlat,::vintlon],
                      angles='xy',units='width',scale=scale,alpha=1,color='k',
                      pivot='mid',headwidth=10,headlength=10,latlon=lonlat)# scale is just a scale for figsize and ua&va
    else:        
        q1 = m.quiver(mlons[::vintlat,::vintlon],mlats[::vintlat,::vintlon],
                      uino[::vintlat,::vintlon],vino[::vintlat,::vintlon],
                      angles='xy',units='width',scale=scale,alpha=1,color='black',
                      pivot='mid',headwidth=10,headlength=10,latlon=lonlat) # scale is just a scale for figsize and ua&va
    #####
    if print_veclen:
        qk = plt.quiverkey(q1, 0.84, 1.06, veclen, '{0}'.format(veclen)+vecunit, labelpos='E',fontproperties={'size':8}) # 1 means 1 size of vector in m/s??
            
    return m,cs1,q1

    
#----main
if __name__ == '__main__':
    
    zlev1 = 54
    zlev2 = 1.0
    
    #--------------------------------------#
    # read main data
    #--------------------------------------#
    #----NAGA
    uos,lons,lats,levs = get_data('data/trend/pace/uo_DJFtrend_pace.nc','uo')    
    vos,_,_,_          = get_data('data/trend/pace/vo_DJFtrend_pace.nc','vo')
    uoss,_,_,_         = get_data('data/trend/pace/uo_DJFtrend_pace_sig.nc','uo')    
    voss,_,_,_         = get_data('data/trend/pace/vo_DJFtrend_pace_sig.nc','vo')
    
    uvo_mask = (uoss.mask | voss.mask)
    uoss.data[uvo_mask]=0
    voss.data[uvo_mask]=0    
    uoss = np.ma.array(uoss.data, fill_value=0., mask=uvo_mask)
    voss = np.ma.array(voss.data, fill_value=0., mask=uvo_mask)
    
    tos,_,_,_          = get_data('data/trend/pace/to_trend_pace.nc','to')
    ssts_sig,_,_,_ = get_data('data/trend/pace/sst_DJFtrend_pace_sig.nc','sst')
    
    sics,_,_,_         = get_data('data/trend/pace/aig_trend_pace.nc','aig')
    sics *= 1.e2 # to %
    sics_sig,_,_,_ = get_data('data/trend/pace/aig_DJFtrend_pace_sig.nc','aig')
    sics_sig *= 1.e2 # to %
    
    uis,_,_,_          = get_data('data/trend/pace/ui_DJFtrend_pace.nc','ui')    
    vis,_,_,_          = get_data('data/trend/pace/vi_DJFtrend_pace.nc','vi')
    uiss,_,_,_          = get_data('data/trend/pace/ui_DJFtrend_pace_sig.nc','ui')    
    viss,_,_,_          = get_data('data/trend/pace/vi_DJFtrend_pace_sig.nc','vi')
    #uvi_mask = (uiss.data<-10) | (viss.data<-10)
    uvi_mask = (uiss.mask | viss.mask)
    uiss.data[uvi_mask]=0
    viss.data[uvi_mask]=0    
    uiss = np.ma.array(uiss.data, fill_value=0., mask=uvi_mask)
    viss = np.ma.array(viss.data, fill_value=0., mask=uvi_mask)
    
    u10s,lonas,latas,levas = get_data('data/trend/pace/u10_DJFtrend_pace.nc','u10')
    v10s,_,_,_             = get_data('data/trend/pace/v10_DJFtrend_pace.nc','v10')
    #
    u10ss,_,_,_ = get_data('data/trend/pace/u10_DJFtrend_pace_sig.nc','u10')
    v10ss,_,_,_ = get_data('data/trend/pace/v10_DJFtrend_pace_sig.nc','v10')
    #uv10_mask = (u10ss.data<-10)|(v10ss.data<-10)
    uv10_mask = (u10ss.mask | v10ss.mask)
    u10ss.data[uv10_mask]=0
    v10ss.data[uv10_mask]=0        
    u10ss = np.ma.array(u10ss.data, fill_value=0., mask=uv10_mask)
    v10ss = np.ma.array(v10ss.data, fill_value=0., mask=uv10_mask)
    #
    t2s,_,_,_              = get_data('data/trend/pace/t2_trend_pace.nc','T2')
    t2s_sig,_,_,_ = get_data('data/trend/pace/T2_DJFtrend_pace_sig.nc','T2')
    slps,_,_,_             = get_data('data/trend/pace/slp_trend_pace.nc','slp')    
    #

    #----HIST
    uom,lonm,latm,levm = get_data('data/trend/miroc6/uo_trend_miroc6.nc','uo')
    vom,_,_,_          = get_data('data/trend/miroc6/vo_trend_miroc6.nc','vo')
    tom,_,_,_          = get_data('data/trend/miroc6/to_trend_miroc6.nc','to')
    
    sicm,_,_,_         = get_data('data/trend/miroc6/aig_trend_miroc6.nc','aig')
    sicm *= 1.e2 # to %
    uim,_,_,_          = get_data('data/trend/miroc6/ui_trend_miroc6.nc','ui')    
    vim,_,_,_          = get_data('data/trend/miroc6/vi_trend_miroc6.nc','vi')        
    
    u10m,lonam,latam,levam = get_data('data/trend/miroc6/u10_trend_miroc6.nc','u10')
    v10m,_,_,_             = get_data('data/trend/miroc6/v10_trend_miroc6.nc','v10')
    t2m,_,_,_              = get_data('data/trend/miroc6/t2_trend_miroc6.nc','T2')
    slpm,_,_,_             = get_data('data/trend/miroc6/slp_trend_miroc6.nc','slp')    
    
    #----mask lev
    ilev2 = np.where(levs==zlev2)[0][0]    

    #-------------------------------------#    
    # fig2
    #-------------------------------------#
    #diff = False
    diff = True
    
    fontsize  = 8
    fontsize2 = 9
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #----make figure
    fig = plt.figure(figsize=(10,12))
    
    #----make gridspec & axis
    gs_master = GridSpec(nrows=3, ncols=26, height_ratios=[1,1,1],hspace=0.1,wspace=0.6) 
    
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,0:10]) #
    ax1 = fig.add_subplot(gs_1[:, :])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,14:24])
    ax2 = fig.add_subplot(gs_2[:, :])
    gs_3l = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0,10]) #
    ax3l = fig.add_subplot(gs_3l[:, :])
    gs_3r = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0,24]) #
    ax3r = fig.add_subplot(gs_3r[:, :])

    gs_3rr = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:,25]) #
    ax3rr = fig.add_subplot(gs_3rr[:, :])
    ax3rr.axis("off")    
    
    gs_7 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[1,0:10]) #
    ax7 = fig.add_subplot(gs_7[:, :])
    gs_8 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[1,14:24])
    ax8 = fig.add_subplot(gs_8[:, :])    
    gs_9l = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[1,10]) #
    ax9l = fig.add_subplot(gs_9l[:, :])
    gs_9r = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[1,24]) #
    ax9r = fig.add_subplot(gs_9r[:, :])    

    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,0:10]) #
    ax4 = fig.add_subplot(gs_4[:, :])
    gs_5 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,14:24])
    ax5 = fig.add_subplot(gs_5[:, :])    
    gs_6l = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,10]) #
    ax6l = fig.add_subplot(gs_6l[:, :])
    gs_6r = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,24]) #
    ax6r = fig.add_subplot(gs_6r[:, :])    
    
    #----plot
    scale = 10
    veclen = 0.5
    
    #----SIC (first row)
    m1,cs1,q1 = plot_BK_each(uis[0],vis[0],sics[0],0,lons,lats,fig,ax1,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-20,2,5),
                             cmap=plt.cm.Blues_r,
                             plot_contour=False,
                             shadein_sig=sics_sig[0],
                             usig=uiss[0],vsig=viss[0])
    
    m2,cs2,q2 = plot_BK_each(uis[0]-uim[0],vis[0]-vim[0],sics[0]-sicm[0],0,lons,lats,fig,ax2,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-5,6,1),
                             cmap=plt.cm.coolwarm,
                             plot_contour=False,
                             print_veclen=True,
                             vecunit="cm$s^{{-}1}$ decade$^{-1}$",
                             shadein_sig=sics_sig[0]-sicm[0],
                             usig=uiss[0]-uim[0],vsig=viss[0]-vim[0])
    #----colorbar
    cb1 = plt.colorbar(cs1,ax=ax3l, orientation="vertical",fraction=0.3,ticks=[-20,-15,-10,-5,0])
    cb1.ax.set_yticklabels(['-20','-15','-10','-5','0'])
    ax3l.axis("off")
    cb1.set_label('[% decade$^{-1}$]',fontsize=9)
    
    cb2 = plt.colorbar(cs2,ax=ax3r, orientation="vertical",fraction=0.3,ticks=[-4,-2,0,2,4])
    cb2.ax.set_yticklabels(['-4','-2','0','2','4'])
    ax3r.axis("off")
    cb2.set_label('[% decade$^{-1}$]',fontsize=9)
    
    #----draw box
    draw_lonlat_box([20,70,65,85],m1,ax1)
    draw_lonlat_box([20,70,65,85],m2,ax2)    
    #----draw title    
    ax1.set_title('(a) Sea-ice (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax2.set_title('(b) Sea-ice (NAGA-HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')

    #----------------------------------------------------------#
    #----ocean surface u,v,t (second row)
    m4,cs4,q4 = plot_BK_each(uos[ilev2],vos[ilev2],tos[ilev2],sics[0],lons,lats,fig,ax4,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-0.1,1.1,0.1),
                             cmap=plt.cm.Reds,
                             plot_contour=False,
                             cclevs=np.arange(-20,2,5),
                             shadein_sig=ssts_sig[0],
                             usig=uoss[0],vsig=voss[0]) 
    m5,cs5,q5 = plot_BK_each(uos[ilev2]-uom[ilev2],vos[ilev2]-vom[ilev2],tos[ilev2]-tom[ilev2],sics[0]-sicm[0],lons,lats,fig,ax5,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-0.5,0.6,0.1),
                             cmap=plt.cm.coolwarm,
                             plot_contour=False,
                             cclevs=np.arange(-6,7,3),
                             print_veclen=True,
                             vecunit="cm$s^{{-}1}$ decade$^{-1}$",
                             shadein_sig=ssts_sig[0]-tom[ilev2],
                             usig=uoss[0]-uom[ilev2],vsig=voss[0]-vom[ilev2])                              
    #----colorbar
    cb4 = plt.colorbar(cs4,ax=ax6l, orientation="vertical",fraction=0.3,ticks=[0,0.2,0.4,0.6,0.8,1])
    ax6l.axis("off")
    cb4.set_label('[$^{\circ}$C decade$^{-1}$]',fontsize=9)
    cb5 = plt.colorbar(cs5,ax=ax6r, orientation="vertical",fraction=0.3,ticks=[-0.4,-0.2,0,0.2,0.4])
    ax6r.axis("off")
    cb5.set_label('[$^{\circ}$C decade$^{-1}$]',fontsize=9)    
        
    #----draw title    
    ax4.set_title('(e) Surface Ocean (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax5.set_title('(f) Surface Ocean (NAGA-HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')

    #----draw box
    linecolor='black'
    linestyle='--'
    draw_lonlat_box([20,20,70,80],m4,ax4,color=linecolor,lw=2,linestyle=linestyle)
    draw_lonlat_box([20,20,70,80],m5,ax5,color=linecolor,lw=2,linestyle=linestyle)
    draw_lonlat_box([-20,20,70,70],m4,ax4,color=linecolor,lw=2,linestyle=linestyle)
    draw_lonlat_box([-20,20,70,70],m5,ax5,color=linecolor,lw=2,linestyle=linestyle)                
    #
    #----------------------------------------------------------#
    #----ATM u10,v10,T2, slp
    scale = 2
    veclen = 0.1
    
    m7,cs7,q7 = plot_BK_each(u10s[0],v10s[0],t2s[0],slps[0],lonas,latas,fig,ax7,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-1,5.5,0.5),
                             cmap=plt.cm.Reds,
                             plot_atm=True,
                             plot_contour=True,
                             cclevs=np.arange(-0.5,0.6,0.1),
                             shadein_sig=t2s_sig[0],
                             usig=u10ss[0],vsig=v10ss[0])
    
    m8,cs8,q8 = plot_BK_each(u10s[0]-u10m[0],v10s[0]-v10m[0],t2s[0]-t2m[0],slps[0]-slpm[0],lonas,latas,fig,ax8,
                             scale=scale,veclen=veclen,
                             sclevs=np.arange(-1.,1.1,0.2),
                             cmap=plt.cm.coolwarm,
                             plot_atm=True,
                             plot_contour=True,
                             cclevs=np.arange(-0.5,0.6,0.1),
                             print_veclen=True,
                             vecunit="m$s^{{-}1}$ decade$^{-1}$",
                             shadein_sig=t2s_sig[0]-t2m[0],
                             usig=u10ss[0]-u10m[0],vsig=v10ss[0]-v10m[0])                             
    #----colorbar
    cb7 = plt.colorbar(cs7,ax=ax9l, orientation="vertical",fraction=0.3,ticks=[-1,0,1,2,3,4,5])
    ax9l.axis("off")
    cb7.set_label('[$^{\circ}$C decade$^{-1}$]',fontsize=9)        
    cb8 = plt.colorbar(cs8,ax=ax9r, orientation="vertical",fraction=0.3,ticks=[-1,-0.5,0,0.5,1])
    ax9r.axis("off")
    cb8.set_label('[$^{\circ}$C decade$^{-1}$]',fontsize=9)            
    #----draw box
    #draw_lonlat_box([20,70,65,85],m7,ax7)
    #draw_lonlat_box([20,70,65,85],m8,ax8)    
    #----draw title    
    ax7.set_title('(c) Bottom Atmosphere (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax8.set_title('(d) Bottom Atmosphere (NAGA-HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')

    plt.savefig('fig2.pdf', bbox_inches="tight", pad_inches=0.3)
    #plt.savefig('Mainfig2_revise_hatch_test.png', bbox_inches="tight", pad_inches=0.3)    
    subprocess.call('open fig2.pdf',shell=True)    
    plt.close()    
    
    ###############################################################
    ###############################################################
    ###############################################################
    
