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
import copy
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

    #----remove nan
    nan_mask = np.isnan(tmpdata.data)
    mask_candidate = np.array([nan_mask,tmpdata.mask])
    new_mask = np.any(mask_candidate,axis=0) 
    #print(mask_candidate.shape,nan_mask.shape,tmpdata.mask.shape,new_mask.shape)        
    tmpdata = np.ma.array(tmpdata.data, fill_value=-999., mask=new_mask)
    
    return tmpdata,lon,lat,lev

def plot_each(uin,vin,shadein,contourin,inlon,inlat,fig,ax,
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
              vsig=None,
              vecoff=False):

    #---- per decade
    uino = uin*10.
    vino = vin*10.
    sino = shadein*10.    
    cino = contourin*10.
    if (usig is not None)&(vsig is not None):
        usig = usig*10.
        vsig = vsig*10.

    lonos, latos = np.meshgrid(inlon,inlat) # make lats and lons
    
    #----make basemap
    if lonlat:
        m = Basemap(llcrnrlon=-30,llcrnrlat=60,urcrnrlon=30,urcrnrlat=85,projection='cyl',resolution='c',ax=ax)
    else:
        m = Basemap(resolution='l',projection='lcc', lat_0=55, lon_0=-40,width=8500000, height=6000000,ax=ax)
        
    m.drawmapboundary(fill_color='white',color='black',linewidth=1.0)
    if not plot_atm:
        m.fillcontinents(color='gray',lake_color='gray')    
    else:
        m.drawcoastlines(linewidth=1.0)
    m.drawparallels(np.arange(-90.,99.,10.),labels=[True,False,False,False],fontsize=fontsize,linewidth=0.3)
    m.drawmeridians(np.arange(-180.,180.,30.),labels=[False,False,False,True],fontsize=fontsize,linewidth=0.3)
    mlons, mlats = m(lonos,latos)
    
    #----shade
    cs1 = m.contourf(mlons,mlats,sino,cmap=cmap,extend='both',levels=sclevs,alpha=1,latlon=lonlat)
    if not shadein_sig is None:
        sino_sig = shadein_sig*10.                
        mpl.rcParams['hatch.linewidth'] = 0.2   # hatch linewidth
        mpl.rcParams['hatch.color'] = 'black' # hatch color
        cs_hatch = m.contourf(mlons,mlats,sino_sig,colors='none',extend='both',
                              levels=sclevs,alpha=0,latlon=lonlat,hatches=['/////'])
    #----vector
    vintlon = 3 #vector interval
    vintlat = 3 #vector interval
    #####
    if vecoff:
        return m,cs1,None
    else:
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
    
    #--------------------------------------#
    # read data
    data_path = './data/trend/' 
    
    #----NAGA
    uos,lons,lats,levs = get_data(data_path + 'pace/um_DJFtrend_pace.nc','um')    
    vos,_,_,_          = get_data(data_path + 'pace/vm_DJFtrend_pace.nc','vm')
    uoss,_,_,_         = get_data(data_path + 'pace/um_DJFtrend_pace_sig.nc','um')    
    voss,_,_,_         = get_data(data_path + 'pace/vm_DJFtrend_pace_sig.nc','vm')
    uvo_mask = (uoss.mask | voss.mask)
    uoss.data[uvo_mask]=0
    voss.data[uvo_mask]=0    
    uoss = np.ma.array(uoss.data, fill_value=0., mask=uvo_mask)
    voss = np.ma.array(voss.data, fill_value=0., mask=uvo_mask)
    tos,_,_,_          = get_data(data_path + 'pace/tm_DJFtrend_pace.nc','tm')
    tos_sig,_,_,_ = get_data(data_path + 'pace/tm_DJFtrend_pace_sig.nc','tm')

    # NAGA mld budget
    velos,_,_,_    = get_data(data_path + 'pace/velo_DJFtrend_pace.nc','velo')    
    velos_sig,_,_,_= get_data(data_path + 'pace/velo_DJFtrend_pace_sig.nc','velo')    
    advs,_,_,_     = get_data(data_path + 'pace/adv_DJFtrend_pace.nc','adv')
    advs_sig,_,_,_ = get_data(data_path + 'pace/adv_DJFtrend_pace_sig.nc','adv')
    hfxs,_,_,_     = get_data(data_path + 'pace/hfx_DJFtrend_pace.nc','hfx')
    hfxs_sig,_,_,_ = get_data(data_path + 'pace/hfx_DJFtrend_pace_sig.nc','hfx')

    ums = copy.copy(uos)
    vms = copy.copy(vos)
    
    #--------------------------------------#    
    #----HIST
    uom,lonm,latm,levm = get_data(data_path + 'miroc6/um_DJFtrend_miroc6.nc','um')    
    vom,_,_,_          = get_data(data_path + 'miroc6/vm_DJFtrend_miroc6.nc','vm')
    uoms,_,_,_         = get_data(data_path + 'miroc6/um_DJFtrend_miroc6_sig.nc','um')    
    voms,_,_,_         = get_data(data_path + 'miroc6/vm_DJFtrend_miroc6_sig.nc','vm')
    uvo_mask = (uoms.mask | voms.mask)
    uoms.data[uvo_mask]=0
    voms.data[uvo_mask]=0    
    uoms = np.ma.array(uoms.data, fill_value=0., mask=uvo_mask)
    voms = np.ma.array(voms.data, fill_value=0., mask=uvo_mask)
    tom,_,_,_          = get_data(data_path + 'miroc6/tm_DJFtrend_miroc6.nc','tm')
    tom_sig,_,_,_ = get_data(data_path + 'miroc6/tm_DJFtrend_miroc6_sig.nc','tm')
    #
    # HIST mld budget
    velom,_,_,_     = get_data(data_path + 'miroc6/velo_DJFtrend_miroc6.nc','velo')    
    velom_sig,_,_,_ = get_data(data_path + 'miroc6/velo_DJFtrend_miroc6_sig.nc','velo')    
    advm,_,_,_      = get_data(data_path + 'miroc6/adv_DJFtrend_miroc6.nc','adv')
    advm_sig,_,_,_  = get_data(data_path + 'miroc6/adv_DJFtrend_miroc6_sig.nc','adv')
    hfxm,_,_,_      = get_data(data_path + 'miroc6/hfx_DJFtrend_miroc6.nc','hfx')
    hfxm_sig,_,_,_  = get_data(data_path + 'miroc6/hfx_DJFtrend_miroc6_sig.nc','hfx')
    umm = copy.copy(uom)
    vmm = copy.copy(vom)
    #
    # m/s to cm/s
    ums *= 100.
    vms *= 100.
    umm *= 100.
    vmm *= 100.
    velos *= 100.
    velom *= 100.
    velos_sig *= 100.
    velom_sig *= 100.    
    uom  *= 100.
    vom  *= 100.
    uoms *= 100.
    voms *= 100.
    uos  *= 100.
    vos  *= 100.
    uoss *= 100.
    voss *= 100.            
    #
    
    #----mask lev
    ilev2 = 0
    
    #-------------------------------------#    
    # fig2
    #-------------------------------------#
    fontsize  = 8
    fontsize2 = 9
    params = {'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    pylab.rcParams.update(params)
    
    #----make figure
    fig = plt.figure(figsize=(9,13))
    
    #----make gridspec & axis
    gs_master = GridSpec(nrows=7, ncols=22, height_ratios=[10,2,10,2,10,2,10],hspace=0,wspace=0) 
    
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,0:10]) #
    ax1 = fig.add_subplot(gs_1[:, :])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[0,11:21])
    ax2 = fig.add_subplot(gs_2[:, :])
    gs_3l = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0,10]) #
    ax3l = fig.add_subplot(gs_3l[:, :])
    gs_3r = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0,21]) #
    ax3r = fig.add_subplot(gs_3r[:, :])
    
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,0:10]) #
    ax4 = fig.add_subplot(gs_4[:, :])
    gs_5 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,11:21])
    ax5 = fig.add_subplot(gs_5[:, :])    
    gs_6l = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,10]) #
    ax6l = fig.add_subplot(gs_6l[:, :])
    gs_6r = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[2,21]) #
    ax6r = fig.add_subplot(gs_6r[:, :])    
    
    gs_7 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,0:10]) #
    ax7 = fig.add_subplot(gs_7[:, :])
    gs_8 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,11:21])
    ax8 = fig.add_subplot(gs_8[:, :])    
    gs_9l = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,10]) #
    ax9l = fig.add_subplot(gs_9l[:, :])
    gs_9r = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[4,21]) #
    ax9r = fig.add_subplot(gs_9r[:, :])    

    gs_10 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[6,0:10]) #
    ax10 = fig.add_subplot(gs_10[:, :])
    gs_11 = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[6,11:21])
    ax11 = fig.add_subplot(gs_11[:, :])    
    gs_12r = GridSpecFromSubplotSpec(nrows=1, ncols=10, subplot_spec=gs_master[6,21]) #
    ax12r = fig.add_subplot(gs_12r[:, :])    
    
    #----------------------------------------------------------#
    #----u,v,velo clim
    scale = 30
    veclen = 2
    m1,cs1,q1 = plot_each(uom[0],vom[0],velom[0],0,lons,lats,fig,ax1,
                          scale=scale,veclen=veclen,
                          sclevs=np.arange(-1,1.1,0.1),
                          cmap=plt.cm.coolwarm,
                          plot_contour=False,
                          shadein_sig=velom_sig[0],
                          usig=uoms[0],vsig=voms[0])
    
    m2,cs2,q2 = plot_each(uos[0],vos[0],velos[0],0,lons,lats,fig,ax2,
                          scale=scale,veclen=veclen,
                          sclevs=np.arange(-1,1.1,0.1),
                          cmap=plt.cm.coolwarm,
                          plot_contour=False,
                          print_veclen=True,
                          vecunit="cm$s^{{-}1}$ decade$^{-1}$",
                          shadein_sig=velos_sig[0],
                          usig=uoss[0],vsig=voss[0])
    
    #----colorbar
    ax3l.axis("off")
    cb2 = plt.colorbar(cs2,ax=ax3r, orientation="vertical",fraction=0.3,ticks=[-1,-0.5,0,0.5,1])
    ax3r.axis("off")
    cb2.set_label('[cm s$^{-1}$ decade$^{-1}$]',fontsize=12)    
    #
    #----draw title    
    ax1.set_title('(a) Velocity (HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax2.set_title('(b) Velocity (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    #
    draw_lonlat_box([-80,-30,30,50],m1,ax1)
    draw_lonlat_box([-80,-30,30,50],m2,ax2)

    #----------------------------------------------------------#
    #----ocean surface u,v,t (second row)
    m4,cs4,q4 = plot_each(uom[ilev2],vom[ilev2],tom[ilev2],velom[0],lons,lats,fig,ax4,
                          scale=scale,veclen=veclen,
                          sclevs=np.arange(-0.5,0.6,0.1),                             
                          cmap=plt.cm.coolwarm,
                          plot_contour=True,
                          cclevs=[0.2,0.6,1.0],
                          vecoff=True,
                          shadein_sig=tom_sig[0],
                          usig=uoms[0],vsig=voms[0])
    
    m5,cs5,q5 = plot_each(uos[ilev2],vos[ilev2],tos[ilev2],velos[0],lons,lats,fig,ax5,
                          scale=scale,veclen=veclen,
                          sclevs=np.arange(-0.5,0.6,0.1),
                          cmap=plt.cm.coolwarm,
                          plot_contour=True,
                          print_veclen=False,
                          vecunit="cm$s^{{-}1}$ decade$^{-1}$",
                          cclevs=[0.2,0.6,1.0],                             
                          vecoff=True,                             
                          shadein_sig=tos_sig[0],
                          usig=uoms[0],vsig=voms[0])
    #----colorbar
    ax6l.axis("off")
    cb5 = plt.colorbar(cs5,ax=ax6r, orientation="vertical",fraction=0.3,ticks=[-0.4,-0.2,0,0.2,0.4])
    ax6r.axis("off")
    cb5.set_label('[$^{\circ}$C decade$^{-1}$]',fontsize=12)

    #----draw title    
    ax4.set_title('(c) Potential temperature (HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax5.set_title('(d) Potential temperature (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    #
    draw_lonlat_box([-80,-30,30,50],m4,ax4)
    draw_lonlat_box([-80,-30,30,50],m5,ax5)    
    #
    #----------------------------------------------------------#
    #----advection,um,vm
    advs  *= 1.e8
    advm  *= 1.e8
    
    sclevs = np.arange(-5.,5.1,1)
    sclevsdiff = np.arange(-5.,5.1,1)
    sticks = [-4,-2,0,2,4]
    sticksdiff = [-4,-2,0,2,4]
    
    m7,cs7,q7 = plot_each(umm[0],vmm[0],advm[0],velom[0],lons,lats,fig,ax7,
                          scale=scale,veclen=veclen,
                          cmap=plt.cm.coolwarm,
                          sclevs=sclevs,
                          plot_contour=True,
                          vecoff=True,
                          cclevs=[0.2,0.6,1.0],                               
                          shadein_sig=advm_sig[0])
    m8,cs8,q8 = plot_each(ums[0],vms[0],advs[0],velos[0],lons,lats,fig,ax8,
                          scale=scale,veclen=veclen,
                          cmap=plt.cm.coolwarm,
                          sclevs=sclevs,
                          plot_contour=True,
                          vecoff=True,
                          cclevs=[0.2,0.6,1.0],                             
                          shadein_sig=advs_sig[0])
    
    #----colorbar
    ax9l.axis("off")
    cb8 = plt.colorbar(cs8,ax=ax9r, orientation="vertical",fraction=0.3,ticks=sticks)
    ax9r.axis("off")
    cb8.set_label('[Ks$^{-1}$ decade$^{-1}$]',fontsize=12)    

    #----draw box
    draw_lonlat_box([-80,-30,30,50],m7,ax7)
    draw_lonlat_box([-80,-30,30,50],m8,ax8)        
    #----draw title    
    ax7.set_title('(e) Oceanic term (HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax8.set_title('(f) Oceanic term (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    
    #----------------------------------------------------------#
    #----heatflux term, net heatflux
    hfxs *= 1.e8
    hfxm *= 1.e8
    m10,cs10,q10 = plot_each(umm[0],vmm[0],hfxm[0],velom[0],lons,lats,fig,ax10,
                             scale=scale,veclen=veclen,
                             cmap=plt.cm.coolwarm,
                             sclevs=sclevs,
                             plot_contour=True,
                             cclevs=[0.2,0.6,1.0],                                                             
                             vecoff=True,
                             shadein_sig=hfxm_sig[0])
    m11,cs11,q11 = plot_each(ums[0],vms[0],hfxs[0],velos[0],lons,lats,fig,ax11,
                             scale=scale,veclen=veclen,
                             sclevs=sclevs,
                             cmap=plt.cm.coolwarm,
                             plot_contour=True,
                             cclevs=[0.2,0.6,1.0],                                                             
                             vecoff=True,
                             shadein_sig=hfxs_sig[0])
    #----colorbar
    cb11 = plt.colorbar(cs8,ax=ax12r, orientation="vertical",fraction=0.3,ticks=sticks)
    ax12r.axis("off")
    cb11.set_label('[Ks$^{-1}$ decade$^{-1}$]',fontsize=12)
    
    #----draw title    
    ax10.set_title('(g) Atmospheric term (HIST)',fontsize=fontsize2, y=1,x=0.0,loc='left')
    ax11.set_title('(h) Atmospheric term (NAGA)',fontsize=fontsize2, y=1,x=0.0,loc='left')    

    draw_lonlat_box([-80,-30,30,50],m10,ax10)
    draw_lonlat_box([-80,-30,30,50],m11,ax11)

    #plt.show()
    #sys.exit()
    plt.savefig('fig4.pdf', bbox_inches="tight", pad_inches=0.3)
    subprocess.call('open fig4.pdf',shell=True)
    plt.close()    
    
