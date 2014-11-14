#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting tools

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import cPickle
from mpl_toolkits.basemap import Basemap

import analysis_tools as at
import cosmo_tools as ct

from analysis_tools import _z_bins
from cosmo_tools import _d2r


_markers = ['ro','bo','go','ko','yo']

# --------------- #
# -- Utilities -- #
# --------------- #

def healpy_hist(l,b,NSIDE=4,mask_zero=False):
    """
    """
    l = np.array(l)
    b = np.array(b)
    
    pixels = hp.ang2pix(NSIDE,(90-b)*np.pi/180,l*np.pi/180)
    pix_hist = np.histogram(pixels,bins=range(hp.nside2npix(NSIDE)+1))[0]
    
    if mask_zero:
        return (pix_hist[pix_hist>0],
                np.arange(hp.nside2npix(NSIDE))[pix_hist>0])
    else:
        return pix_hist, np.arange(hp.nside2npix(NSIDE))

# ---------------- #
# -- Histograms -- #
# ---------------- #



# ------------------- #
# --   Sky plots   -- #
# -- using Basemap -- #
# ------------------- #

def basic_basemap(projection='moll',figsize=(8,6),color='k',
                  frame='galactic',marks=True,label_color='k'):
    """
    """
    if label_color is None:
        label_color = color

    fig = plt.figure(figsize=figsize)
    m = Basemap(projection=projection,lon_0=0,lat_0=0,celestial=True)
    m.drawparallels(np.arange(-90.,90.,30.),color=color,linewidth=0.5)
    m.drawmeridians(np.arange(-180.,180.,60.),color=color,linewidth=0.5)   

    pol_l = [180,120,60,0,-60,-120,-180,0,0,0,0]
    pol_b = [0,0,0,0,0,0,0,30,60,-30,-60]
    pol = ['180','120','60','0','300','240','','30','60','-30','-60']  

    tick_x,tick_y=m(pol_l,pol_b)
    for name,xpt,ypt in zip(pol,tick_x,tick_y):
        plt.text(xpt+50000,ypt+50000,name,color=color,size=8)

    Marked=[['SSC',2.1480,3.4287],
            ['CMB',2.0478543,2.908704],
            ['N',0,0],['S',np.pi,0]]

    if frame == 'galactic':
        plt.xlabel('l',fontsize=18,color=label_color)
        plt.ylabel('b',fontsize=18,color=label_color)

        if marks:
            for item in Marked:
                mark=color+'.'
                l1,b1 = ct.radec2gcs(item[2]*180/np.pi,(np.pi/2-item[1])*180/np.pi)
                l_temp,b_temp = m([l1],[b1])
                if item[0] in ['N','S']:    
                    plt.plot(l_temp,b_temp,color+'+',ms=8,lw=2)
                else:
                    plt.plot(l_temp,b_temp,mark,markersize=10)
                if item[0]=='SSC':
                    plt.text(l_temp[0]-700000,b_temp[0]+400000,item[0],
                             color=color,size=14)
                else:
                    plt.text(l_temp[0]+200000,b_temp[0]+200000,item[0],
                             color=color,size=14)
    elif frame == 'j2000':        
        plt.xlabel('RA',fontsize=18,color=label_color)
        plt.ylabel('Dec',fontsize=18,color=label_color)

        if marks:
            for item in Marked:
                mark=color+'.'
                l1,b1 = (item[2]*180/np.pi,(np.pi/2-item[1])*180/np.pi)
                l_temp,b_temp = m([l1],[b1])
                if item[0] not in ['N','S']:    
                    plt.plot(l_temp,b_temp,mark,markersize=10)
                    if item[0]=='SSC':
                        plt.text(l_temp[0]-2000000,b_temp[0]+300000,item[0],
                                 color=color,size=14)
                    else:
                        plt.text(l_temp[0]+200000,b_temp[0]+200000,item[0],
                                 color=color,size=14)
        
    else:
        raise ValueError('frame unknown {}; must be "galactic" or "j2000"'.format(frame))
    

    return fig, m


def healpy_basemap(values,NSIDE=4,pixels=None,steps=4,vmin=None,
                   vmax=None,projection='moll',figsize=(8,6),
                   cmap=plt.get_cmap('Blues'),color='k',
                   frame='galactic',marks=True,cbar=True,cbar_label=None,
                   cbar_orientation='horizontal'):
    """
    """
    fig, m = basic_basemap(projection=projection,figsize=figsize,
                           color=color,frame=frame,marks=marks)

    if pixels is None:
        pixels = range(hp.nside2npix(NSIDE))
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)

    for pix,count in zip(pixels,values):
        corners = hp.boundaries(NSIDE,pix,step=steps)
        corners_b, corners_l = hp.vec2ang(np.transpose(corners))
        l_raw = corners_l/_d2r
        l_edges = (corners_l/_d2r)%360
        b_edges = 90 - corners_b/_d2r

        l_new = np.zeros((steps+1,steps+1))
        b_new = np.zeros((steps+1,steps+1))
        for new, old in zip([l_new,b_new],[l_edges,b_edges]):
            new[0,:] = old[:steps+1]
            new[1:,-1] = old[steps+1:2*steps+1]
            new[-1,-2::-1] = old[2*steps+1:3*steps+1]
            new[-2:0:-1,0] = old[3*steps+1:]

            for k in xrange(1,steps):
                new[k,:-1] = np.ones(steps)*new[k,0]

        count_arr = np.ones((steps+1,steps+1)) * count

        if np.sum((np.abs(l_new-180)<10).astype(int)) > 0:  
            l_temp = np.fmin(l_new,179.9999)
            # Check that there are points not on the edges
            num_on_edges = np.sum(((l_temp != 179.9999) 
                                 & (np.abs(b_new) != 90)).astype(int))
            if num_on_edges > 0:
                x,y = m(l_temp,b_new)
                m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap)

            l_temp = np.fmax(l_new,180.0001)
            # Check that there are points not on the edges
            num_on_edges = np.sum(((l_temp != 180.0001) 
                                 & (np.abs(b_new) != 90)).astype(int))
            if num_on_edges > 0:
                x,y = m(l_temp,b_new)
                m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap)
        else:
            x,y = m(l_new,b_new)
            m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap)

    if not cbar:
        return fig, m
    else:
        cbar = plt.colorbar(orientation=cbar_orientation)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

        return fig, m, cbar
    

# ----------------- #
# -- Other plots -- #
# ----------------- #

def plot_results(result,prefixes=None,names=None,cumulative=False,figsize=(8,6),save2file=None,
                 z_range=(0.,0.12),y_range=None,y_label=None,legend='upper left',
                 connect_w_line=True,title=None,z_bins=None,markers=None,median_fct=np.median):
    """
    """
    if prefixes is None:
        prefixes = sorted(result.keys())
        
    if names is None:
        names = [prefix.replace('_',' ') for prefix in prefixes]

    if len(names) < len(prefixes):
        raise ValueError('Require as many names as prefixes')

    if z_bins is None:
        z_bins = _z_bins
    
    if markers is None:
        markers = _markers

    if len(markers) < len(prefixes):
        raise ValueError('Require as many markers as prefixes')

    if cumulative:
        z = z_bins[1:]
    else:
        z = [np.mean([z_min,z_max]) for z_min, z_max in zip(z_bins[:-1],z_bins[1:])]
    
    fig = plt.figure(figsize=figsize)
    for (prefix,name,marker) in zip(prefixes,names,markers):
        plt.plot(z,[np.median(a) for a in result[prefix]],marker,ms=8,label=name)
        if connect_w_line:
            plt.plot(z,[median_fct(a) for a in result[prefix]],marker[0]+'-')
        
    if not cumulative:
        for z_val in z_bins:
            plt.plot([z_val,z_val],[-1e6,1e6],'k--',scaley=False)
    
    if legend is not None:
        plt.legend(loc=legend)
        
    if z_range is not None:
        plt.xlim(z_range)
        
    if y_range is not None:
        plt.ylim(y_range)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if cumulative:
        plt.xlabel(r'$z_{max}$',fontsize=20)
    else:
        plt.xlabel(r'$z_{mean}$',fontsize=20)        
    
    if y_label is not None:
        plt.ylabel(y_label,fontsize=20)

    if title is not None:
        plt.title(title,fontsize=20)

    if save2file is not None:
        plt.savefig(save2file)
        
    return fig
