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

import warnings
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import cPickle
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import coords

# import analysis_tools as at
# import cosmo_tools as ct

_z_bins = [0.015,0.035,0.045,0.06,0.1]
_d2r = np.pi / 180

_nest = True
_markers = ['ro','bo','go','ko','yo']
_cmaps = ['Reds','Blues','Greens','Greys','Oranges']

# --------------- #
# -- Utilities -- #
# --------------- #

def healpy_hist(l,b,NSIDE=4,mask_zero=False,nest=_nest):
    """
    """
    l = np.array(l)
    b = np.array(b)
    
    pixels = hp.ang2pix(NSIDE,(90-b)*_d2r,l*_d2r,nest=nest)
    pix_hist = np.histogram(pixels,bins=range(hp.nside2npix(NSIDE)+1))[0]
    
    if mask_zero:
        return (pix_hist[pix_hist>0],
                np.arange(hp.nside2npix(NSIDE))[pix_hist>0])
    else:
        return pix_hist, np.arange(hp.nside2npix(NSIDE))

def healpy_grid(NSIDE,nest=_nest):
    """
    """
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), nest=nest)
    l = phi / _d2r
    b = 90 - theta / _d2r

    return l, b

def healpy_values(l, b, values, NSIDE=None, nest=_nest, project_j2000=False):
    """
    """
    if NSIDE is None:
        NSIDE = hp.npix2nside(len(values))

    if not project_j2000:
        phi = (l * _d2r).flatten()
        theta = ((90 - b) * _d2r).flatten()
    else:
        ra = []
        dec = []
        for x in zip(l,b):
            pos = coords.Position(x, system='galactic')
            ra.append(pos.j2000()[0])
            dec.append(pos.j2000()[1])
        ra = np.array(ra)
        dec = np.array(dec)
        phi = (ra * _d2r).flatten()
        theta = ((90 - dec) * _d2r).flatten()
        
    pix_id = hp.ang2pix(NSIDE, theta, phi, nest=nest)

    return values[pix_id]

# ---------------- #
# -- Histograms -- #
# ---------------- #



# ------------------- #
# --   Sky plots   -- #
# -- using Basemap -- #
# ------------------- #

def basic_basemap(projection='moll',figsize=(8,6),color='k',
                  frame='galactic',marks=True,label_color='k',
                  **kwargs):
    """
    """
    if label_color is None:
        label_color = color

    fig = plt.figure(figsize=figsize)
    m = Basemap(projection=projection,lon_0=0,lat_0=0,celestial=True,**kwargs)
    m.drawparallels(np.arange(-90.,90.,30.),color=color,linewidth=0.5)
    m.drawmeridians(np.arange(-180.,180.,60.)[1:],color=color,linewidth=0.5)   

    pol_l = [180,120,60,0,-60,-120,-180,0,0,0,0]
    pol_b = [0,0,0,0,0,0,0,30,60,-30,-60]
    pol = ['180','120','60','0','300','240','','30','60','-30','-60']  

    tick_x,tick_y=m(pol_l,pol_b)
    for name,xpt,ypt in zip(pol,tick_x,tick_y):
        plt.text(xpt+50000,ypt+50000,name,color=color,size=12)

    Marked=[['SSC',2.1480,3.4287],
            ['CMB',2.0478543,2.908704],
            ['N',0,0],['S',np.pi,0]]

    if frame == 'galactic':
        # plt.xlabel('l',fontsize=18,color=label_color)
        # plt.ylabel('b',fontsize=18,color=label_color)
        plt.xlabel(r'$l$',fontsize=22,color=label_color)
        plt.ylabel(r'$b$',fontsize=22,color=label_color)

        if marks:
            for item in Marked:
                mark=color+'.'
                l1,b1 = radec2gcs(item[2]*180/np.pi,(np.pi/2-item[1])*180/np.pi)
                l_temp,b_temp = m([l1],[b1])
                if item[0] in ['N','S']:    
                    plt.plot(l_temp,b_temp,color+'+',ms=8,lw=2)
                else:
                    plt.plot(l_temp,b_temp,mark,markersize=10)
                if item[0]=='SSC':
                    plt.text(l_temp[0]-700000,b_temp[0]+400000,item[0],
                             color=color,size=16)
                else:
                    plt.text(l_temp[0]+200000,b_temp[0]+200000,item[0],
                             color=color,size=16)
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

def healpy_basemap(values, NSIDE=4, vmin=None, vmax=None, projection='moll',
                   figsize=(8,6), cmap='Blues', color='k', frame='galactic', 
                   marks=True, cbar=True, cbar_label=None, nest=_nest, alpha=1,
                   cbar_orientation='horizontal', fig_m=None, contour=False, 
                   n_img_pix=(800,400), cbar_ticks=None, project_j2000=False,
                   healpy=True, **kwargs):
    """
    """
    if fig_m is None:
        fig, m = basic_basemap(projection=projection,figsize=figsize,
                               color=color,frame=frame,marks=marks)
    else:
        fig, m = fig_m

    # if pixels is None:
    #     pixels = range(hp.nside2npix(NSIDE))
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    if not contour:
        xg, yg, zgm = project_bitmap(m, healpy_values, args=(values,),
                                     kwargs={'nest': nest, 
                                             'project_j2000': project_j2000}, 
                                     n_img_pix=n_img_pix, healpy=healpy)
        plt.pcolormesh(xg, yg, zgm, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        if contour is True:
            contour = 'fill'
        
        if contour not in ['fill', 'line']:
            raise ValueError('contour must be "fill" or "line".')
        
        xg, yg, zgm = project_bitmap(m, healpy_values, args=(values,), 
                                     kwargs={'nest': nest, 
                                             'project_j2000': project_j2000}, 
                                     n_img_pix=n_img_pix, 
                                     for_contour=True, healpy=healpy)
        if contour == 'fill':
            crange = kwargs.pop('crange',None)
            if crange is None:
                plt.contourf(xg, yg, zgm, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            else:
                plt.contourf(xg, yg, zgm, crange, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        else:
            nlines= kwargs.pop('nlines',None)
            if nlines is None:
                plt.contour(xg, yg, zgm, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            else:
                plt.contour(xg, yg, zgm, nlines, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    if not cbar:
        return fig, m
    else:
        cbar = plt.colorbar(orientation=cbar_orientation, shrink=0.92, pad=0.08, 
                            ticks=cbar_ticks)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=20)

        return fig, m, cbar

def healpy_fancy(values, NSIDE, title=False, z_bin=False, save2file=False, 
                 mark=None, z_label=False, **kwargs):
    """
    """
    if 'figsize' not in kwargs.keys():
        if not kwargs.get('cbar',True):
            if title:
                figsize=(8,4.3)
            else:
                figsize=(8,4.1)
        else:
            if title:
                figsize=(8,5.4)
            else:
                figsize=(8,5.1)

    if kwargs.get('cbar',True):
        fig, m, cbar = healpy_basemap(values, NSIDE, figsize=figsize, **kwargs)
    else:
        fig, m = healpy_basemap(values, NSIDE, figsize=figsize, **kwargs)

    if mark is not None:
        for l, b, mcs in mark:
            x, y = m(l,b)
            plt.plot(x, y, mcs, ms=15)

    if z_label:
        plt.text(-2e6, 1.7e7, z_label, fontsize=22)    

    if z_bin and not z_label:
        plt.text(-2e6, 1.7e7, r'${}<z<{}$'.format(*z_bin), fontsize=22)    

    if title:
        plt.title(title, fontsize=24)
        if not kwargs.get('cbar',True):
            plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.05)
        else:
            plt.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.06)
    else:
        if not kwargs.get('cbar',True):
            plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06)
        else:
            plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06)

    
    if save2file:
        print 'saving', save2file
        if save2file.split('.')[-1] in ['png', 'jpg']:
            plt.savefig(save2file, dpi=100)
            plt.close(fig)
        else:
            if kwargs.get('contour',False) not in ['fill', 'line']:
                warnings.warn('Saving bitmap as vector graphics will create large files. Consider using contour plots instead.')
            plt.savefig(save2file)
            plt.close(fig)
    else:
        if kwargs.get('cbar',True):
            return fig, m, cbar
        else:
            return fig, m

def project_bitmap(m, f, args=None, kwargs=None, n_img_pix=(800,400), for_contour=False,
                   healpy=False):
    """
    """
    if args is None:
        args = ()

    if kwargs is None:
        kwargs = {}

    if type(n_img_pix) == int:
        n_img_pix = (n_img_pix, n_img_pix)

    l, b = np.meshgrid(np.linspace(-180,180,1000),np.linspace(-90,90,1000))
    x,y = m(l,b)

    xmin, xmax = np.min(x[x < 1e30]), np.max(x[x < 1e30])
    ymin, ymax = np.min(y[y < 1e30]), np.max(y[y < 1e30])
    
    xran = xmax - xmin
    yran = ymax - ymin

    dx = xran / n_img_pix[0]
    dy = yran / n_img_pix[1]
    
    x0, y0 = np.meshgrid(np.linspace(xmin - 0.05 * xran, xmax + 0.05 * xran, n_img_pix[0]),
                         np.linspace(ymin - 0.05 * yran, ymax + 0.05 * yran, n_img_pix[1]))

    l0, b0 = m(x0, y0, inverse=True)
    x1, y1 = m(l0, b0)
    mask = (((x0 - x1) ** 2 + (y0 - y1) ** 2) < 1).flatten()
    #mask = (((x0 - x1) ** 2 + (y0 - y1) ** 2) < 1e30).flatten()

    if not healpy:
        xg, yg = np.meshgrid(np.linspace(x0[0,0] - dx / 2, x0[-1,-1] + dx / 2,
                                         n_img_pix[0] + 1),
                             np.linspace(y0[0,0] - dy / 2, y0[-1,-1] + dy / 2, 
                                         n_img_pix[1] + 1))

        z = np.zeros(l0.shape).flatten()
        z[mask] = f(l0.flatten()[mask], b0.flatten()[mask], *args, **kwargs)
        z[~mask] = np.NaN

    if not for_contour:
        zg = z.reshape((n_img_pix[1], n_img_pix[0]))
        zgm = np.ma.array(zg, mask=np.isnan(zg))
        return xg, yg, zgm
    else:
        if healpy:
            xg, yg = m(*healpy_grid(hp.npix2nside(len(args[0])), nest=kwargs))
            zg = griddata((xg, yg), args[0], (x0, y0), method='linear')
            zgm = np.ma.array(zg, mask=~mask.reshape((n_img_pix[1], n_img_pix[0])))
            return x0, y0, zgm
        else:
            zg = griddata((x0.flatten()[mask], y0.flatten()[mask]), z[mask], 
                          (x0, y0), method='cubic')
            zgm = np.ma.array(zg, mask=~mask.reshape((n_img_pix[1], n_img_pix[0])))
            return x0, y0, zgm

def healpy_chi2(filename, attractor=False, fit_threshold=0, 
                void_healpy=None, **kwargs):
    """
    """
    results = cPickle.load(open(filename))

    chi2_no_v = kwargs.pop('chi2_no_v', results['chi2_no_v'])
    
    NSIDE = hp.npix2nside(len(results['chi2_grid']))
    values = np.zeros(len(results['chi2_grid']))
    mask = results['fit_grid'] > fit_threshold
    if len(mask.shape) > 1:
        mask = mask.T[0]
    values[mask] = chi2_no_v - results['chi2_grid'][mask]

    if attractor:
        mark = []
        neg = results['fit_grid'].T[0] < 0
        values[neg] = -values[neg]
        if void_healpy is not None:
            exvoid = results['fit_grid'].T[0] < -1
            values[exvoid] = void_healpy[exvoid] - chi2_no_v
            print chi2_no_v, results['chi2_no_v'], np.min(values), np.max(values)
    else:
        mark = [(results['fit_sph'][0][1],results['fit_sph'][0][2],'w*')]

    return healpy_fancy(values, NSIDE, mark=mark, **kwargs)

def healpy_Q(filename, **kwargs):
    results = cPickle.load(open(filename))
    
    NSIDE = hp.npix2nside(len(results['Q']))
    values = results['Q']

    mark = [
        (results['l_min'],results['b_min'],'w*'),
        (results['l_max'],results['b_max'],'k*')
    ]

    return healpy_fancy(values, NSIDE, mark=mark, **kwargs)

def plot_results_l_b(result_l,result_b,prefix,NSIDE=4,names=None,cumulative=False,
                     figsize=(8,6),save2file=None,legend='upper left',title=None,
                     z_bins=None,cmaps=None,hist=False,pixels=None,steps=4,vmin=None,
                     vmax=None,color='k',frame='galactic',marks=True,nest=_nest,
                     cbar=False,cbar_label=None,cbar_orientation='horizontal',
                     mask_zero=False,median_fct=np.median,markers=None,projection='moll'):
    """
    Overlapping hists do not work well.
    """        
    if z_bins is None:
        z_bins = _z_bins
    
    if cmaps is None:
        cmaps = [plt.get_cmap(cmap) for cmap in _cmaps]

    if hist and len(cmaps) < len(z_bins) - 1:
        raise ValueError('Require as many colormaps as z bins')
    
    if markers is None:
        markers = _markers

    if not hist and len(markers) < len(z_bins) - 1:
        raise ValueError('Require as many markers as z bins')

    fig, m  = basic_basemap(projection=projection,figsize=figsize,
                            color=color,frame=frame,marks=marks)

    for z_min,z_max,cmap,marker,l,b in zip(z_bins[:-1],z_bins[1:],cmaps,markers,
                                           result_l[prefix],result_b[prefix]):
        if cumulative:
            z_min = z_bins[0]
        if hist:
            values, pixels = healpy_hist(l,b,mask_zero=mask_zero,NSIDE=NSIDE,nest=nest)
            healpy_basemap(values,NSIDE=NSIDE,pixels=pixels,steps=steps,vmin=vmin,
                           vmax=vmax,projection=projection,cmap=cmap,cbar=cbar,
                           cbar_label=cbar_label,nest=_nest,cbar_orientation=cbar_orientation,
                           fig_m=(fig,m),alpha=0.2)
        else:
            l_median = median_fct(l)
            b_median = median_fct(b)
            x,y = m(l_median,b_median)
            zlabel = r'${:.3f}<z<{:.3f}$'.format(z_min,z_max)
            plt.plot(x,y,marker,ms=8,label=zlabel)

    if legend is not None:
        plt.legend(loc=legend)   

    return fig, m

# ----------------- #
# -- Other plots -- #
# ----------------- #

def plot_results(result,prefixes=None,names=None,cumulative=False,figsize=(8,6),save2file=None,
                 z_range=None,y_range=None,y_label=None,legend='upper left',
                 connect_w_line=True,title=None,z_bins=None,markers=None,median_fct=np.median,
                 errors=None):
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

    if z_range is None:
        if cumulative:
            z_range = (z_bins[1]-0.01,z_bins[-1]+0.01)
        else:
            z_range = (0,z_bins[-1]+0.01)

    if cumulative:
        z = z_bins[1:]
    else:
        z = [np.mean([z_min,z_max]) for z_min, z_max in zip(z_bins[:-1],z_bins[1:])]
    
    fig = plt.figure(figsize=figsize)
    for (prefix,name,marker) in zip(prefixes,names,markers):
        if errors is None:
            plt.plot(z,[median_fct(a) for a in result[prefix]],marker,ms=8,label=name)
        else:
            plt.errorbar(z,[median_fct(a) for a in result[prefix]],
                         yerr=[median_fct(a) for a in errors[prefix]],
                         fmt=marker,ms=8,label=name)
        if connect_w_line:
            plt.plot(z,[median_fct(a) for a in result[prefix]],marker[0]+'-')
        
    if not cumulative:
        for z_val in z_bins:
            plt.plot([z_val,z_val],[-1e6,1e6],'k--',scaley=False)
    
    if legend is not None:
        plt.legend(loc=legend)
        
    plt.xlim(z_range)
        
    if y_range is not None:
        plt.ylim(y_range)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if cumulative:
        plt.xlabel(r'$z_{max}$',fontsize=22)
    else:
        plt.xlabel(r'$z_{mean}$',fontsize=22)        
    
    if y_label is not None:
        plt.ylabel(y_label,fontsize=22)

    if title is not None:
        plt.title(title,fontsize=20)

    if save2file is not None:
        plt.savefig(save2file)
        
    return fig

def plot_hist(result,figsize=(8,6),save2file=None,hist_range=None,bins=50,
              xlim=None,ylim=None,y_label=None,x_label=None,title=None,normed=False):
    """
    """  
    fig = plt.figure(figsize=figsize)
    plt.hist(result,range=hist_range,bins=bins,normed=normed)
        
    if xlim is not None:
        plt.xlim(xlim)
        
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if x_label is not None:
        plt.xlabel(x_label,fontsize=22)
    
    if normed:
        plt.ylabel('pdf',fontsize=22)
    else:
        plt.ylabel('Count',fontsize=22)

    if title is not None:
        plt.title(title,fontsize=20)

    if save2file is not None:
        plt.savefig(save2file)
        
    return fig

# -------------------------------- #
# ----  FROM THE SNf ToolBox ----- #
# -------------------------------- #

def radec2gcs(ra, dec, deg=True):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Convert *(ra,dec)* equatorial coordinates (J2000, in degrees if
    *deg*) to Galactic Coordinate System coordinates *(lII,bII)* (in
    degrees if *deg*).

    Sources:

    - http://www.dur.ac.uk/physics.astrolab/py_source/conv.py_source
    - Rotation matrix from
      http://www.astro.rug.nl/software/kapteyn/celestialbackground.html

    .. Note:: This routine is only roughly accurate, probably at the
              arcsec level, and therefore not to be used for
              astrometric purposes. For most accurate conversion, use
              dedicated `kapteyn.celestial.sky2sky` routine.

    >>> radec2gal(123.456, 12.3456)
    (210.82842704243518, 23.787110745502183)
    """

    if deg:
        ra  =  ra * _d2r
        dec = dec * _d2r

    rmat = np.array([[-0.054875539396, -0.873437104728, -0.48383499177 ],
                    [ 0.494109453628, -0.444829594298,  0.7469822487  ],
                    [-0.867666135683, -0.198076389613,  0.455983794521]])
    cosd = np.cos(dec)
    v1 = np.array([np.cos(ra)*cosd,
                  np.sin(ra)*cosd,
                  np.sin(dec)])
    v2 = np.dot(rmat, v1)
    x,y,z = v2

    c,l = rec2pol(x,y)
    r,b = rec2pol(c,z)

    assert np.allclose(r,1), "Precision error"

    if deg:
        l /= _d2r
        b /= _d2r

    return l, b

def rec2pol(x,y, deg=False):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Conversion of rectangular *(x,y)* to polar *(r,theta)*
    coordinates
    """

    r = np.hypot(x,y)
    t = np.arctan2(y,x)
    if deg:
        t /= RAD2DEG

    return r,t

#-------------------#
#-- Old functions --#
#-------------------#

def healpy_basemap_old(values,NSIDE=4,pixels=None,steps=4,vmin=None,
                       vmax=None,projection='moll',figsize=(8,6),
                       cmap='Blues',color='k',frame='galactic',marks=True,
                       cbar=True,cbar_label=None,nest=_nest,alpha=1,
                       cbar_orientation='horizontal',fig_m=None):
    """
    """
    if fig_m is None:
        fig, m = basic_basemap(projection=projection,figsize=figsize,
                               color=color,frame=frame,marks=marks)
    else:
        fig, m = fig_m

    if pixels is None:
        pixels = range(hp.nside2npix(NSIDE))
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    for pix,count in zip(pixels,values):
        corners = hp.boundaries(NSIDE,pix,step=steps,nest=nest)
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
                m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha)

            l_temp = np.fmax(l_new,180.0001)
            # Check that there are points not on the edges
            num_on_edges = np.sum(((l_temp != 180.0001) 
                                 & (np.abs(b_new) != 90)).astype(int))
            if num_on_edges > 0:
                x,y = m(l_temp,b_new)
                m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha)
        else:
            x,y = m(l_new,b_new)
            m.pcolor(x,y,count_arr,vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha)

    if not cbar:
        return fig, m
    else:
        cbar = plt.colorbar(orientation=cbar_orientation, shrink=0.92, pad=0.08)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=20)

        return fig, m, cbar
