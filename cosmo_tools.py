#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cosmology tools

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de; unless noted otherwise)
"""

import numpy as np
import warnings
from copy import deepcopy

from scipy.integrate import romberg
from scipy.optimize import leastsq

# --------------- #
# -- Constants -- #
# --------------- #

_c = 299792.458     # speed of light in km s^-1
_H_0 = 70.            # Hubble constant in km s^-1 Mpc^-1
_d2r = np.pi/180    # conversion factor from degrees to radians
_O_M = 0.3          # default matter density
_v_mode = 'hui'

# --------------------- #
# -- Basic functions -- #
# --------------------- #
def d_l(z,O_M=_O_M,O_L=None,w=-1,H_0=_H_0,v_dip=None,v_cart=None,v_mon=0,
        coords=None,v_mode=_v_mode,**kwargs):
    """
    Luminosity distance in Mpc

    Arguments:
    z -- redshift (observed, i.e. velocities are subtracted)

    Keyword arguments:
    O_M    -- matter density
    O_L    -- Dark Energy density (will be set to (1 - O_M) if None)
    w      -- Dark Energy equation of state parameter (for constant EOS)
    H_0    -- Hubble constant in km s^-1 Mpc^-1
    v_dip  -- dipole velocity in km s^-1 as an interable 
              (e.g. list or np.array) in spherical coordinates 
              (angles in degrees), i.e. v_dip = [abs(v), RA, Dec] or 
              [abs(v), l, b]; ignored if coords is empty
    v_cart -- dipole velocity in km s^-1 in Cartesian coordinates
              If both v_cart and v_dip are not empty, v_dip will be used
              ignored if coords is empty
    v_mon  -- monopole velocity in km s^-1
    coords -- coordinates of the SN, e.g. [RA, Dec] or [l,b]
              system needs to match that used for v_dip
    
    v_mode -- hui, bonvin or none
    memo   -- dict for memoization
    """
    if v_mode is None:
        v_mode = 'none'

    if O_L is None:
        Flat = True
        O_L = 1 - O_M
        O_K = 0
    else:
        Flat = False
        O_K = 1 - O_M - O_L

    H_rec = lambda x: 1 / np.sqrt(O_M * (1+x)** 3 +
                                  O_K * (1+x) ** 2 +
                                  O_L * (1+x)**(3*(1+w)))

    if coords or v_mon != 0:
        if not Flat:
            print 'Warning: dipole not properly implemented for',
            print 'non-flat cosmology'
        if v_dip is not None:
            cos_theta = (np.cos(coords[1]*_d2r) * np.cos(v_dip[2]*_d2r) *
                        np.cos((coords[0]-v_dip[1])*_d2r) +
                        np.sin(coords[1]*_d2r) * np.sin(v_dip[2]*_d2r))
            v_proj = v_mon + v_dip[0]*cos_theta
        elif v_cart is not None:
            l = coords[0]
            b = coords[1]
            v_proj =  (v_mon + v_cart[0]*np.cos(b*_d2r)*np.cos(l*_d2r) +
                       v_cart[1]*np.cos(b*_d2r)*np.sin(l*_d2r)+ 
                       v_cart[2]*np.sin(b*_d2r))
        else:
            v_proj = v_mon
    else:
        v_proj = 0
        
    if v_mode == 'none':
        z = (1 + z) / (1 + v_proj/_c) - 1 # see Harrison 1974

    integral = romberg(H_rec,0,z)

    if O_K == 0:
        result = _c * (1+z) / H_0 * integral
    elif O_K < 0:
        result = (_c * (1+z) / H_0 / np.sqrt(-O_K) *
                  np.sin(np.sqrt(-O_K)*integral))
    else:
        result = (_c * (1+z) / H_0 / np.sqrt(O_K) *
                  np.sinh(np.sqrt(O_K)*integral))
            
    if v_mode == 'bonvin':
        result = result - (1+z)**2 * H_rec(z) / H_0 * v_proj
    elif v_mode == 'hui':
        result = result * (1 + v_proj / _c) - (1+z)**2 * H_rec(z) / H_0 * v_proj
    elif v_mode != 'none':
        raise ValueError('Unknown v_mode: {}'.format(v_mode))

    return result

def d_p(z,**kwargs):
    """
    Proper distance

    Arguments as for d_l() 
    """
    d = d_l(z,**kwargs)
    result = d/(1+z)
    return result

def d_a(z,**kwargs):
    """
    Angular diameter distance

    Arguments as for d_l() 
    """
    d = d_l(z,**kwargs)
    result = d/(1+z)**2
    return result


def mu(z,**kwargs):
    """
    Distance modulus

    Arguments as for d_l() 
    """
    d = d_l(z,**kwargs)
    result = 5*np.log10(d)+25
    return result

# ----------------------------- #
# -- chi^2-related fucntions -- #
# ----------------------------- #

def residuals(p,data,options):
    """
    Main residual function for fitting
    """
    # List of fit parameter names:
    opt = deepcopy(options)
    para_names = ['O_M','O_L','w','H_0','dM']
    k = 0
    for name in para_names:
        if name not in opt.keys():
            opt[name] = p[k]
            k += 1
    v_comp_vals = []
    if 'offsets' not in opt.keys():
        opt['offsets'] = p[k:k+len(data)]
        k += len(data)
    if 'sig_int' not in opt.keys():
        opt['sig_int'] = [0 for x in data]
    if 'v_comp' in opt.keys():
        for v_comps in opt['v_comp']:
            v_comp_vals += [np.dot(v_comps,p[k:])]
    else:
        v_comp_vals += [np.zeros(len(x)) for x in data]
    if 'v_pec' not in opt.keys():
        opt['v_pec'] = [np.zeros(len(x)) for x in data]
    if 'weights' not in opt.keys():
        opt['weights'] = [np.ones(len(x)) for x in data]
    out=[]
    for x1,off,v_comp,v_pec,sig_int,ws in zip(data,opt['offsets'],v_comp_vals,
                                              opt['v_pec'],opt['sig_int'],
                                              opt['weights']):
        for y,v_comp_val,v_pec_val,w in zip(x1,v_comp,v_pec,ws):
            mu_full=mu(y[1],v_mon=v_comp_val+v_pec_val,**opt)
            out+=[(y[2] + off + opt['dM'] - mu_full) / 
                  np.sqrt(y[3]**2 + sig_int**2) * w]
    out = np.array(out)
    if len(out.shape) == 1: return out
    else: return out.transpose()[0]

def residual_chi2(p,data,options,dof=None):
    a = residuals(p,data,options)
    if dof is None:
        return np.dot(a,a)
    else:
        return np.dot(a,a)/dof

def vresiduals(p,data,options):
    """
    Residual function for fitting when d_l is known (i.e. cosmo fit beforehand)
    Syntax similar to residuals but data must also include luminosity distance; 
    O_M, O_L, w, H_0 and dM must all be in options;
    Can be used to determine offset
    """
    opt = deepcopy(options)
    
    zs = [np.array([a[1] for a in b]) for b in data]
    mus = [np.array([a[2] for a in b]) for b in data]
    dmus = [np.array([a[3] for a in b]) for b in data]
    d_ls = [np.array([a[6] for a in b]) for b in data]

    if 'offsets' not in opt.keys():
        opt['offsets'] = p[:len(zs)]
        k = len(zs)
    else:
        k = 0

    if opt['O_L'] is None:
        opt['O_L'] = 1 - opt['O_M']

    if 'v_mode' not in opt.keys():
        opt['v_mode'] = _v_mode

    if 'sig_int' not in opt.keys():
        opt['sig_int'] = [0 for x in zs]   

    v_comp_vals = []
    if 'v_comp' in opt.keys():
        for v_comps in opt['v_comp']:
            v_comp_vals += [np.dot(v_comps,p[k:])]
    else:
        v_comp_vals += [np.zeros(len(x)) for x in zs]
    if 'v_pec' not in opt.keys():
        opt['v_pec'] = [np.zeros(len(x)) for x in zs]
    if 'weights' not in opt.keys():
        opt['weights'] = [np.ones(len(x)) for x in zs]   

    H_rec = lambda x: 1 / np.sqrt(opt['O_M'] * (1+x)** 3 +
                                  (1 - opt['O_M'] - opt['O_L']) * (1+x) ** 2 +
                                  opt['O_L'] * (1+x)**(3*(1+opt['w'])))

    d_ls_v = []
    if opt['v_mode'] == 'none':
        raise ValueError("v_mode 'none' not supported; use residuals instead")
    elif opt['v_mode'] == 'bonvin':
        for z, d_l, v_comp_val, v_pec in zip(zs, d_ls, v_comp_vals, opt['v_pec']):
            d_ls_v.append(d_l - (1+z)**2 * H_rec(z) / opt['H_0'] * (v_comp_val+v_pec))
    elif opt['v_mode'] == 'hui':
        for z, d_l, v_comp_val, v_pec in zip(zs, d_ls, v_comp_vals, opt['v_pec']):
            v_proj = v_comp_val + v_pec
            d_ls_v.append(d_l * (1 + v_proj / _c) - (1+z)**2 * H_rec(z) / opt['H_0'] * v_proj)
    else:
        raise ValueError('Unknown v_mode: {}'.format(opt['v_mode']))

    out=np.array([])
    for d_l,mu,dmu,off,sig_int,ws in zip(d_ls_v, mus, dmus, opt['offsets'],
                                           opt['sig_int'], opt['weights']):
        out = np.append(out, ((mu + off + opt['dM'] - 5 * np.log10(d_l) - 25) /
                              np.sqrt(dmu**2 + sig_int**2) * ws))
    
    out = np.array(out)
    if len(out.shape) == 1: return out
    else: return out.transpose()[0]

def vresidual_chi2(p,data,options,dof=None):
    a = vresiduals(p,data,options)
    if dof is None:
        return np.dot(a,a)
    else:
        return np.dot(a,a)/dof

def fit_w_sig_int(initial, data, options, sig_int_step=0.1, tol=1e-3, fast=False):
    """
    fast won't work unless cosmo fit already
    data must include luminosity distances
    """
    if fast:
        fit_fct = vresiduals
        chi2_fct = vresidual_chi2
    else:
        fit_fct = residuals
        chi2_fct = residual_chi2

    dof = sum([len(a) for a in data]) - len(initial)
    sig_int = 0.
    options['sig_int'] = [sig_int for x in data]
    chi2 = 2

    # Check whether reduced chi^2 already < 1
    fit = leastsq(fit_fct,initial,args=(data,options))
    chi2 = chi2_fct(fit[0],data,options,dof=dof)
    initial = fit[0]
    
    #print chi2, residual_chi2(fit[0],data,options)
    if chi2 < 1:
        #warnings.simplefilter('always', UserWarning)
        warnings.warn('reduced chi^2 < 1 for sig_int = 0.')
        return fit, options

    # First fits to get the range
    while chi2 > 1:
        sig_int += sig_int_step
        options['sig_int'] = [sig_int for x in data]
        fit = leastsq(fit_fct,initial,args=(data,options))
        chi2 = chi2_fct(fit[0],data,options,dof=dof)
        initial = fit[0]
        #print sig_int, chi2

    sig_range = [sig_int-sig_int_step,sig_int]
    #chi2 = chi2[-2:]
    sig_mid = np.mean(sig_range)
    options['sig_int'] = [sig_mid for x in data]
    fit = leastsq(fit_fct,fit[0],args=(data,options))
    chi2 = chi2_fct(fit[0],data,options,dof=dof)

    while np.abs(chi2 - 1) > tol:
        #print sig_range
        if chi2 > 1:
            sig_range[0] = sig_mid
        else:
            sig_range[1] = sig_mid
        sig_mid = np.mean(sig_range)
        options['sig_int'] = [sig_mid for x in data]
        fit = leastsq(fit_fct,fit[0],args=(data,options))
        chi2 = chi2_fct(fit[0],data,options,dof=dof)
            
    #print chi2, residual_chi2(fit[0],data,options)

    return fit, options

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
