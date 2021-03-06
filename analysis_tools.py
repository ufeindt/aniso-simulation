#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis tools for anisotropy studies

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import cPickle

from scipy.optimize import leastsq, fmin
from copy import deepcopy

import cosmo_tools as ct
import velocity_tools as vt

from cosmo_tools import _d2r

_z_bins = [0.015,0.025,0.035,0.045,0.06,0.1]

l_SSC, b_SSC = 306.44, 29.71
l_CMB, b_CMB = 276.,   30. 

def fit_dipole(data,opt):
    options = deepcopy(opt)
    options['v_comp'] = [np.array(map(lambda x: vt.v_dipole_comp(x[4],x[5]),a))
                         for a in data]

    fit = leastsq(ct.vresiduals,np.zeros(3),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.vresidual_chi2(np.zeros(3),data,options)
    results['chi2'] = ct.vresidual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']
    results['n_SNe'] = np.sum([len(a) for a in data])

    results['fit'] = [fit[0]]
    results['cov'] = [fit[1]]
    for k,key in enumerate(['U_x','U_y','U_z']):
        results[key] = fit[0][k]
        results['d'+key] = np.sqrt(fit[1][k,k])

    fit_sph, cov_sph = vt.convert_spherical(fit[0],cov=fit[1])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    return results

def fit_dipole_shear(data,opt):
    options = deepcopy(opt)
    options['v_comp'] = [np.array(map(lambda x: vt.v_tidal_comp(x[1],x[4],x[5],
                                                                O_M=options['O_M'],
                                                                H_0=options['H_0']),a))
                         for a in data]

    fit = leastsq(ct.vresiduals,np.zeros(9),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.vresidual_chi2(np.zeros(9),data,options)
    results['chi2'] = ct.vresidual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']
    results['n_SNe'] = np.sum([len(a) for a in data])

    results['fit'] = [fit[0]]
    results['cov'] = [fit[1]]
    results['shear'] = [vt.reshape_shear_matrix(fit[0],offset=3)]
    
    fit_trless = vt.remove_tr(fit[0],fit[1])
    results['fit_trless'] = [fit_trless[0]]
    results['cov_trless'] = [fit_trless[1]]
    results['shear_trless'] = [vt.reshape_shear_matrix(fit_trless[0],offset=3)]
    for k,key in enumerate(['U_x','U_y','U_z','U_xx','U_yy','U_zz',
                            'U_xy','U_xz','U_yz','U']):
        results[key] = fit_trless[0][k]
        results['d'+key] = np.sqrt(fit_trless[1][k,k])

    fit_sph, cov_sph = vt.convert_spherical(fit[0][:3],cov=fit[1][:3,:3])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    # distance estimates
    d_cart = vt.get_distance_estimates(fit[0][:6],fit[1][:6,:6])
    results['d_cart'] = [d_cart[0]] 
    results['cov_d_cart'] = [d_cart[1]]

    for k,key in enumerate(['d_x','d_y','d_z']):
        results[key] = d_cart[0][k]
        results['d'+key] = np.sqrt(d_cart[1][k,k])

    d_cart_trless = vt.get_distance_estimates(fit_trless[0][:6],fit_trless[1][:6,:6])
    results['d_cart_trless'] = [d_cart_trless[0]]
    results['cov_d_cart_trless'] = [d_cart_trless[1]]

    for k,key in enumerate(['d_x','d_y','d_z']):
        results[key+'_trless'] = d_cart_trless[0][k]
        results['d'+key+'_trless'] = np.sqrt(d_cart_trless[1][k,k])

    # distance estimate in bulk flow direction
    fit_bf = vt.get_bf_shear(fit[0][:9],fit[1][:9,:9])
    results['fit_bf'] = [fit_bf[0]] 
    results['cov_bf'] = [fit_bf[1]]
    d_bf = vt.get_distance_estimates(fit_bf[0],fit_bf[1])
    results['d_bf'] = d_bf[0][0]
    results['dd_bf'] = np.sqrt(d_bf[1][0,0])
    
    fit_bf_trless = vt.get_bf_shear(fit_trless[0][:9],fit_trless[1][:9,:9])
    results['fit_bf_trless'] = [fit_bf_trless[0]] 
    results['cov_bf_trless'] = [fit_bf_trless[1]]
    d_bf_trless = vt.get_distance_estimates(fit_bf_trless[0],fit_bf_trless[1])
    results['d_bf_trless'] = d_bf_trless[0][0]
    results['dd_bf_trless'] = np.sqrt(d_bf_trless[1][0,0])

    # distance estimates according to Kaiser 1991
    dk = vt.get_distance_estimates_kaiser(fit[0],fit[1])
    results['dk'] = [dk[0]]
    results['ddk'] = [dk[1]]
    
    dk_trless = vt.get_distance_estimates_kaiser(fit_trless[0][:9],fit_trless[1][:9,:9])
    results['dk_trless'] = [dk_trless[0]]
    results['ddk_trless'] = [dk_trless[1]]

    # in eigenvector bases
    results_eig = vt.convert_to_eig_val_base(fit[0],fit[1])
    for k,key in enumerate(['fit_eig','cov_eig','d_eig',
                            'eig_vecs','eig_vec_sph','cos_bf_eig']):
        results[key] = [results_eig[k]]

    results_eig_trless = vt.convert_to_eig_val_base(fit_trless[0],fit_trless[1])
    for k,key in enumerate(['fit_eig','cov_eig','d_eig',
                            'eig_vecs','eig_vec_sph','cos_bf_eig']):
        results[key+'_trless'] = [results_eig_trless[k]]

    return results

def fit_dipole_shear_trless(data,opt):
    options = deepcopy(opt)
    options['v_comp'] = [np.array(map(lambda x: vt.v_tidal_comp_trless(x[1],x[4],x[5],
                                                                       O_M=options['O_M'],
                                                                       H_0=options['H_0']),a))
                         for a in data]

    fit = leastsq(ct.residuals,np.zeros(8),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.residual_chi2(np.zeros(8),data,options)
    results['chi2'] = ct.residual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']
    results['n_SNe'] = np.sum([len(a) for a in data])

    results['fit'] = [fit[0]]
    results['cov'] = [fit[1]]

    fit = vt.add_shear_z(fit[0],fit[1])
    results['fit_wtr'] = [fit[0]]
    results['cov_wtr'] = [fit[1]]
    results['shear'] = [vt.reshape_shear_matrix(fit[0],offset=3)]
    
    for k,key in enumerate(['U_x','U_y','U_z','U_xx','U_yy','U_zz',
                            'U_xy','U_xz','U_yz']):
        results[key] = fit[0][k]
        results['d'+key] = np.sqrt(fit[1][k,k])

    fit_sph, cov_sph = vt.convert_spherical(fit[0][:3],cov=fit[1][:3,:3])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    # distance estimates
    d_cart = vt.get_distance_estimates(fit[0][:6],fit[1][:6,:6])
    results['d_cart'] = [d_cart[0]] 
    results['cov_d_cart'] = [d_cart[1]]

    for k,key in enumerate(['d_x','d_y','d_z']):
        results[key] = d_cart[0][k]
        results['d'+key] = np.sqrt(d_cart[1][k,k])

    # distance estimate in bulk flow direction
    fit_bf = vt.get_bf_shear(fit[0][:9],fit[1][:9,:9])
    results['fit_bf'] = [fit_bf[0]] 
    results['cov_bf'] = [fit_bf[1]]
    d_bf = vt.get_distance_estimates(fit_bf[0],fit_bf[1])
    results['d_bf'] = d_bf[0][0]
    results['dd_bf'] = np.sqrt(d_bf[1][0,0])
                                                                       
    # distance estimates according to Kaiser 1991
    dk = vt.get_distance_estimates_kaiser(fit[0],fit[1])
    results['dk'] = [dk[0]] 
    results['ddk'] = [dk[1]]

    # in eigenvector bases
    results_eig = vt.convert_to_eig_val_base(fit[0],fit[1])
    for k,key in enumerate(['fit_eig','cov_eig','d_eig',
                            'eig_vecs','eig_vec_sph','cos_bf_eig']):
        results[key] = [results_eig[k]]

    return results

def get_Q_min_max(data,options,delta=90,l_grid=None,b_grid=None,full_opt=True,
                  weighted=True,return_Q_grid=False):
    """
    """
    l_res = np.array([a[4] for b in data for a in b])
    b_res = np.array([a[5] for b in data for a in b])
    res = ct.residuals([],data,options)

    if l_grid is None and b_grid is None:
        n_grid = int(180./delta)
        l_grid = np.linspace(-180,180,2*n_grid+2)[:-1]
        b_grid = np.linspace(-90,90,n_grid+2)[1:-1]
    
    Q = scan_Q(l_res,b_res,res,l_grid,b_grid,delta=delta,weighted=weighted)
    Q_max, l_max, b_max = find_max(Q,l_grid,b_grid)
    Q_min, l_min, b_min = find_min(Q,l_grid,b_grid)

    if full_opt:
        fmin_min = fmin(smoothed_res,(l_min,b_min),args=(l_res,b_res,res,delta,weighted),
                        disp=0,full_output=1)
        Q_min = fmin_min[1]
        l_min = fmin_min[0][0]
        b_min = fmin_min[0][1]

        fmin_max = fmin(smoothed_res,(l_max,b_max),args=(l_res,b_res,-res,delta,weighted),
                        disp=0,full_output=1)
        Q_max = -fmin_max[1]
        l_max = fmin_max[0][0]
        b_max = fmin_max[0][1]        

    results = {'Q_max': Q_max,
               'l_max': ((l_max -180)%360)+180,
               'b_max': b_max,
               'Q_min': Q_min,
               'l_min': ((l_min -180)%360)+180,
               'b_min': b_min,
               'dQ': Q_max - Q_min}

    results['n_SNe'] = np.sum([len(a) for a in data])

    if return_Q_grid:
        results['l_grid'] = l_grid
        results['b_grid'] = b_grid
        results['Q_grid'] = Q

    return results

# --------------------------------- #
# -- Smoothed residual functions -- #
# --------------------------------- #

def smoothed_res(lb,l_res,b_res,res,delta,weighted):
    """
    lb       -- tuple of coordinates where to evaluate SR
    l_res    -- numpy array of l values in degrees
    b_res    -- numpy array of b values in degrees
    res      -- numpy array of error-weighted Hubble residuals
    delta    -- smoothing parameter in degrees
    weighted -- boolean whether to divide by sum of weights
    """
    if lb[1] > 90 or lb[1] < -90: 
        return 1e20 

    ang_seps = vt.ang_sep(lb[0],lb[1],l_res,b_res)
    weights = np.exp(-ang_seps**2/(2*delta**2)) / np.sqrt(2*_d2r**2*delta**2)

    Q = np.sum(res*weights) 
    if weighted:
        Q /= np.sum(weights) 

    return Q

def scan_Q(l_res,b_res,res,l_grid,b_grid,delta=90,weighted=True):
    out = []
    for l, b in zip(l_grid,b_grid):
        out.append(smoothed_res((l,b),l_res,b_res,res,delta,weighted))

    return np.array(out)

def find_max(Q,l_grid,b_grid):
    Q_max = Q.max()
    l_max = l_grid[np.where(Q == Q_max)]
    b_max = b_grid[np.where(Q == Q_max)]

    if (len(b_max[(b_max > -90) & (b_max < 90)]) > 1 and
        len(l_max[(l_max > -180) & (l_max < 180)]) > 1):
        print 'Warning: Maximum found at multiple locations' 
        print 'l:', l_max
        print 'b:', b_max

    return Q_max, l_max[0], b_max[0]

def find_min(Q,l_grid,b_grid):
    Q_min = Q.min()
    l_min = l_grid[np.where(Q == Q_min)]
    b_min = b_grid[np.where(Q == Q_min)]

    if (len(b_min[(b_min > -90) & (b_min < 90)]) > 1 and
        len(l_min[(l_min > -180) & (l_min < 180)]) > 1):
        print 'Warning: Minimum found at multiple locations'
        print 'l:', l_min
        print 'b:', b_min

    return Q_min, l_min[0], b_min[0]

# -------------- #
# -- p-values -- #
# -------------- #

def p_value(signal,noise):
    """
    """
    out = []

    if type(signal) in [float,np.float64]:
        signal = np.array([signal])
    
    for sig in signal:
        p = float(len(noise[noise > sig]))/len(noise)
        out.append(p)

    if len(out) > 1:
        return np.array(out)
    else:
        return out[0]

# ----------------------- #
# -- Loading functions -- #
# ----------------------- #

def load_result_file(filename):
    """
    """
    #print filename
    return cPickle.load(file(filename,'r'))

def load_results(key,skeys,prefixes,signal_mode=1,cumulative=False,
                 z_bins=None,resultdir='results/'):
    """
    """
    if type(skeys) in [list,tuple]:
        out = [{prefix: [] for prefix in prefixes} for skey in skeys]
    elif type(skeys) == str:
        skeys = [skeys]
        out = [{prefix: [] for prefix in prefixes}]
    
    if z_bins is None:
        z_bins = _z_bins

    if cumulative:
        z_mins = [z_bins[0] for z in z_bins[:-1]]
    else:
        z_mins = z_bins[:-1]
    z_maxs = z_bins[1:]
    
    for prefix in prefixes:
        for z_min, z_max in zip(z_mins,z_maxs):
            filestart = '{}{}_s{}_{:.3f}_{:.3f}'.format(
                resultdir,prefix,signal_mode,z_min,z_max
            ).replace('.','')
            
            results = load_result_file('{}.pkl'.format(filestart))
            for k,skey in enumerate(skeys):
                #print key, skey
                out[k][prefix].append(results[key][skey])
            
    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]

def function_of_results(f,*results):
    """
    Apply function to results previously loaded
    """
    out = {}

    for key in results[0].keys():
        out[key] = []
        for result_list in zip(*[a[key] for a in results]):
            out[key].append(f(*result_list))

    return out


def load_p_values(key,skey,prefixes,signal_mode=1,cumulative=False,
                z_bins=None,resultdir='results/'):
    """
    """
    p = {}
    signal = load_results(key,skey,prefixes,z_bins=z_bins,resultdir=resultdir,
                          cumulative=cumulative,signal_mode=signal_mode)
    noise  = load_results(key,skey,prefixes,z_bins=z_bins,resultdir=resultdir,
                          cumulative=cumulative,signal_mode=0)

    for key in signal.keys():
        p[key] = []
        for slist,nlist in zip(signal[key],noise[key]):
            p[key].append(p_value(slist,nlist))

    return p

#-----------#
#-- Other --#
#-----------#
def nmad(vals, m=None, factor=1.4826):
    if m is None:
        m = np.median(vals)
    return np.median(np.abs(vals-m)) * factor

def ang_sep_err(l1,b1,l2,b2,cov1,cov2):
    cov_tmp1 = np.zeros((3,3))
    cov_tmp1[1:,1:] = cov1             
    v1, cov_v1 = convert_cartesian([1,l1,b1],cov_tmp1)
    
    cov_tmp2 = np.zeros((3,3))
    cov_tmp2[1:,1:] = cov2    
    v2, cov_v2 = convert_cartesian([1,l2,b2],cov_tmp2)
                                                      
    cos_as = v1.dot(v2)                               
    cov_cos_as = v1.dot(cov_v2).dot(v1) + v2.dot(cov_v1).dot(v2) 
                                                                 
    ang_sep = np.arccos(cos_as) / dp._d2r                        
    err_ang_sep = np.sqrt(cov_cos_as/(1 - cos_as ** 2))  / dp._d2r
                                                                  
    return ang_sep, err_ang_sep


def wpercentile(a, q, weights=None):
   """Compute weighted percentiles *q* [%] of input 1D-array *a*."""

   a = np.asarray(a)
   if a.ndim > 1:
       raise NotImplementedError("implemented on 1D-arrays only")

   if weights is None:
       weights = np.ones_like(a)
   else:
       assert len(weights)==len(a), "incompatible weight and input arrays"
       assert (weights>0).all(), "weights are not always strictly positive"

   isorted = np.argsort(a)
   sa = a[isorted]
   sw = weights[isorted]
   sumw = np.cumsum(sw)                        # Strictly increasing
   scores = 1e2*(sumw - 0.5*sw)/sumw[-1]      # 0-100 score at center of bins

   def interpolate(q):
       i = scores.searchsorted(q)
       if i==0:                        # Below 1st score
           val = sa[0]
       elif i==len(a):                 # Above last score
           val = sa[-1]
       else:                           # Linear score interpolation
           val = (sa[i-1]*(scores[i] - q) + sa[i]*(q - scores[i-1])) / \
                 (scores[i] - scores[i-1])
       return val

   out = np.array([ interpolate(qq) for qq in np.atleast_1d(q) ])

   return out.reshape(np.shape(q))      # Same shape as input q


def median_stats(a, weights=None, axis=None, scale=1.4826, corrected=True):
   """Compute [weighted] median and :func:`nMAD` of array *a* along
   *axis*. Weighted computation is implemented for *axis* = None
   only."""

   if weights is not None:
       if axis is not None:
           raise NotImplementedError("implemented on 1D-arrays only")
       else:
           med  = wpercentile(a, 50., weights=weights)
           nmad = wpercentile(np.absolute(a - med), 50., weights=weights)*scale
   else:
       med = np.median(a, axis=axis)
       if axis is None:
           umed = med                      # Scalar
       else:
           umed = np.expand_dims(med, axis) # Same ndim as a
       nmad = np.median(np.absolute(a - umed), axis=axis) * scale

   if corrected:
       # Finite-sample correction on nMAD (Croux & Rousseeuw, 1992)
       if axis is None:
           n = np.size(a)
       else:
           n = np.shape(a)[axis]
       if n<=9:
           c = [0,0,1.196,1.495,1.363,1.206,1.200,1.140,1.129,1.107][n]
       else:
           c = n/(n-0.8)
       nmad *= c

   return med,nmad
