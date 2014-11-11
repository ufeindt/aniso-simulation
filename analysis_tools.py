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

from scipy.optimize import leastsq, fmin

import cosmology_tools as ct
import velocity_tools as vt

def fit_dipole(data,options):
    options['v_comp'] = [np.array(map(lambda x: vt.v_dipole_comp(x[4],x[5]),a))
                         for a in data]

    fit = leastsq(ct.residuals,np.zeros(3),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.residual_chi2(np.zero(3),data,options)
    results['chi2'] = ct.residual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']

    results['fit'] = [fit[0]]
    results['cov'] = [fit[1]]
    for k,key in enumerate(['U_x','U_y','U_z']):
        results[key] = fit[0][k]
        results['d'+key] = np.sqrt(fit[1][k,k])

    fit_sph, cov_sph = vt.convert_spherical(fit[0],cov=fit[1]])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[0][k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    return results

def fit_dipole_shear(data,options):
    options['v_comp'] = [np.array(map(lambda x: vt.v_tidal_comp(x[4],x[5],
                                                                O_M=options['O_M'],
                                                                H_0=options['H_0']),a))
                         for a in data]

    fit = leastsq(ct.residuals,np.zeros(9),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.residual_chi2(np.zero(9),data,options)
    results['chi2'] = ct.residual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']

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

    fit_sph, cov_sph = vt.convert_spherical(fit[0][:3],cov=fit[1][:3,:3]])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[0][k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    # distance estimates
    d_cart = vt.get_distance_estimates(fit[0][:6],fit][1][:6,:6])
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
    fit_bf = get_bf_shear(fit[0][:9],fit[1][:9,:9])
    results['fit_bf'] = [fit_bf[0]] 
    results['cov_bf'] = [fit_bf[1]]
    d_bf = get_distance_estimates(fit_bf[0],fit_bf[1])
    results['d_bf'] = d_bf[0][0]
    results['dd_bf'] = np.sqrt(d_bf[1][0,0])
    
    fit_bf_trless = get_bf_shear(fit_trless[0][:9],fit_trless[1][:9,:9])
    results['fit_bf_trless'] = [fit_bf_trless[0]] 
    results['cov_bf_trless'] = [fit_bf_trless[1]]
    d_bf_trless = get_distance_estimates(fit_bf_trless[0],fit_bf_trless[1])
    results['d_bf_trless'] = d_bf_trless[0][0]
    results['dd_bf_trless'] = np.sqrt(d_bf_trless[1][0,0])

    # distance estimates according to Kaiser 1991
    dk = get_distance_estimates_kaiser(fit[0],fit[1])
    results['dk'] = [dk[0]]
    results['ddk'] = [dk[1]]
    
    dk_trless = get_distance_estimates_kaiser(fit_trless[0],fit_trless[1])
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
        results[key+'rless'] = [results_eig_trless[k]]

    return results

def fit_dipole_shear_trless(data,options):
    options['v_comp'] = [np.array(map(lambda x: vt.v_tidal_comp_trless(x[4],x[5],
                                                                       O_M=options['O_M'],
                                                                       H_0=options['H_0']),a))
                         for a in data]

    fit = leastsq(ct.residuals,np.zeros(8),args=(data,options),full_output=1)

    results = {}
    
    results['chi2_0'] = ct.residual_chi2(np.zero(8),data,options)
    results['chi2'] = ct.residual_chi2(fit[0],data,options)
    results['dchi2'] = results['chi2_0'] - results['chi2']

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

    fit_sph, cov_sph = vt.convert_spherical(fit[0][:3],cov=fit[1][:3,:3]])
    results['fit_sph'] = [fit_sph]
    results['cov_sph'] = [cov_sph]
    for k,key in enumerate(['v','l','b']):
        results[key] = fit_sph[0][k]
        results['d'+key] = np.sqrt(cov_sph[k,k])

    # distance estimates
    d_cart = vt.get_distance_estimates(fit[0][:6],fit][1][:6,:6])
    results['d_cart'] = [d_cart[0]] 
    results['cov_d_cart'] = [d_cart[1]]

    for k,key in enumerate(['d_x','d_y','d_z']):
        results[key] = d_cart[0][k]
        results['d'+key] = np.sqrt(d_cart[1][k,k])

    # distance estimate in bulk flow direction
    fit_bf = get_bf_shear(fit[0][:9],fit[1][:9,:9])
    results['fit_bf'] = [fit_bf[0]] 
    results['cov_bf'] = [fit_bf[1]]
    d_bf = get_distance_estimates(fit_bf[0],fit_bf[1])
    results['d_bf'] = d_bf[0][0]
    results['dd_bf'] = np.sqrt(d_bf[1][0,0])
                                                                       
    # distance estimates according to Kaiser 1991
    dk = get_distance_estimates_kaiser(fit[0],fit[1])
    results['dk'] = [dk[0]] 
    results['ddk'] = [dk[1]]

    # in eigenvector bases
    results_eig = vt.convert_to_eig_val_base(fit[0],fit[1])
    for k,key in enumerate(['fit_eig','cov_eig','d_eig',
                            'eig_vecs','eig_vec_sph','cos_bf_eig']):
        results[key] = [results_eig[k]]

    return results

