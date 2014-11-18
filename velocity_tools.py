#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Velocity and coordinate tools

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import cosmo_tools as ct
import cPickle

from copy import deepcopy

from cosmo_tools import _O_M, _h, _d2r, _c

# ------------------------- #
# -- Velocity components -- #
# ------------------------- #

def ang_sep(l1,b1,l2,b2):
    """
    Angular separation between two positions on the sky 
    (l1,b1) and (l2,b2) in degrees.
    """
    cos_theta = (np.cos(b1 * _d2r) * np.cos(b2 * _d2r) *
                 np.cos((l1 - l2) * _d2r) +
                 np.sin(b1 * _d2r) * np.sin(b2 * _d2r))
    return np.arccos(cos_theta) / _d2r

def velocity(z,l,b,z_A,l_A,b_A,R_A,delta,O_M=_O_M,H_0=_h):
    """
    v_pec for a constant spherical overdensity projected to line of sight
    
    Arguments:
    (z,l,b)       -- Spherical redshift-space coordinates for the SN 
    (z_A,l_A,b_A) -- Spherical redshift-space coordinates of the attractor
    R_A           -- Attractor radius in Mpc
    delta         -- Overdensity

    Keyword arguments:
    O_M -- matter density parameter
    H_0 -- Hubble constant in km s^-1 Mpc^-1
    """
    d = ct.d_l(z,O_M=O_M,H_0=H_0) / (1+z) #proper distance
    d_A = ct.d_l(z_A,O_M=O_M,H_0=H_0) / (1+z_A)
    dist = d_sph(d,l,b,d_A,l_A,b_A)
    
    out = O_M**.55 * H_0 * delta / (3 * (1+z) * dist**2)
    if dist > R_A: 
        out *= R_A**3
    else: 
        out *= dist**3
        
    # vec_components = np.array([np.cos(b) * np.cos(l),
    #                            np.cos(b) * np.sin(l),
    #                            np.sin(l)])
    
    vec_components = (convert_cartesian([d_A, l_A, b_A]) 
                      - convert_cartesian([d, l, b]))
    vec_components /= dist
    
    return out * vec_components

def load_superclusters():
    sgw_factor = 46.11

    superclusters_raw = [('184+003+007',230.3,14.16,56.9),
                         ('173+014+008',242.0,12.31,50.3),
                         ('202-001+008',255.6,12.92,107.8),
                         ('152-000+009',285.1,9.81,39.0),
                         ('187+008+008',267.4,9.33,54.2),
                         ('170+000+010',302.1,8.55,20.1),
                         ('175+005+009',291.0,8.23,28.0),
                         ('159+004+006',206.6,7.61,15.4),
                         ('168+002+007',227.7,7.49,28.1),
                         ('214+001+005',162.6,7.22,19.5),
                         ('189+003+008',254.1,6.75,32.2),
                         ('198+007+009',276.0,6.33,13.1),
                         ('157+003+007',219.1,5.28,13.3)]

    superclusters = []
    sum_dense = np.sum([dense for RADec, dist, dense, diam in superclusters_raw])
    for RADec, dist, dense, diam in superclusters_raw:
        RA = float(RADec[:3])
        Dec = float(RADec[3:7])
        superclusters += [ct.radec2gcs(RA,Dec)+(dist/3e3,dense/sum_dense,diam/2)]

    return superclusters

def v_attractor(z,l,b,od_factors,O_M=_O_M,H_0=_h):
    '''
    velocity field for SSC + SGW
    '''
    superclusters = load_superclusters()

    z_A1, l_A1, b_A1, R_A1 = 0.046, 306.4, 29.7, 50
    v = velocity(z,l,b,z_A1,l_A1,b_A1,R_A1,od_factors[0],O_M=O_M,H_0=H_0)

    for l_A, b_A, z_A, delta_A, R_A in superclusters:
        v += velocity(z,l,b,z_A,l_A,b_A,R_A,delta_A*od_factors[1],O_M=O_M,
                      H_0=H_0)

    return v

def v_dipole_comp(l,b):
    """
    Dipole components

    Arguments:
    l,b -- angular coordinates in degrees 
    """
    out = (np.cos(b*_d2r) * np.cos(l*_d2r),
           np.cos(b*_d2r) * np.sin(l*_d2r),
           np.sin(b*_d2r))
    return out

def v_tidal_comp(z,l,b,O_M=_O_M,H_0=_h):
    '''
    returns the components of the tidal velocity model in the following order:
    U_x, U_y, U_z, U_xx, U_yy, U_zz, U_xy, U_xz, U_yz
    '''
    v_bulk = convert_cartesian([1,l,b])
    
    d = ct.d_l(z,O_M=O_M,H_0=H_0) / (1+z) #proper distance
    r = convert_cartesian([d,l,b]) #positonal vector
    v_diag = r**2 / d
    
    v_offdiag = 2 / d * np.array([r[0]*r[1],r[0]*r[2],r[1]*r[2]])
    
    return np.concatenate([v_bulk,v_diag,v_offdiag])

def v_tidal_comp_trless(z,l,b,O_M=_O_M,H_0=_h):
    '''
    returns the components of the tidal velocity model 
    without monopole in the following order:
    U_x, U_y, U_z, U_xx, U_yy, U_xy, U_xz, U_yz
    (U_zz = -(U_xx + U_yy))
    '''
    v_bulk = convert_cartesian([1,l,b])
    
    d = ct.d_l(z,O_M=O_M,H_0=H_0) / (1+z) #proper distance
    r = convert_cartesian([d,l,b]) #positonal vector
    v_diag = (r[:2]**2 - r[2]**2) / d
    
    v_offdiag = 2 / d * np.array([r[0]*r[1],r[0]*r[2],r[1]*r[2]])
    
    return np.concatenate([v_bulk,v_diag,v_offdiag])

def d_sph(r1,l1,b1,r2,l2,b2):
    """
    Distance of two vectors in spherical coordinates in Euclidian (!) space
    e.g. separation of two SNe or distance between SN and an attractor

    Arguments:
    r1, l1, b1 -- spherical coordinates of to vectors, to determine r1 and r2
    r2, l2, b2    use proper distance, i.e.  d_l(z) / (1 + z)
    """
    return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 *
                   (np.cos(b1 * _d2r) * np.cos(b2 * _d2r) *
                    np.cos((l1 - l2) * _d2r) +
                    np.sin(b1 * _d2r)*np.sin(b2 * _d2r)))

# --------------------- #
# -- Transformations -- #
# --------------------- #

def get_eigval_transform(eig_vecs):
    """
    Returns linear transformation matrix to get bulk flow in eigenvector
    directions and eigenvalues from shear matrix. Can be used to estimate
    covariance of those quantities as well.
    
    eig_vecs - list of three eigenvectors
    """
    return np.array([
        [eig_vecs[0][0],
         eig_vecs[0][1],
         eig_vecs[0][2],
         0,0,0,0,0,0],
        [eig_vecs[1][0],
         eig_vecs[1][1],
         eig_vecs[1][2],
         0,0,0,0,0,0],
        [eig_vecs[2][0],
         eig_vecs[2][1],
         eig_vecs[2][2],
         0,0,0,0,0,0],
        [0,0,0,
         eig_vecs[0][0]**2,
         eig_vecs[0][1]**2,
         eig_vecs[0][2]**2,
         2*eig_vecs[0][0]*eig_vecs[0][1],
         2*eig_vecs[0][0]*eig_vecs[0][2],
         2*eig_vecs[0][1]*eig_vecs[0][2]],
        [0,0,0,
         eig_vecs[1][0]**2,
         eig_vecs[1][1]**2,
         eig_vecs[1][2]**2,
         2*eig_vecs[1][0]*eig_vecs[1][1],
         2*eig_vecs[1][0]*eig_vecs[1][2],
         2*eig_vecs[1][1]*eig_vecs[1][2]],
        [0,0,0,
         eig_vecs[2][0]**2,
         eig_vecs[2][1]**2,
         eig_vecs[2][2]**2,
         2*eig_vecs[2][0]*eig_vecs[2][1],
         2*eig_vecs[2][0]*eig_vecs[2][2],
         2*eig_vecs[2][1]*eig_vecs[2][2]],
        [0,0,0,
         eig_vecs[0][0]*eig_vecs[1][0],
         eig_vecs[0][1]*eig_vecs[1][1],
         eig_vecs[0][2]*eig_vecs[1][2],
         eig_vecs[0][0]*eig_vecs[1][1] + eig_vecs[0][1]*eig_vecs[1][0],
         eig_vecs[0][0]*eig_vecs[1][2] + eig_vecs[0][2]*eig_vecs[1][0],
         eig_vecs[0][1]*eig_vecs[1][2] + eig_vecs[0][2]*eig_vecs[1][1]],
        [0,0,0,
         eig_vecs[0][0]*eig_vecs[2][0],
         eig_vecs[0][1]*eig_vecs[2][1],
         eig_vecs[0][2]*eig_vecs[2][2],
         eig_vecs[0][0]*eig_vecs[2][1] + eig_vecs[0][1]*eig_vecs[2][0],
         eig_vecs[0][0]*eig_vecs[2][2] + eig_vecs[0][2]*eig_vecs[2][0],
         eig_vecs[0][1]*eig_vecs[2][2] + eig_vecs[0][2]*eig_vecs[2][1]],
        [0,0,0,
         eig_vecs[1][0]*eig_vecs[2][0],
         eig_vecs[1][1]*eig_vecs[2][1],
         eig_vecs[1][2]*eig_vecs[2][2],
         eig_vecs[1][0]*eig_vecs[2][1] + eig_vecs[1][1]*eig_vecs[2][0],
         eig_vecs[1][0]*eig_vecs[2][2] + eig_vecs[1][2]*eig_vecs[2][0],
         eig_vecs[1][1]*eig_vecs[2][2] + eig_vecs[1][2]*eig_vecs[2][1]]])

def convert_to_eig_val_base(fit,cov):
    # In eigenvalue base
    tmp_eigvals, tmp_eigvecs =  np.linalg.eig(reshape_shear_matrix(fit,offset=3))
    idx = np.argsort(tmp_eigvals)[::-1]
    eig_vals = tmp_eigvals[idx]
    eig_vecs = [tmp_eigvecs[:,idx[l]] for l in xrange(3)]

    # Check if angle between Eigenvector direction and BF acute
    for k in xrange(3):
        if fit[:3].dot(eig_vecs[k]) < 0:
            eig_vecs[k] *= -1

    M = get_eigval_transform(eig_vecs)
    fit_eig = M.dot(fit[:9])
    cov_eig = M.dot(cov[:9,:9].dot(M.T))
        
    d_eig = get_distance_estimates(fit_eig[:6],cov_eig[:6,:6])
    eig_vec_sph = [convert_spherical(eig_vecs[k]) for k in xrange(3)]
    cos_bf_eig = [(fit[:3].dot(eig_vecs[k]) / np.sqrt(fit[:3].dot(fit[:3]))) for k in xrange(3)]

    return fit_eig, cov_eig, d_eig, eig_vecs, eig_vec_sph, cos_bf_eig

def get_bf_shear(fit,cov,mono=False):
    """
    Get shear in bulk flow direction
    returns bulk flow amplitude and their covariance as well

    fit -- numpy array of shape (9,)
           first 3 entries are bulkflow
           last 6 are shear matrix, first diagonal terms, then off-diag
    cov -- numpy array of shape (9,9), covariance matrix of fit
    """   
    U = np.sqrt(np.sum(fit[:3]**2))
    C = np.sum(fit[3:6]*fit[:3]**2) + 2 * (fit[0]*fit[1]*fit[6] +
                                           fit[0]*fit[2]*fit[7] +
                                           fit[1]*fit[2]*fit[8])
    
    if mono:
        out = np.array([U,C/U**2,fit[9]])
        # Jacobian 
        J = np.array([
            [
                fit[0]/U,fit[1]/U,fit[2]/U,0,0,0,0,0,0,0
            ],
            [
                2*((fit[0]*fit[3]+fit[6]*fit[1]+fit[7]*fit[2])/U**2 -C*fit[0]/U**4),
                2*((fit[1]*fit[4]+fit[6]*fit[0]+fit[8]*fit[2])/U**2 -C*fit[1]/U**4),
                2*((fit[2]*fit[5]+fit[7]*fit[0]+fit[8]*fit[1])/U**2 -C*fit[2]/U**4),
                fit[0]**2/U**2,fit[1]**2/U**2,fit[2]**2/U**2,
                2*fit[0]*fit[1]/U**2,2*fit[0]*fit[2]/U**2,2*fit[1]*fit[2]/U**2,
                0
            ],
            [
                0,0,0,0,0,0,0,0,0,1
            ]
        ])
    else:
        out = np.array([U,C/U**2])
        # Jacobian 
        J = np.array([
            [
                fit[0]/U,fit[1]/U,fit[2]/U,0,0,0,0,0,0
            ],
            [
                2*((fit[0]*fit[3]+fit[6]*fit[1]+fit[7]*fit[2])/U**2 -C*fit[0]/U**4),
                2*((fit[1]*fit[4]+fit[6]*fit[0]+fit[8]*fit[2])/U**2 -C*fit[1]/U**4),
                2*((fit[2]*fit[5]+fit[7]*fit[0]+fit[8]*fit[1])/U**2 -C*fit[2]/U**4),
                fit[0]**2/U**2,fit[1]**2/U**2,fit[2]**2/U**2,
                2*fit[0]*fit[1]/U**2,2*fit[0]*fit[2]/U**2,2*fit[1]*fit[2]/U**2
            ]
        ])
    
    return out, J.dot(cov).dot(J.T)

def get_distance_estimates(fit,cov):
    """
    Get distance estimates and their covariance from bulk flow and shear

    fit -- numpy array of shape (N,)
           first N/2 entries are bulkflow
           last N/2 are shear
    cov -- numpy array of shape (N,N)
           covariance of fit
    """
    N = fit.shape[0]/2
    d = fit[:N]/fit[N:]

    # Jacobian 
    J = np.zeros((N,2*N))
    for k in xrange(N):
        J[k,k] = 1/fit[N+k]
        J[k,k+N] = -fit[k]/fit[N+k]**2

    cov_d = J.dot(cov).dot(J.T)

    return d, cov_d

def get_distance_estimates_mono(fit,cov):
    """
    Get distance estimates and their covariance from bulk flow, shear and monopole

    fit -- numpy array of shape (2*N+1,)
           first N entries are bulkflow
           second N are shear
           last entry is monopole
    cov -- numpy array of shape (2*N+1,2*N+1)
           covariance of fit
    """
    N = fit.shape[0]/2
    v = fit[:N]
    s = fit[N:2*N]
    m = fit[2*N]

    d = 2*v/(s-2*m)

    # Jacobian 
    J = np.zeros((N,2*N+1))
    for k in xrange(N):
        J[k,k] = 2/(s[k]-2*m)
        J[k,k+N] = -2*v[k]/(s[k]-2*m)**2
        J[k,2*N] =  4*v[k]/(s[k]-2*m)**2

    cov_d = J.dot(cov).dot(J.T)

    return d, cov_d

def get_distance_estimates_kaiser(fit,cov,gamma=2.):
    """
    Get distance estimates and their covariance from bulk flow and shear

    fit -- numpy array of shape (9,)
           first 3 entries are bulkflow
           last 6 are shear matrix, first diagonal terms, then off-diag
    cov -- numpy array of shape (9,9)
           covariance of fit
    """
    B = 2. * (1. + gamma) / 3.
    U = np.sqrt(np.sum(fit[:3]**2))
    C = np.sum(fit[3:6]*fit[:3]**2) + 2 * (fit[0]*fit[1]*fit[6] +
                                           fit[0]*fit[2]*fit[7] +
                                           fit[1]*fit[2]*fit[8])

    d = B * U**3 / C
    
    # Jacobian 
    J = np.array([
        3*U*fit[0]/C - 2*U**3/C**2*(fit[0]*fit[3]+fit[1]*fit[6]+fit[2]*fit[7]),
        3*U*fit[1]/C - 2*U**3/C**2*(fit[1]*fit[4]+fit[0]*fit[6]+fit[2]*fit[8]),
        3*U*fit[2]/C - 2*U**3/C**2*(fit[2]*fit[5]+fit[0]*fit[7]+fit[1]*fit[8]),
        -U**3*fit[0]**2/C**2,
        -U**3*fit[1]**2/C**2,
        -U**3*fit[2]**2/C**2,
        -2*U**3*fit[0]*fit[1]/C**2,
        -2*U**3*fit[0]*fit[2]/C**2,
        -2*U**3*fit[1]*fit[2]/C**2
        ]) * B

    cov_d = J.dot(cov).dot(J)

    return d, cov_d

def make_bf_rhs(v):
    """
    Returns three vectors right-handed orthonormal coordinate system with
    v/|v| as first vector. Can then be used to get distance estimate in
    bulk flow direction.
    """

    v1 = v/np.sqrt(v.dot(v))

    v2 = np.array([0,v1[2],-v1[1]])
    v2 = v2/np.sqrt(v2.dot(v2))  
    
    v3 = np.cross(v1,v2)
    v3 = v3/np.sqrt(v3.dot(v3))                                       

    return [v1,v2,v3]

def remove_tr(fit,cov):
    """
    Remove trace from shear matrix including monopole
    Expects fit results for bulk flow + shear (non-traceless)
    fit = (array of length 9: U_x, U_y, U_z, U_xx, U_yy, U_zz, U_xy, U_xz, U_yz)
    Return
    fit_notrace = (array of length 9: U_x, U_y, U_z, U~_xx, U~_yy, U~_zz, U_xy, U_xz, U_yz, U)
    """
    J = np.array([[ 1,  0,  0,     0,      0,     0,  0,  0,  0],
                  [ 0,  1,  0,     0,      0,     0,  0,  0,  0],
                  [ 0,  0,  1,     0,      0,     0,  0,  0,  0],
                  [ 0,  0,  0,  2./3,  -1./3, -1./3,  0,  0,  0],
                  [ 0,  0,  0, -1./3,   2./3, -1./3,  0,  0,  0],
                  [ 0,  0,  0, -1./3,  -1./3,  2./3,  0,  0,  0],
                  [ 0,  0,  0,     0,      0,     0,  1,  0,  0],
                  [ 0,  0,  0,     0,      0,     0,  0,  1,  0],
                  [ 0,  0,  0,     0,      0,     0,  0,  0,  1],
                  [ 0,  0,  0,  1./3,   1./3,  1./3,  0,  0,  0]])

    return J.dot(fit),  J.dot(cov.dot(J.T))

def add_shear_z(fit,cov):
    """
    Add the z_component to shear matrix without monopole
    Expects fit results for bulk flow + shear (traceless)
    fit = (array of length 9: U_x, U_y, U_z, U~_xx, U~_yy, U_xy, U_xz, U_yz)
    """
    J = np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
                  [ 0,  1,  0,  0,  0,  0,  0,  0],
                  [ 0,  0,  1,  0,  0,  0,  0,  0],
                  [ 0,  0,  0,  1,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  1,  0,  0,  0],
                  [ 0,  0,  0, -1, -1,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  1,  0,  0],
                  [ 0,  0,  0,  0,  0,  0,  1,  0],
                  [ 0,  0,  0,  0,  0,  0,  0,  1]])

    return J.dot(fit),  J.dot(cov.dot(J.T))

def reshape_shear_matrix(flat_shear,offset=0):
    """
    Reshape flattened shear (array of length 6: U_xx, U_yy, U_zz, U_xy, U_xz, U_yz) to matrix.
    """
    shear_idx = np.array([[0,3,4],
                          [3,1,5],
                          [4,5,2]])
    return flat_shear[shear_idx+offset]

def flatten_shear_matrix(shear):
    shear_idx = np.array([0,4,8,1,2,5])
    return shear.flatten()[shear_idx]

# -------------------------- #
# -- Conversion functions -- #
# -------------------------- #

def convert_spherical(best_fit, cov=None):
    """
    Convert fit results in Cartesian coordinates to spherical coordinates 
    (angles in degrees). Convariance matrix can be converted as well
    if it is stated.
    """
    x = best_fit[0]
    y = best_fit[1] 
    z = best_fit[2] 

    v = np.sqrt(x**2 + y**2 + z**2)
    v_sph = np.array([v, (np.arctan2(y,x) / _d2r + 180) % 360 - 180, 
                          np.arcsin(z/v) / _d2r])
    
    if cov is None:
        return v_sph
    else:
        cov_out = deepcopy(cov)    

        jacobian = np.zeros((3,3))
        jacobian[0,0] = x / v
        jacobian[1,0] = - y / (x**2 + y**2)
        jacobian[2,0] = - x * z / (v**2 * np.sqrt(x**2 + y**2))
        jacobian[0,1] = y / v
        jacobian[1,1] = x / (x**2 + y**2)
        jacobian[2,1] = - y * z / (v**2 * np.sqrt(x**2 + y**2))
        jacobian[0,2] = z / v
        jacobian[1,2] = 0
        jacobian[2,2] = np.sqrt(x**2 + y**2) / (v**2)

        cov_sph = (jacobian.dot(cov_out)).dot(jacobian.T)
        cov_sph[1,1] /= _d2r**2
        cov_sph[2,2] /= _d2r**2
        cov_sph[2,1] /= _d2r**2
        cov_sph[1,2] /= _d2r**2
    
        cov_sph[0,1] /= _d2r
        cov_sph[0,2] /= _d2r
        cov_sph[1,0] /= _d2r
        cov_sph[2,0] /= _d2r    

        return v_sph, cov_sph

def convert_cartesian(best_fit, cov=None):
    """
    Convert fit results in spherical coordinates (angles in degrees)
    to Cartesian coordinates. Convariance matrix can be converted as well
    if it is stated.
    """
    v = best_fit[0]
    l = best_fit[1]*_d2r
    b = best_fit[2]*_d2r

    v_cart = np.array([v*np.cos(b)*np.cos(l), v*np.cos(b)*np.sin(l), 
                       v*np.sin(b)])    

    if cov is None:
        return v_cart
    else:
        cov_out = deepcopy(cov)
        cov_out[1,1] *= _d2r**2
        cov_out[2,2] *= _d2r**2
        cov_out[2,1] *= _d2r**2
        cov_out[1,2] *= _d2r**2
        cov_out[0,1] *= _d2r
        cov_out[0,2] *= _d2r
        cov_out[1,0] *= _d2r
        cov_out[2,0] *= _d2r

        jacobian = np.zeros((3,3))
        jacobian[0,0] = np.cos(b) * np.cos(l)
        jacobian[1,0] = np.cos(b) * np.sin(l)
        jacobian[2,0] = np.sin(b)
        jacobian[0,1] = - v * np.cos(b) * np.sin(l)
        jacobian[1,1] = v * np.cos(b) * np.cos(l)
        jacobian[2,1] = 0
        jacobian[0,2] = - v * np.sin(b) * np.cos(l)
        jacobian[1,2] = - v * np.sin(b) * np.sin(l)
        jacobian[2,2] = v * np.cos(b)

        cov_cart = (jacobian.dot(cov_out)).dot(jacobian.T)

        return v_cart, cov_cart

def sex2deg(RA, Dec, hours=True):
    """
    Convert coordinates from sexagesimal to degrees.

    Arguments:
    RA  -- Right ascension (in hours if hours is True else in degrees)
    Dec -- Declination (in degrees)
    """
    RA = np.sum([float(a) * 60**-k for k,a in enumerate(RA.split(':'))])
    if hours:
        RA *= 15
    sign = 1
    if Dec[0] == '-':
        sign = -1
    Dec = np.sum([np.abs(float(a)) * 60**-k 
                  for k,a in enumerate(Dec.split(':'))]) * sign
    return RA, Dec

def attractor_mass(delta,R_A,H_0=_h,O_M=_O_M):
    G = 6.67384e-11 # in m^3 kg^-1 s^-2
    M_solar = 1.9891e30 # in kg
    Mpc = 3.05987e22 # metres in an Mpc
    mass_factor=0.5*(H_0*1000/Mpc)**2/G*(R_A*Mpc)**3/M_solar*O_M
    return mass_factor*(1+delta)
