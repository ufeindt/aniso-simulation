#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation tools

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np

import cosmo_tools as ct
import velocity_tools as vt
from cosmo_tools import _d2r, _O_M, _H_0, _c

def load_from_files(*filenames,**kwargs):
    """
    Load data from files.
    
    Requires files to contain header with keys for each column.
    Only columns according to kwargs keys are returned. If keys are
    given dtype must be given (use object for strings).

    Can filter for redshifts and will look for redshift key (named 
    'redshift' or starting with 'z'). If none is found or results 
    ambiguous, it must be stated manually.

    Returns list of as many numpy array as keys (+ fileindex array 
    if return_fileindex is True)
    """
    if 'keys' in kwargs.keys() and 'dtype' not in kwargs.keys():
        raise ValueError('Please set dtype as well.')
    elif 'keys' in kwargs.keys() and 'dtype' in kwargs.keys():
        if len(kwargs['keys']) != len(kwargs['dtype']):
            raise ValueError('Length of keys and dtype must match.')

    z_range = kwargs.pop('z_range',None)
    z_key = kwargs.pop('z_key',None)
    keys = kwargs.pop('keys',['Name','RA','Dec','z'])
    dtypes = kwargs.pop('dtype',[object,float,float,float])
    case_sensitive = kwargs.pop('case_sensitive',False)
    comments = kwargs.pop('comments','#')
    delimiter = kwargs.pop('delimeter',None)
    return_fileindex = kwargs.pop('return_fileindex',False)

    if kwargs != {}:
        unknown_kw = ' '.join(kwargs.keys())
        raise TypeError('load_from_files got unknown keyword arguments: {}'.format(unknown_kw))

    if not case_sensitive:
        keys = [a.upper() for a in keys]

    if z_range is not None and z_key is None:
        z_keys = [key for key in keys 
                  if key[0].upper() == 'Z' or key.upper() == "REDSHIFT"] 
        if len(z_keys) == 0:
            raise ValueError('Failed to determine z_key, please set kwarg z_key')
        elif len(z_keys) > 1:
            raise ValueError('Ambiguous z_key, please set kwargs z_key manually')
        else:
            z_key = z_keys[0]

    out = None
    fileindex = []

    for k,filename in enumerate(filenames):
        tmp = np.genfromtxt(filename,names=True,comments=comments,dtype=None,
                            case_sensitive=case_sensitive,delimiter=delimiter)
        
        if z_range is None:
            tmp2 = np.zeros((len(tmp),),dtype=zip(keys,dtypes))
            fileindex.extend([k for a in range(len(tmp))])
            for key in keys:
                tmp2[key] = tmp[key]
        else:
            z_filter = (tmp[z_key] >= z_range[0]) & (tmp[z_key] < z_range[1]) 
            tmp2 = np.zeros((np.sum(z_filter),),dtype=zip(keys,dtypes))
            fileindex.extend([k for a in range(np.sum(z_filter))])
            for key in keys:
                tmp2[key] = tmp[key][z_filter]
                    
        if out is None:
            out = tmp2
        else:
            out = np.concatenate((out,tmp2))
            
    if return_fileindex:
        return [out[key] for key in keys] + [np.array(fileindex)]
    else:
        return [out[key] for key in keys]

def simulate_l_b_coverage(Npoints,MW_exclusion=10,ra_range=(-180,180),dec_range=(-90,90),
                          output_frame='galactic'):
    """
    """
    # ----------------------- #
    # --                   -- #
    # ----------------------- #
    def _draw_radec_(Npoints_,ra_range_,dec_sin_range_):
        """
        """
        ra = np.random.random(Npoints_)*(ra_range_[1] - ra_range_[0]) + ra_range_[0]
        dec = np.arcsin(np.random.random(Npoints_)*(dec_sin_range_[1] - dec_sin_range_[0]) + dec_sin_range_[0]) / _d2r

        return ra,dec

    def _draw_without_MW_(Npoints_,ra_range_,dec_sin_range_,MW_exclusion_):
        """
        """
        
        l,b = np.array([]),np.array([])
        while( len(l) < Npoints_ ):
            ra,dec = _draw_radec_(Npoints_ - len(l),ra_range_,dec_sin_range_)
            l_,b_ = ct.radec2gcs(ra,dec)
            if output_frame == 'galactic':
                l = np.concatenate((l,l_[np.abs(b_)>MW_exclusion_]))
                b = np.concatenate((b,b_[np.abs(b_)>MW_exclusion_]))
            else:
                l = np.concatenate((l,ra[np.abs(b_)>MW_exclusion_]))
                b = np.concatenate((b,dec[np.abs(b_)>MW_exclusion_]))                

        return l,b

    # ----------------------- #
    # --                   -- #
    # ----------------------- #

    if output_frame not in ['galactic','j2000']:
        raise ValueError('output_frame must "galactic" or "j2000"')

    if ra_range[0] < -180 or ra_range[1] > 180 or ra_range[0] > ra_range[1]:
        raise ValueError('ra_range must be contained in [-180,180]')

    if dec_range[0] < -90 or dec_range[1] > 90 or dec_range[0] > dec_range[1]:
        raise ValueError('dec_range must be contained in [-180,180]')

    dec_sin_range = (np.sin(dec_range[0]*_d2r),np.sin(dec_range[1]*_d2r)) 

    if MW_exclusion > 0.:
        return _draw_without_MW_(Npoints,ra_range,dec_sin_range,MW_exclusion)
    else:
        ra,dec = _draw_radec_(Npoints,ra_range,dec_sin_range)
        if output_frame == 'galactic':
            return ct.radec2gcs(ra,dec)
        else:
            return ra,dec

def simulate_z_coverage(NPoints,z_range,z_pdf=None,z_pdf_bins=None):
    """

    """
    if (len(z_range) != 2 or z_range[0] > z_range[1]):
        raise ValueError('Invalid z_range')
        
    if z_pdf is None:
        if z_pdf_bins is None:
            z_pdf = np.ones(1)
            z_pdf_bins = np.array(z_range)
            widths = np.array([z_range[1]-z_range[0]])
        else:
            z_pdf_bins = np.array(z_pdf_bins)
            z_pdf = np.ones(len(z_pdf_bins)-1)/(len(z_pdf_bins)-1)
    else:
        if z_pdf_bins is None:
            z_pdf_bins = np.linspace(z_range[0],z_range[1],len(z_pdf)+1)
        elif (np.abs(z_pdf_bins[0] - z_range[0]) / z_range[0] > 1e-9 
              or np.abs(z_pdf_bins[-1] - z_range[1]) / z_range[1] > 1e-9 
              or True in [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]):
            print np.abs(z_pdf_bins[0] - z_range[0]) / z_range[0] > 1e-9 
            print np.abs(z_pdf_bins[-1] - z_range[1]) / z_range[1] > 1e-9 
            print [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]
            print True in [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]
            raise ValueError('Invalid z_pdf_bins')
        else:
            z_pdf_bins = np.array(z_pdf_bins)

    widths = z_pdf_bins[1:]-z_pdf_bins[:-1]
    z_pdf = np.array(z_pdf,dtype=float)/np.sum(np.array(z_pdf*widths))
    print np.sum(z_pdf*widths)
    print z_pdf

    if len(z_pdf) > 1:
        z_cdf = np.cumsum(z_pdf*widths)
        val_uni = np.random.random(NPoints)
        val_bins = np.array([np.where(z_cdf > val)[0][0] for val in val_uni])
        val_rem = ((val_uni - z_cdf[val_bins-1])%1)/((z_cdf[val_bins]-z_cdf[val_bins-1])%1)

        z = z_pdf_bins[val_bins] + (z_pdf_bins[val_bins+1] - z_pdf_bins[val_bins]) * val_rem
    else:
        z = np.random.random(NPoints) * (z_range[1]-z_range[0]) + z_range[0]

    return z
         
def simulate_data(names,l,b,z,v=None,O_M=_O_M,H_0=_H_0,#v_dispersion=0,
                  intrinsic_dispersion=0.1,error_distribution=(0.1,0.02),
                  error_min=0.03,add=None,v_mode='hui'):
    """
    """
    if add is not None and add['number'] > 0:
        new_names = np.array(['add{:6.0f}'.format(k) for k in xrange(add['number'])])
        new_z = simulate_z_coverage(add['number'],add['z_range'],
                                    add['z_pdf'],add['z_pdf_bins'])
        new_RA, new_Dec = simulate_l_b_coverage(add['number'],add['ZoA'],
                                               add['ra_range'],add['dec_range'],'j2000')
        new_l, new_b = ct.radec2gcs(new_RA,new_Dec)

        z_filter = (new_z >= add['z_limits'][0]) & (new_z < add['z_limits'][1])
        names = np.concatenate((names,new_names[z_filter]))
        z = np.concatenate((z,new_z[z_filter]))
        l = np.concatenate((l,new_l[z_filter]))
        b = np.concatenate((b,new_b[z_filter]))

        if v is not None:
            new_v = get_peculiar_velocities(add['signal_mode'],add['parameters'],
                                            new_l[z_filter],new_b[z_filter],new_z[z_filter])
            v = np.concatenate((v,new_v))

    if v is None:
        v = np.zeros(len(z)) 

    #z4mu = (1 - z) / (1 - v/_c) - 1
    #if v_dispersion > 0:
    #    z4mu += np.random.normal(0,v_dispersion,len(z))
    
    mu = np.array([ct.mu(z_,O_M=O_M,H_0=H_0,v_mon=v_,v_mode=v_mode) 
                   for z_,v_ in zip(z,v)])
    
    dmu = np.random.normal(error_distribution[0],error_distribution[1],len(z))

    for k in xrange(len(z)):
        while dmu[k] <= error_min:
            dmu[k] = np.random.normal(error_distribution[0],error_distribution[1])

        mu[k] += np.random.normal(0,np.sqrt(dmu[k]**2 + intrinsic_dispersion**2))
        
    data = [zip(names,z,mu,dmu,l,b)]
    options = {'O_L':None,'w':-1,'dM':0,'H_0':H_0,'O_M':O_M}
    fit, options = ct.fit_w_sig_int([0], data, options)
    options['offsets'] = fit[0]
        
    return data, options

def get_peculiar_velocities(mode,p,l,b,z):
    v_fcts = {0: (lambda p_,l_,b_,z_: 
                  np.zeros(len(z_))),
              1: (lambda p_,l_,b_,z_: 
                  np.array(map(lambda x1,x2: p_.dot(vt.v_dipole_comp(x1,x2)),l_,b_))),
              2: (lambda p_,l_,b_,z_: 
                  np.array(map(lambda x1,x2,x3: p_.dot(vt.v_tidal_comp(x1,x2,x3)),z_,l_,b_))),
              3: (lambda p_,l_,b_,z_:
                  np.array(map(lambda x1,x2,x3: vt.convert_cartesian([1,x2,x3]).
                               dot(vt.v_attractor(x1,x2,x3,[p_[0],0])),z_,l_,b_))),
              4: (lambda p_,l_,b_,z_:
                  np.array(map(lambda x1,x2,x3: vt.convert_cartesian([1,x2,x3]).
                               dot(vt.v_attractor(x1,x2,x3,p_)),z_,l_,b_)))}

    return v_fcts[mode](p,l,b,z)
