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
from cosmo_tools import _d2r

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
        else:
            z_pdf_bins = np.array(z_pdf_bins)
            z_pdf = np.ones(len(z_pdf_bins)-1)/(len(z_pdf_bins)-1)
    else:
        z_pdf = np.array(z_pdf,dtype=float)/np.sum(np.array(z_pdf))
        if z_pdf_bins is None:
            z_pdf_bins = np.linspace(z_range[0],z_range[1],len(z_pdf)+1)
        elif (z_pdf_bins[0] != z_range[0] or z_pdf_bins[-1] != z_range[1]
              or True in [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]):
            print z_pdf_bins[0] != z_range[0]
            print z_pdf_bins[1] != z_range[-1]
            print [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]
            print True in [a>b for a,b in zip(z_pdf_bins[:-1],z_pdf_bins[1:])]
            raise ValueError('Invalid z_pdf_bins')
        else:
            z_pdf_bins = np.array(z_pdf_bins)
    
    z_cdf = np.cumsum(z_pdf)
    val_uni = np.random.random(NPoints)
    val_bins = np.array([np.where(z_cdf > val)[0][0] for val in val_uni])
    val_rem = ((val_uni - z_cdf[val_bins-1])%1)/((z_cdf[val_bins]-z_cdf[val_bins-1])%1)

    z = z_pdf_bins[val_bins] + (z_pdf_bins[val_bins+1] - z_pdf_bins[val_bins]) * val_rem

    return z
         
