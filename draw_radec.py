#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Draw random SNe redshifts 

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import os

from argparse import ArgumentParser

import simulation_tools as st

def _def_parser():
    parser = ArgumentParser(description='Draw RA, Dec and redshifts for test sample')
    parser.add_argument('name',type=str,help='Sample name; data will be save in name.dat if not stated otherwise')
    parser.add_argument('-z','--redshift',default=None,nargs=2,
                        help='redshift boundaries',type=float)
    parser.add_argument('-r','--ra-range',default=None,nargs=2,
                        help='RA range in degrees',type=float)
    parser.add_argument('-d','--dec-range',default=None,nargs=2,
                        help='Dec range in degrees',type=float)
    parser.add_argument('--z-pdf',default=None,nargs='*',
                        help='pdf values for non-flat distribution (if --z-cdf-bins not stated, redshift range will be split uniformly)',type=float)
    parser.add_argument('--z-pdf-bins',default=None,nargs='*',
                        help='redshift bins for non-flat distribution',type=float)
    parser.add_argument('-a','--zone-of-avoidance',default=None,nargs=1,
                        help='size of the ZoA in degrees',type=float)
    parser.add_argument('-v','--verbose',default=False,action='store_true',
                        help='verbosity')
    parser.add_argument('-o','--outfile',default=None,type=str,
                        help='outfile name')
    parser.add_argument('-n','--number',default=100,type=int,
                        help='number of SNe')

    return parser
      
def _process_args(args):
    messages = ['Name: {}'.format(args.name),
                'Number of SNe: {}'.format(args.number)]

    if args.redshift is None:
        args.redshift = [0.015,0.1]
        messages.append('Redshift range: {:.3f} -- {:.3f} (default)'.format(*args.redshift))
    elif args.redshift[0] > args.redshift[1]:
        raise ValueError('Invalid redshift range: {:.3f} -- {:.3f}'.format(*args.redshift))
    else:
        messages.append('Redshift range: {:.3f} -- {:.3f}'.format(*args.redshift))

    if args.ra_range is None:
        args.ra_range = [-180.,180.]
        messages.append('RA range: {:.1f} deg -- {:.1f} deg (default)'.format(*args.ra_range))
    elif (args.ra_range[0] < -180 or args.ra_range[1] > 180 
          or args.ra_range[0] >  args.ra_range[1]):
        raise ValueError('Invalid RA range: {:.1f} -- {:.1f}'.format(*args.ra_range))
    else:
        messages.append('RA range: {:.1f} deg -- {:.1f} deg'.format(*args.ra_range))

    if args.dec_range is None:
        args.dec_range = [-90.,90.]
        messages.append('Dec range: {:.1f} deg -- {:.1f} deg (default)'.format(*args.dec_range))
    elif (args.dec_range[0] < -90 or args.dec_range[1] > 90 
          or args.dec_range[0] >  args.dec_range[1]):
        raise ValueError('Invalid Dec range: {:.1f} -- {:.1f}'.format(*args.dec_range))
    else:
        messages.append('Dec range: {:.1f} deg -- {:.1f} deg'.format(*args.dec_range))

    
    if args.zone_of_avoidance is None:
        args.zone_of_avoidance = 10.
        messages.append('ZoA size: {:.1f} deg (default)'.format(args.zone_of_avoidance))
    else:
        messages.append('ZoA size: {:.1f} deg'.format(args.zone_of_avoidance))

    if args.z_pdf is None:
        if args.z_pdf_bins is None:
            args.z_pdf = np.ones(1)
            args.z_pdf_bins = np.array(z_range)
        else:
            args.z_pdf_bins = np.array(args.z_pdf_bins)
            args.z_pdf = np.ones(len(args.z_pdf_bins)-1)/(len(args.z_pdf_bins)-1)         
    else:
        args.z_pdf = np.array(args.z_pdf)/np.sum(np.array(args.z_pdf))
        if args.z_pdf_bins is None:
            args.z_pdf_bins = np.linspace(args.redshift[0],args.redshift[1],len(args.z_pdf)+1)
        elif (args.z_pdf_bins[0] != args.redshift[0] or args.z_pdf_bins[-1] != args.redshift[1]
              or True in [a>b for a,b in zip(args.z_pdf_bin[:-1],args.z_pdf_bin[1:])]):
            raise ValueError('Invalid redshift pdf bins')
        else:
            args.z_pdf_bins = np.array(args.z_pdf_bins)

    messages.append('Redshift pdf: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                           for val in args.z_pdf])))
    messages.append('Redshift pdf bins: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                                 for val in args.z_pdf_bins])))

    if args.outfile is None:
        args.outfile = '{}.dat'.format(args.name)
    messages.append('Output file: {}'.format(args.outfile))

    outdir = '/'.join(args.outfile.split('/')[:-1])
    if outdir != '' and not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.verbose:
        print '\n'.join(messages)
    
    return args

def _save_sample(name,RA,Dec,z,outfile):
    """
    """
    out = np.zeros((len(name),),dtype=[('NAME', 'S6'), ('RA', '<f8'), ('DEC', '<f8'), ('Z', '<f8')])
    out['NAME'] = name
    out['RA'] = RA
    out['DEC'] = Dec
    out['Z'] = z

    _save_structured_array(out,outfile)

def _save_structured_array(array,outfile,delimiter=' '):
    """
    """
    header = ' '.join(array.dtype.names)

    fmt = ['%f' if a[1][:2] == '<f' else '%i' if a[1][:2] == '<i' else '%s' 
           for a in array.dtype.descr]

    np.savetxt(outfile,array,delimiter=delimiter,header=header,fmt=('%s','%f','%f','%f'))

def _main():
    parser = _def_parser()
    args = parser.parse_args()
    args = _process_args(args)

    print args

    RA, Dec = st.simulate_l_b_coverage(args.number,args.zone_of_avoidance,
                                       args.ra_range,args.dec_range,'j2000')
    z = st.simulate_z_coverage(args.number,args.redshift,args.z_pdf,args.z_pdf_bins)

    names = np.array(['{}{}'.format(args.name,k) for k in range(len(RA))])

    _save_sample(names,RA,Dec,z,args.outfile)

if __name__ == '__main__':
    _main()
