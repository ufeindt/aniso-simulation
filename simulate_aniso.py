#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate random realizations of distance  

Note: The convention for spherical coordinates used here is
      azimuth (i.e. RA or l) first and then inclination (i.e. Dec or b).
      All angles are given in degrees within azimuths of -180 degrees and 180 degrees
      and inclinations between -90 degrees and 90 degrees.

Author: Ulrich Feindt (feindt@physik.hu-berlin.de)
"""

import numpy as np
import os
import sys

from argparse import ArgumentParser

import analysis_tools as at
import cosmo_tools as ct
import simulation_tools as st
import velocity_tools as vt

signal_modes = ['0: no signal',
                '1: dipole, 3 parameters (U_x,U_y,U_z)',
                '2: dipole + shear, 9 parameters (U_x,U_y,U_z,U_xx,U_yy,U_zz,U_xy,U_xz,U_yz)',
                '3: SSC, 1 parameter (overdensity)',
                '4: SSC + SGW, 2 parameters (overdensities)',
                '5: SSC + Vela, 2 parameters (overdensities)']

example_dipole = vt.convert_cartesian([300,300,30])
rhs = np.array(vt.make_bf_rhs(example_dipole))
example_shear = np.array([[1.5,   0.,   0.],
                          [ 0., -.75,   0.],
                          [ 0.,   0., -.75]])
example_shear = vt.flatten_shear_matrix(rhs.T.dot(example_shear.dot(rhs))) 

parameter_default = [[],
                     example_dipole,
                     np.concatenate((example_dipole,example_shear)),
                     [1],
                     [1,1],
                     [1,1]]

def _def_parser():
    parser = ArgumentParser(description='Simulate random realizations of distance')
    parser.add_argument('files',type=str,nargs='*',help='coordinate files')
    parser.add_argument('-z','--redshift',default=None,nargs=2,
                        help='redshift boundaries',type=float)
    parser.add_argument('-v','--verbose',default=False,action='store_true',
                        help='verbosity')
    parser.add_argument('-o','--outdir',default=None,type=str,
                        help='outdir')
    parser.add_argument('--short-name',default=None,type=str,
                        help='short name for saving the results')
    parser.add_argument('-n','--number',default=100,type=int,
                        help='number of realization')
    parser.add_argument('-s','--signal-mode',default=0,type=int,
                        help='signal mode: {}'.format('; '.join(signal_modes)))
    parser.add_argument('-p','--parameters',default=None,nargs='*',type=float,
                        help='non-default parameters for signal')

    return parser
      
def _process_args(args):
    messages = ['Files:','\n'.join(args.files),'',
                'Number of realizations: {}'.format(args.number)]

    if args.redshift is None:
        args.redshift = [0.015,0.1]
        messages.append('Redshift range: {:.3f} -- {:.3f} (default)'.format(*args.redshift))
    elif args.redshift[0] > args.redshift[1]:
        raise ValueError('Invalid redshift range: {:.3f} -- {:.3f}'.format(*args.redshift))
    else:
        messages.append('Redshift range: {:.3f} -- {:.3f}'.format(*args.redshift))

    if args.outdir is None:
        args.outdir = 'results/'
    elif args.outdir[-1] != '/':
        args.outdir = '{}/'.format(args.outdir)
    messages.append('Output dir: {}'.format(args.outdir[:-1]))
   
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.signal_mode not in range(6):
        raise ValueError('Unknown signal mode: {}'.format(args.signal_mode))
    messages.append('Signal mode {}'.format(signal_modes[args.signal_mode]))

    if args.parameters is None:
        args.parameters = parameter_default[args.signal_mode]
    elif len(args.parameters) != len(parameter_default[args.signal_mode]):
        raise ValueError('Wrong number of parameters for chosen signal mode.')
    messages.append('Parameters: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                          for val in args.parameters])))

    if args.verbose:
        print '\n'.join(messages)    
    
    return args

def _make_outfile_name(args):
    if args.short_name is None:
        outfile = '_'.join(args.files)
    else:
        outfile = args.short_name

    outfile = '{}_s{}_{:.3f}_{:.3f}'.format(outfile,args.signal_mode,*args.redshift)
    
    outfile = '{}.pkl'.format(outfile.replace('.',''))

    return outfile

def _load_coordinates(*filenames,**kwargs):
    z_range = kwargs.pop('z_range',(0.015,0.1))

    out = ([],[],[],[])
    for filename in filenames:
        f = file(filename,'r')
        data = [line.split() for line in f if line[0] != '#']
        f.close()
        
        out[0].extend([a[0] for a in data 
                       if float(a[3]) >= z_range[0] and float(a[3]) < z_range[1]])
        for k in range(1,4):
            out[k].extend([float(a[k]) for a in data
                           if float(a[3]) >= z_range[0] and float(a[3]) < z_range[1]])

        return (np.array(a) for a in out)

def _get_peculiar_velocities(mode,p,l,b,z):
    v_fcts = {0: (lambda p_,l_,b_,z_: 
                  np.zeros(len(z_))),
              1: (lambda p_,l_,b_,z_: 
                  np.array(map(lambda x1,x2: p_.dot(vt.v_dipole_comp(x1,x2)),l_,b_))),
              2: (lambda p_,l_,b_,z_: 
                  np.array(map(lambda x1,x2,x3: p_.dot(vt.v_tidal_comp(x1,x2,x3)),z_,l_,b_)))}
    if mode > 2:
        raise ValueError('Signal mode not implemented yet.')

    return v_fcts[mode](p,l,b,z)

def _simulate_aniso(num_sim,names,l,b,z,v):
    results = {'dipole': {},
               'dipole+shear': {},
               'dipole+shear_trless': {},
               'sr_90': {},
               'sr_45': {},
               'sr_22.5:': {},
               'sr_nw_90': {},
               'sr_nw_45': {},
               'sr_nw_22.5:': {}}

    analysis_fcts = {'dipole': at.fit_dipole,
                     'dipole+shear': at.fit_dipole_shear,
                     'dipole+shear_trless': at.fit_dipole_shear_trless,
                     'sr_90': (lambda data,options: at.get_sr_min_max(data,option,delta=90)),
                     'sr_45': (lambda data,options: at.get_sr_min_max(data,option,delta=45)),
                     'sr_22.5:': (lambda data,options: at.get_sr_min_max(data,option,
                                                                         delta=22.5)),
                     'sr_nw_90': (lambda data,options: at.get_Q_min_max(data,option,delta=90,
                                                                        weighted=False)),
                     'sr_nw_45': (lambda data,options: at.get_Q_min_max(data,option,delta=45,
                                                                        weighted=False)),
                     'sr_nw_22.5:': (lambda data,options: at.get_Q_min_max(data,option,
                                                                           delta=22.5,
                                                                           weighted=False))}
    
    for k in xrange(num_sim):
        data, options, res = st.simulate_data(names,l,b,z,v=v)
        for key in sorted(results.keys()):
            tmp = analysis_fct[key](data,options)
            for skey in tmp.keys():
                if skey not in results[key].keys():
                    if type(tmp[skey]) == list:
                        results[key][skey] = tmp[skey]
                    elif type(tmp[skey]) == float:
                        results[key][skey] = np.array([tmp[skey]])
                    else:
                        raise TypeError('Output of analysis functions must be a dictionary of lists and floats.')
                else:
                    if type(tmp[skey]) == list:
                        results[key][skey].extend(tmp[skey])
                    elif type(tmp[skey]) == float:
                        results[key][skey] = np.append(results[key][skey],tmp[skey])
                    else:
                        raise TypeError('Output of analysis functions must be a dictionary of lists and floats.')

    return results

def _save_results(results,outfile):
    pass

def _main():
    parser = _def_parser()
    args = parser.parse_args()
    args = _process_args(args)

    outfile = _make_outfile_name(args)

    names, RA, Dec, z = _load_coordinates(*args.files,z_range=args.redshift)
    l, b = ct.radec2gcs(RA, Dec)

    v = _get_peculiar_velocities(args.signal_mode,args.parameters,l,b,z)

    results = _simulate_aniso(names,l,b,z,v)
    #sys.exit()
    
    _save_results(results,'{}{}'.format(args.outdir,outfile))

if __name__ == '__main__':
    _main()
