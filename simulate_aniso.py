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

__version__ = '1.2' 

import numpy as np
import os
import re 
import warnings
import cPickle
import datetime

from argparse import ArgumentParser

import analysis_tools as at
import cosmo_tools as ct
import simulation_tools as st
import velocity_tools as vt

from cosmo_tools import _O_M, _H_0

signal_modes = ['0: no signal',
                '1: dipole, 3 parameters (U_x,U_y,U_z)',
                '2: dipole + shear, 9 parameters (U_x,U_y,U_z,U_xx,U_yy,U_zz,U_xy,U_xz,U_yz)',
                '3: SSC, 1 parameter (overdensity)',
                '4: SSC + SGW, 2 parameters (overdensities)']

example_dipole = vt.convert_cartesian([300,300,30])
rhs = np.array(vt.make_bf_rhs(example_dipole))
example_shear = np.array([[1.5,   0.,   0.],
                          [ 0., -.75,   0.],
                          [ 0.,   0., -.75]])
example_shear = vt.flatten_shear_matrix(rhs.T.dot(example_shear.dot(rhs))) 

parameter_default = [[],
                     example_dipole,
                     np.concatenate((example_dipole,example_shear)),
                     np.array([2.78]), # maybe wrong
                     np.array([1.078,12.32])]
                     # np.array([1.75,15.2])] # thesis values
def _def_parser():
    parser = ArgumentParser(description='Simulate random realizations of distance')
    parser.add_argument('files',type=str,nargs='*',help='coordinate files')
    parser.add_argument('-z','--redshift',default=None,nargs=2,
                        help='redshift boundaries',type=float)
    parser.add_argument('-v','--verbosity',action='count',
                        help='verbosity')
    parser.add_argument('-o','--outdir',default=None,type=str,
                        help='outdir')
    parser.add_argument('--short-name',default=None,type=str,
                        help='short name for saving the results')
    parser.add_argument('-n','--number',default=100,type=int,
                        help='number of realization')
    parser.add_argument('-s','--signal-mode',default=0,type=int,choices=range(len(signal_modes)),
                        help='signal mode:\n {}'.format('\n'.join(signal_modes)))
    parser.add_argument('-p','--parameters',default=None,nargs='*',type=float,
                        help='non-default parameters for signal')

    parser.add_argument('--sim-number',default=None,type=int,
                        help='number of additional SNe')
    parser.add_argument('--sim-redshift',default=None,nargs=2,
                        help='simulation redshift boundaries',type=float)
    parser.add_argument('--sim-ra-range',default=None,nargs=2,
                        help='simulation RA range in degrees',type=float)
    parser.add_argument('--sim-dec-range',default=None,nargs=2,
                        help='simulation Dec range in degrees',type=float)
    parser.add_argument('--sim-zone-of-avoidance',default=None,nargs=1,
                        help='size of the ZoA for simulation in degrees',type=float)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sim-snrate',default=3e-5,type=float,
                       help='Volumetric supernova rate for simulation')
    group.add_argument('--sim-z-pdf',default=None,nargs='*',
                       help='simulation pdf values for non-flat distribution (if --z-pdf-bins not stated, redshift range will be split uniformly)',type=float)

    parser.add_argument('--sim-time',default=365.25,nargs=1,type=float,
                        help='observation time for sn distribution base on volumetric rate')
    parser.add_argument('--sim-area',default=None,nargs=1,type=float,
                        help='observed area for sn distribution base on volumetric rate')
    parser.add_argument('--sim-z-pdf-bins',default=None,nargs='*',
                        help='simulation redshift bins for non-flat distribution (requires --sim-z-pdf)',type=float)

    parser.add_argument('--no-cosmo-fit',action='store_true',
                        help='fit cosmology for simulated data before fitting anisotropy')
    parser.add_argument('--no-determine-sig-int',action='store_true',
                        help='redetermine sig_int when fitting cosmology for simulated data (requires --fit-cosmo)')
    

    return parser
      
def _process_args(args, parser):
    messages = ['Files:','\n'.join(args.files),'',
                'Number of realizations: {}'.format(args.number)]

    if args.redshift is None:
        args.redshift = [0.015,0.1]
        messages.append('Redshift range: {:.3f} -- {:.3f} (default)'.format(*args.redshift))
    elif args.redshift[0] > args.redshift[1]:
        raise parser.error('Invalid redshift range: {:.3f} -- {:.3f}'.format(*args.redshift))
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
        raise parser.error('Unknown signal mode: {}'.format(args.signal_mode))
    messages.append('Signal mode {}'.format(signal_modes[args.signal_mode]))

    if args.parameters is None:
        args.parameters = parameter_default[args.signal_mode]
    elif len(args.parameters) != len(parameter_default[args.signal_mode]):
        raise parser.error('Wrong number of parameters for chosen signal mode.')
    messages.append('Parameters: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                          for val in args.parameters])))

    if args.sim_redshift is None:
        # args.sim_redshift = [0.015,0.1]
        messages.append('Redshift range: {:.3f} -- {:.3f} (default)'.format(*args.sim_redshift))
        simulate = False
    elif args.sim_redshift[0] > args.sim_redshift[1]:
        raise parser.error('Invalid simulation redshift range: {:.3f} -- {:.3f}'.format(*args.sim_redshift))
    else:
        messages.append('Redshift range: {:.3f} -- {:.3f}'.format(*args.sim_redshift))
        simulate = True

    if args.sim_ra_range is None:
        args.sim_ra_range = [-180.,180.]
        if simulate:
            messages.append('RA range: {:.1f} deg -- {:.1f} deg (default)'.format(*args.sim_ra_range))
    elif (args.sim_ra_range[0] < -180 or args.sim_ra_range[1] > 180 
          or args.sim_ra_range[0] >  args.sim_ra_range[1]):
        raise parser.error('Invalid simulation RA range: {:.1f} -- {:.1f}'.format(*args.sim_ra_range))
    else:
        if simulate:
            messages.append('RA range: {:.1f} deg -- {:.1f} deg'.format(*args.sim_ra_range))

    if args.sim_dec_range is None:
        args.sim_dec_range = [-90.,90.]
        if simulate:
            messages.append('Dec range: {:.1f} deg -- {:.1f} deg (default)'.format(*args.sim_dec_range))
    elif (args.sim_dec_range[0] < -90 or args.sim_dec_range[1] > 90 
          or args.sim_dec_range[0] >  args.sim_dec_range[1]):
        raise parser.error('Invalid simulation Dec range: {:.1f} -- {:.1f}'.format(*args.sim_dec_range))
    else:
        if simulate:
            messages.append('Dec range: {:.1f} deg -- {:.1f} deg'.format(*args.sim_dec_range))

    
    if args.sim_zone_of_avoidance is None:
        args.sim_zone_of_avoidance = 10.
        if simulate:
            messages.append('ZoA size: {:.1f} deg (default)'.format(args.sim_zone_of_avoidance))
    else:
        if simulate:
            messages.append('ZoA size: {:.1f} deg'.format(args.sim_zone_of_avoidance))

        
    # if args.sim_z_pdf is None:
    #     if args.sim_z_pdf_bins is None:
    #         args.sim_z_pdf = np.ones(1)
    #         args.sim_z_pdf_bins = np.array(args.sim_redshift)
    #     else:
    #         args.sim_z_pdf_bins = np.array(args.sim_z_pdf_bins)
    #         args.sim_z_pdf = np.ones(len(args.sim_z_pdf_bins)-1)/(len(args.sim_z_pdf_bins)-1)         
    # else:
    if args.sim_z_pdf is not None:
        args.sim_z_pdf = np.array(args.sim_z_pdf)/np.sum(np.array(args.sim_z_pdf))
        if args.sim_z_pdf_bins is None:
            args.sim_z_pdf_bins = np.linspace(args.sim_redshift[0],args.sim_redshift[1],len(args.sim_z_pdf)+1)
        elif (args.sim_z_pdf_bins[0] != args.sim_redshift[0] or args.sim_z_pdf_bins[-1] != args.sim_redshift[1]
              or True in [a>b for a,b in zip(args.sim_z_pdf_bins[:-1],args.sim_z_pdf_bins[1:])]):
            raise parser.error('Invalid simulation redshift pdf bins')
        else:
            args.z_pdf_bins = np.array(args.sim_z_pdf_bins)

        if simulate:
            messages.append('Redshift pdf: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                                    for val in args.sim_z_pdf])))
            messages.append('Redshift pdf bins: [ {} ]'.format(' '.join(['{:.3f}'.format(val) 
                                                                     for val in args.sim_z_pdf_bins])))
            messages.extend(['','Number of additional SNe: {}'.format(args.sim_number)])            
    else:
        if simulate:
            messages.append('SN rate: {} Mpc^-3 yr^-1'.format(args.sim_snrate))
            if args.sim_number is not None:
                messages.extend(['','Number of additional SNe: {}'.format(args.sim_number)])
            else:
                messages.append('Observation time: {} days'.format(args.sim_time))
                if args.sim_area is not None:
                    messages.append('Observation area: {} square degrees'.format(args.sim_area))

    if args.verbosity:
        print '\n'.join(messages)    
    
    return args

def _make_outfile_name(args):
    if args.short_name is None:
        outfile = '_'.join([a.split('.')[0] for a in args.files])
    else:
        outfile = args.short_name

    outfile = '{}_s{}_{:.3f}_{:.3f}'.format(outfile,args.signal_mode,*args.redshift)
    
    outfile = '{}.pkl'.format(outfile.replace('.',''))

    return outfile

def _simulate_aniso(num_sim,names,l,b,z,v,d_l,verbosity,add,
                    options=None,**kwargs):
    if verbosity > 1:
        print
        print 'Running simulations.'

    if options is None:
        options = {'O_L':None,'w':-1,'dM':0,'H_0':_H_0,'O_M':_O_M}

    results = {
        'dipole': {},
        'dipole+shear': {},
        #'dipole+shear_trless': {},
        #'sr_90': {},
        #'sr_45': {},
        #'sr_22.5:': {},
        #'sr_nw_90': {},
        #'sr_nw_45': {},
        #'sr_nw_22.5:': {}
    }

    analysis_fcts = {'dipole': at.fit_dipole,
                     'dipole+shear': at.fit_dipole_shear,
                     'dipole+shear_trless': at.fit_dipole_shear_trless,
                     'sr_90': (lambda data,options: at.get_Q_min_max(data,options,delta=90)),
                     'sr_45': (lambda data,options: at.get_Q_min_max(data,options,delta=45)),
                     'sr_22.5:': (lambda data,options: at.get_Q_min_max(data,options,
                                                                         delta=22.5)),
                     'sr_nw_90': (lambda data,options: at.get_Q_min_max(data,options,delta=90,
                                                                        weighted=False)),
                     'sr_nw_45': (lambda data,options: at.get_Q_min_max(data,options,delta=45,
                                                                        weighted=False)),
                     'sr_nw_22.5:': (lambda data,options: at.get_Q_min_max(data,options,
                                                                           delta=22.5,
                                                                           weighted=False))}
    
    for k in xrange(num_sim):
        if verbosity > 2:
            print 'Realization {}'.format(k)
        data, options = st.simulate_data(names,l,b,z,v=v,add=add,d_l=d_l,**kwargs)
        for key in sorted(results.keys()):
            analysis = analysis_fcts[key](data,options)
            for skey,result in analysis.items():
                if skey not in results[key].keys():
                    if type(result) == list:
                        results[key][skey] = result
                    elif type(result) in [float,np.float16,np.float32,np.float64,
                                          int, np.int8,np.int16,np.int32,np.int64]:
                        results[key][skey] = np.array([result])
                    else:
                        raise TypeError('Output of analysis functions must be a dictionary of lists, integers and floats.')
                else:
                    if type(result) == list:
                        results[key][skey].extend(result)
                    elif type(result) in [float,np.float16,np.float32,np.float64,
                                          int, np.int8,np.int16,np.int32,np.int64]:
                        results[key][skey] = np.append(results[key][skey],result)
                    else:
                        raise TypeError('Output of analysis functions must be a dictionary of lists, integers and floats.')

    return results

def _save_results(results,outfile,arg_dict,verbosity=False):
    """
    """
    new = True
    if os.path.isfile(outfile):
        out = cPickle.load(file(outfile,'r'))
        checks = [out['version'] == __version__,
                  out['args']['files'] == arg_dict['files'],
                  out['args']['signal_mode'] == arg_dict['signal_mode']]
        if len(out['args']['parameters']) > 0 and len(arg_dict['parameters']) > 0: 
            checks.append((out['args']['parameters'] == arg_dict['parameters']).all())
        if False not in checks:
            new = False
        else:
            outfile = _get_conflict_file_name(outfile)
            if not checks[0]:
                warnings.warn('conflicting versions; new results saved as {}'.format(outfile))
            else:
                warnings.warn('conflicting results found; new results saved as {}'.format(outfile))
                

    if new:
        out = {'args': arg_dict, 'version': __version__}
        for key in results.keys():
            out[key] = results[key]
        cPickle.dump(out,file(outfile,'w'))
    else:
        for key,analysis in results.items():
            for skey,result in analysis.items():
                if type(result) == list:
                    out[key][skey].extend(result)
                elif type(result) == np.ndarray:
                    out[key][skey] = np.append(out[key][skey],result)
        cPickle.dump(out,file(outfile,'w'))

    if verbosity > 0:
        print
        print 'Results saved.'

def _get_conflict_file_name(outfile):
    """
    """
    outdir = '/'.join(outfile.split('/')[:-1])
    outfilename = outfile.split('/')[-1]
    filelist = os.listdir(outdir)
    previous_conflicts = [filename.split('.')[2] for filename in filelist 
                          if filename.startswith(outfilename) and len(filename.split('.')) == 3]
    if previous_conflicts:
        max_conflict = max([int(re.find('[0-9]{3}',conflict)[0]) 
                            for conflict in previous_conflicts])
    else:
        max_conflict = -1
    
    return '{}.conflict_{:03.0f}'.format(outfile,max_conflict+1)
    

def _main():
    t_start = datetime.datetime.now()

    parser = _def_parser()
    args = parser.parse_args()
    # print args
    # import sys
    # sys.exit()
    args = _process_args(args, parser)

    outfile = _make_outfile_name(args)

    options = {'O_L':None,'w':-1,'dM':0,'H_0':_H_0,'O_M':_O_M}

    if len(args.files) > 0:
        names, RA, Dec, z = st.load_from_files(*args.files,z_range=args.redshift)
        l, b = ct.radec2gcs(RA, Dec)        
        v = st.get_peculiar_velocities(args.signal_mode,args.parameters,l,b,z)
        d_l = np.array([ct.d_l(a,**options) for a in z])

        if args.verbosity > 0:
            print 
            print 'Data loaded.'
            print 'Number of SNe: {}'.format(len(z))
    else:
        names = np.array([])
        RA  = np.array([])
        Dec = np.array([])
        z = np.array([])
        l = np.array([])
        b = np.array([])
        v = np.array([])
        d_l = np.array([])

        if args.verbosity > 0:
            print 
            print 'No data loaded.'

    if args.sim_redshift is not None: 
        add = {
            'number': args.sim_number,
            'z_range': args.sim_redshift,
            'ra_range': args.sim_ra_range,
            'dec_range': args.sim_dec_range,
            'snratefunc': (lambda z: args.sim_snrate),
            'time': args.sim_time,
            'z_pdf': args.sim_z_pdf,
            'z_pdf_bins': args.sim_z_pdf_bins,
            'ZoA': args.sim_zone_of_avoidance,
            'z_limits': args.redshift,
            'signal_mode': args.signal_mode,
            'parameters': args.parameters
        }

        if args.sim_z_pdf is None and args.sim_area is None:
            add['area'] = st.covered_area(add['ZoA'],add['ra_range'],add['dec_range'])
            if args.verbosity > 0:
                print 
                print 'Covered area (MC estimate): %.2f square degrees'%add['area']
                print 'To speed up the next run please provide this number using --sim-area.'
        else:
            add['area'] = args.sim_area
    else:
        add = None

    results = _simulate_aniso(args.number, names, l, b, z, v, d_l,
                              verbosity=args.verbosity, add=add,
                              fit_cosmo=(not args.no_cosmo_fit),
                              determine_sig_int=(not args.no_determine_sig_int))
    
    arg_dict = vars(args)
    del arg_dict['number']
    _save_results(results,'{}{}'.format(args.outdir,outfile),arg_dict,verbosity=args.verbosity)

    if args.verbosity > 1:
        t_end = datetime.datetime.now()
        diff = t_end - t_start
        print 'Total running time: {}'.format(diff)

if __name__ == '__main__':
    _main()
