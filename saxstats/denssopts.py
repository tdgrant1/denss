#!/usr/bin/env python

from ._version import __version__
import os, argparse
import imp
try:
    imp.find_module('matplotlib')
    matplotlib_found = True
    import matplotlib.pyplot as plt
    from  matplotlib.colors import colorConverter as cc
    import matplotlib.gridspec as gridspec
except ImportError:
    matplotlib_found = False

import numpy as np
from . import saxstats as saxs

def store_parameters_as_string(sasrec):
    param_str = ("Parameter Values:\n"
    "Dmax  = {dmax:.5e}\n"
    "alpha = {alpha:.5e}\n"
    "Rg    = {rg:.5e} +- {rgerr:.5e}\n"
    "I(0)  = {I0:.5e} +- {I0err:.5e}\n"
    "Vp    = {Vp:.5e} +- {Vperr:.5e}\n"
    "MW_Vp = {mwVp:.5e} +- {mwVperr:.5e}\n"
    "MW_Vc = {mwVc:.5e} +- {mwVcerr:.5e}\n"
    "Lc    = {lc:.5e} +- {lcerr:.5e}\n"
    ).format(dmax=sasrec.D,alpha=sasrec.alpha,rg=sasrec.rg,rgerr=sasrec.rgerr,
        I0=sasrec.I0,I0err=sasrec.I0err,Vp=sasrec.Vp,Vperr=sasrec.Vperr,
        mwVp=sasrec.mwVp,mwVperr=sasrec.mwVperr,mwVc=sasrec.mwVc,mwVcerr=sasrec.mwVcerr,
        lc=sasrec.lc,lcerr=sasrec.lcerr)
    return param_str

def parse_arguments(parser):

    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
    parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat, .fit, or .out)")
    parser.add_argument("-u", "--units", default="a", type=str, help="Angular units (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\")")
    parser.add_argument("-m", "--mode", default="SLOW", type=str, help="Mode. F(AST) sets default options to run quickly for simple particle shapes. S(LOW) useful for more complex molecules. M(EMBRANE) mode allows for negative contrast. (default SLOW)")
    parser.add_argument("-d", "--dmax", default=None, type=float, help="Estimated maximum dimension")
    parser.add_argument("-n", "--nsamples", default=None, type=int, help="Number of samples, i.e. grid points, along a single dimension. (Sets voxel size, overridden by --voxel. Best optimization with n=power of 2. Default=64)")
    parser.add_argument("-ncs", "--ncs", default=0, type=int, help="Rotational symmetry")
    parser.add_argument("-ncs_steps","--ncs_steps", default=[3000,5000,7000,9000], nargs='+', help="List of steps for applying NCS averaging (default=3000,5000,7000,9000)")
    parser.add_argument("-ncs_axis", "--ncs_axis", default=1, type=int, help="Rotational symmetry axis (options: 1, 2, or 3 corresponding to (long,medium,short) principal axes)")
    parser.add_argument("-s", "--steps", default=None, help="Maximum number of steps (iterations)")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
    parser.add_argument("-v", "--voxel", default=None, type=float, help="Set desired voxel size, setting resolution of map")
    parser.add_argument("-os","--oversampling", default=3., type=float, help="Sampling ratio")
    parser.add_argument("--ne", default=10000, type=float, help="Number of electrons in object")
    parser.add_argument("--seed", default=None, help="Random seed to initialize the map")
    parser.add_argument("-ld_on","-ld_on","--limit_dmax_on", dest="limit_dmax", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-ld_off","--limit_dmax_off", dest="limit_dmax", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("-ld_steps","--limit_dmax_steps", default=None, type=int, nargs='+', help=argparse.SUPPRESS)
    parser.add_argument("-rc","-rc_on", "--recenter_on", dest="recenter", action="store_true", help="Recenter electron density when updating support. (default)")
    parser.add_argument("-rc_off", "--recenter_off", dest="recenter", action="store_false", help="Do not recenter electron density when updating support.")
    parser.add_argument("-rc_steps", "--recenter_steps", default=None, type=int, nargs='+', help="List of steps to recenter electron density.")
    parser.add_argument("-rc_mode", "--recenter_mode", default="com", type=str, help="Recenter based on either center of mass (com, default) or maximum density value (max)")
    parser.add_argument("-p","-p_on","--positivity_on", dest="positivity", action="store_true", help="Enforce positivity restraint inside support. (default)")
    parser.add_argument("-p_off","--positivity_off", dest="positivity", action="store_false", help="Do not enforce positivity restraint inside support.")
    parser.add_argument("-min","--minimum_density", default=None, type=float, help="Minimum density value in e-/angstrom^3 (must also set --ne to be meaningful)")
    parser.add_argument("-max","--maximum_density", default=None, type=float, help="Maximum density value in e-/angstrom^3 (must also set --ne to be meaningful)")
    parser.add_argument("-rho", "--rho_start", default=None, type=str, help="Starting electron density map filename (for use in denss.refine.py only)")
    parser.add_argument("--add_noise", default=None, type=float, help="Add noise to starting density map. Uniformly distributed random density is added to each voxel, by default from 0 to 1. The argument is a scale factor to multiply this by.")
    parser.add_argument("-e","-e_on","--extrapolate_on", dest="extrapolate", action="store_true", help="Extrapolate data by Porod law to high resolution limit of voxels. (default)")
    parser.add_argument("-e_off","--extrapolate_off", dest="extrapolate", action="store_false", help="Do not extrapolate data by Porod law to high resolution limit of voxels.")
    parser.add_argument("-sw","-sw_on","--shrinkwrap_on", dest="shrinkwrap", action="store_true", help="Turn shrinkwrap on (default)")
    parser.add_argument("-sw_off","--shrinkwrap_off", dest="shrinkwrap", action="store_false", help="Turn shrinkwrap off")
    parser.add_argument("-sw_om","-sw_old","-sw_om_on","--shrinkwrap_old_method_on", dest="shrinkwrap_old_method", action="store_true", help="Use the old method of shrinkwrap.")
    parser.add_argument("-sw_om_off","--shrinkwrap_old_method_off", dest="shrinkwrap_old_method", action="store_false", help="Use the new method of shrinkwrap (default).")
    parser.add_argument("-sw_start","--shrinkwrap_sigma_start_in_A", default=None, type=float, help="Starting sigma for Gaussian blurring, in angstroms")
    parser.add_argument("-sw_end","--shrinkwrap_sigma_end_in_A", default=None, type=float, help="Ending sigma for Gaussian blurring, in angstroms")
    parser.add_argument("-sw_start_vox","--shrinkwrap_sigma_start_in_vox", default=None, type=float, help="Starting sigma for Gaussian blurring, in voxels")
    parser.add_argument("-sw_end_vox","--shrinkwrap_sigma_end_in_vox", default=None, type=float, help="Ending sigma for Gaussian blurring, in voxels")
    parser.add_argument("-sw_decay","--shrinkwrap_sigma_decay", default=0.99, type=float, help="Rate of decay of sigma, fraction")
    parser.add_argument("-sw_threshold","--shrinkwrap_threshold_fraction", default=None, type=float, help="Minimum threshold defining support, in fraction of maximum density")
    parser.add_argument("-sw_iter","--shrinkwrap_iter", default=20, type=int, help="Number of iterations between updating support with shrinkwrap")
    parser.add_argument("-sw_minstep","--shrinkwrap_minstep", default=None, type=int, help="First step to begin shrinkwrap")
    parser.add_argument("-ec","-ec_on","--enforce_connectivity_on", dest="enforce_connectivity", action="store_true", help="Enforce connectivity of support, i.e. remove extra blobs (default)")
    parser.add_argument("-ec_off","--enforce_connectivity_off", dest="enforce_connectivity", action="store_false", help="Do not enforce connectivity of support")
    parser.add_argument("-ec_steps","--enforce_connectivity_steps", default=None, type=int, nargs='+', help="List of steps to enforce connectivity")
    parser.add_argument("-cef", "--chi_end_fraction", default=0.001, type=float, help="Convergence criterion. Minimum threshold of chi2 std dev, as a fraction of the median chi2 of last 100 steps.")
    parser.add_argument("--write_xplor_format", default=False, action="store_true", help="Write out XPLOR map format (default only write MRC format).")
    parser.add_argument("--write_freq", default=100, type=int, help="How often to write out current density map (in steps, default 100).")
    parser.add_argument("--cutout_on", dest="cutout", action="store_true", help="When writing final map, cut out the particle to make smaller files.")
    parser.add_argument("--cutout_off", dest="cutout", action="store_false", help="When writing final map, do not cut out the particle to make smaller files (default).")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not display running statistics. (default False)")
    parser.add_argument("-gpu", "--gpu", dest="DENSS_GPU", action="store_true", help="Use GPU acceleration (requires CuPy). (default False)")
    parser.set_defaults(limit_dmax=False)
    parser.set_defaults(shrinkwrap=True)
    parser.set_defaults(shrinkwrap_old_method=False)
    parser.set_defaults(recenter=True)
    parser.set_defaults(positivity=None)
    parser.set_defaults(extrapolate=True)
    parser.set_defaults(enforce_connectivity=True)
    parser.set_defaults(cutout=False)
    parser.set_defaults(quiet = False)
    parser.set_defaults(DENSS_GPU = False)
    if matplotlib_found:
        parser.set_defaults(plot=True)
    else:
        parser.set_defaults(plot=False)
    args = parser.parse_args()

    if args.output is None:
        basename, ext = os.path.splitext(args.file)
        args.output = basename
    else:
        args.output = args.output

    #A bug appears to be present when disabling the porod extrapolation.
    #for now that option will be disabled until I come up with a fix
    if args.extrapolate is False:
        print ("There is currently a bug when disabling the Porod "
               "extrapolation (the -e_off option). \n For now, extrapolation "
               "has been re-enabled until a bug fix is released. ")
        args.extrapolate = True

    q, I, sigq, Ifit, file_dmax, isout = saxs.loadProfile(args.file, units=args.units)
    Iq = np.zeros((q.size,3))
    #for denss, I is actually the fit, since we want the smooth data
    #for reconstructions
    #also store the raw data for plotting
    Iq[:,0] = q
    Iq[:,1] = Ifit
    Iq[:,2] = sigq
    qraw = np.copy(q)
    Iraw = np.copy(I)

    Iq = saxs.clean_up_data(Iq)
    is_raw_data = saxs.check_if_raw_data(Iq)
    #now that we've cleaned up the data, reset the q, I, sigq arrays
    q = Iq[:,0]
    I = Iq[:,1]
    sigq = Iq[:,2]

    if args.dmax is not None and args.dmax >= 0:
        dmax = args.dmax
    elif file_dmax == -1:
        #if dmax from loadProfile() is -1, then dmax was not able
        #to be extracted from the file
        #in that case, estimate dmax directly from the data
        dmax, sasrec = saxs.estimate_dmax(Iq)
    else:
        dmax = file_dmax

    if is_raw_data:
        #in case a user gives raw experimental data, first, fit the data
        #using Sasrec and dmax
        #create a calculated q range for Sasrec
        qmin = np.min(q)
        qmax = np.max(q)
        dq = (qmax-qmin)/(q.size-1)
        nq = int(qmin/dq)
        qc = np.concatenate(([0.0],np.arange(nq)*dq+(qmin-nq*dq),q))
        sasrec = saxs.Sasrec(Iq, dmax, qc=qc)
        #now, set the Iq values to be the new fitted q values
        q = sasrec.qc
        I = sasrec.Ic
        sigq = sasrec.Icerr

        #save fit, just like from denss.fit_data.py
        param_str = store_parameters_as_string(sasrec)
        #add column headers to param_str for output
        param_str += 'q, I, error, fit'
        #quick, interpolate the raw data, sasrec.I, to the new qc values, but be sure to 
        #put zeros in for the q values not measured behind the beamstop
        Iinterp = np.interp(sasrec.qc, sasrec.q, sasrec.I, left=0.0, right=0.0)
        np.savetxt(args.output+'.fit', np.vstack((sasrec.qc, Iinterp, sasrec.Icerr, sasrec.Ic)).T,delimiter=' ',fmt='%.5e',header=param_str)

    #allow ncs_steps to be either list of ints or string of list of ints
    if isinstance(args.ncs_steps, list):
        if len(args.ncs_steps) == 1:
            args.ncs_steps = np.fromstring(args.ncs_steps[0],sep=' ',dtype=int)
        else:
            args.ncs_steps = [int(x) for x in args.ncs_steps]

    #old default sw_start was 3.0
    #however, in cases where the voxel size is smaller than default,
    #e.g. increasing nsamples, shrinkwrap works differently since sigma
    #is then smaller (in angstroms). We will instead base sigma on
    #physical dimension in angstroms, rather than number of voxels.
    #We will do this by basing sigma on particle size (dmax), rather than 
    #resolution of the grid. 
    #since defaults work well, we'll start there. The parameters
    #defining voxel size are the grid size / nsamples (dmax*oversampling/nsamples)
    #so define sw_start as (voxsize*3) = (3*D/64) * 3, since those are the defaults that work well
    #later we need to convert this physical size into a voxel size (since 
    #ndimage.gaussian_filter works based on number of pixels)

    #some defaults:
    shrinkwrap_sigma_start_in_A = (3.0 * dmax / 64.0) * 3.0
    shrinkwrap_sigma_end_in_A = (3.0 * dmax / 64.0) * 1.5
    shrinkwrap_threshold_fraction = 0.2
    positivity = True

    if args.shrinkwrap_old_method:
        #for FAST or SLOW modes, set some default values for a few options
        if args.mode[0].upper() == "F":
            args.mode = "FAST"
            nsamples = 32
            shrinkwrap_minstep = 1000
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,2502,500))
        elif args.mode[0].upper() == "S":
            args.mode = "SLOW"
            nsamples = 64
            shrinkwrap_minstep = 5000
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,8002,500))
        elif args.mode[0].upper() == "M":
            args.mode = "MEMBRANE"
            nsamples = 64
            positivity = False
            shrinkwrap_minstep = 0
            ec_steps_to_add = np.array([300])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,8002,500))
        else:
            args.mode = "None"
    else:
        #for FAST or SLOW modes, set some default values for a few options
        if args.mode[0].upper() == "F":
            args.mode = "FAST"
            nsamples = 32
            shrinkwrap_minstep = 1000
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,2502,500))
        elif args.mode[0].upper() == "S":
            args.mode = "SLOW"
            nsamples = 64
            shrinkwrap_minstep = 1000
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,8002,500))
        elif args.mode[0].upper() == "M":
            args.mode = "MEMBRANE"
            nsamples = 64
            positivity = False
            shrinkwrap_minstep = 0
            shrinkwrap_sigma_start_in_A *= 2.0
            shrinkwrap_sigma_end_in_A *= 2.0
            ec_steps_to_add = np.array([300, 500, 1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,8002,500))
        else:
            args.mode = "None"

    #allow user to explicitly modify those values by resetting them here to the user defined values
    if args.nsamples is not None:
        nsamples = args.nsamples

    if args.voxel is None and nsamples is None:
        voxel = 5.
    elif args.voxel is None and nsamples is not None:
        voxel = dmax * args.oversampling / nsamples
    else:
        voxel = args.voxel

    if args.positivity is not None:
        positivity = args.positivity

    if args.shrinkwrap_minstep is not None:
        shrinkwrap_minstep = args.shrinkwrap_minstep
        #adjust ec_steps if minstep is given by user by default
        #ec_steps will be overwritten later if user defined
        enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add

    if args.shrinkwrap_threshold_fraction is not None:
        shrinkwrap_threshold_fraction = args.shrinkwrap_threshold_fraction

    #allow user to input sigma as angstroms or voxels
    if args.shrinkwrap_sigma_start_in_A is not None:
        shrinkwrap_sigma_start_in_A = args.shrinkwrap_sigma_start_in_A
    elif args.shrinkwrap_sigma_start_in_vox is not None:
        #if voxel option is set, for now convert to angstroms, 
        #then later convert everything back to voxels
        shrinkwrap_sigma_start_in_A = args.shrinkwrap_sigma_start_in_vox * voxel

    if args.shrinkwrap_sigma_end_in_A is not None:
        shrinkwrap_sigma_end_in_A = args.shrinkwrap_sigma_end_in_A
    elif args.shrinkwrap_sigma_end_in_vox is not None:
        shrinkwrap_sigma_end_in_A = args.shrinkwrap_sigma_end_in_vox * voxel

    #as mentioned above, now that we have the voxel size, we need to convert
    #the shrinkwrap sigma values to voxels, rather than physical distance
    shrinkwrap_sigma_start_in_vox = shrinkwrap_sigma_start_in_A / voxel
    shrinkwrap_sigma_end_in_vox = shrinkwrap_sigma_end_in_A / voxel

    if args.enforce_connectivity_steps is not None:
        enforce_connectivity_steps = args.enforce_connectivity_steps
    if not isinstance(enforce_connectivity_steps, np.ndarray):
        enforce_connectivity_steps = np.asarray(enforce_connectivity_steps)

    if args.recenter_steps is not None:
        recenter_steps = args.recenter_steps
    if not isinstance(recenter_steps, list):
        recenter_steps = [ recenter_steps ]

    if args.limit_dmax_steps is not None:
        limit_dmax_steps = args.limit_dmax_steps
    else:
        limit_dmax_steps = [502]
    if not isinstance(limit_dmax_steps, list):
        limit_dmax_steps = [ limit_dmax_steps ]

    if args.steps is not None:
        steps = args.steps

    #now recollect all the edited options back into args
    args.nsamples = nsamples
    args.positivity = positivity
    args.shrinkwrap_minstep = shrinkwrap_minstep
    args.shrinkwrap_threshold_fraction = shrinkwrap_threshold_fraction
    args.shrinkwrap_sigma_start = shrinkwrap_sigma_start_in_vox
    args.shrinkwrap_sigma_end = shrinkwrap_sigma_end_in_vox
    args.enforce_connectivity_steps = enforce_connectivity_steps
    args.recenter_steps = recenter_steps
    args.limit_dmax_steps = limit_dmax_steps
    args.dmax = dmax
    args.voxel = voxel
    args.q = q
    args.I = I
    args.sigq = sigq
    args.qraw = qraw
    args.Iraw = Iraw

    return args
