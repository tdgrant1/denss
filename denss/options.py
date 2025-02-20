#!/usr/bin/env python

import os, argparse
import numpy as np
import denss


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

    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="SAXS data file for input (either .dat, .fit, or .out)")
    parser.add_argument("-u", "--units", default="a", type=str, help="Angular units (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\")")
    parser.add_argument("-m", "--mode", default="SLOW", type=str, help="Mode. F(AST) sets default options to run quickly for simple particle shapes. S(LOW) useful for more complex molecules. M(EMBRANE) mode allows for negative contrast. (default SLOW)")
    parser.add_argument("-d", "--dmax", default=None, type=float, help="Estimated maximum dimension")
    parser.add_argument("-n", "--nsamples", default=None, type=int, help="Number of samples, i.e. grid points, along a single dimension. (Sets voxel size, overridden by --voxel. Best optimization with n=power of 2. Default=64)")
    parser.add_argument("-ncs", "--ncs", default=0, type=int, help="Rotational symmetry")
    parser.add_argument("-ncs_steps","--ncs_steps", default=[3000,5000,7000,9000], nargs='+', help="Space separated list of steps for applying NCS averaging (default=3000 5000 7000 9000)")
    parser.add_argument("-ncs_axis", "--ncs_axis", default="1", type=str, help="Rotational symmetry axis (options: 1, 2, or 3 corresponding to (long,medium,short) principal axes)")
    parser.add_argument("-ncs_type", "--ncs_type", default="C", type=str, help="Symmetry type, either cyclical (default) or dihedral (i.e. C or D, dihedral (Dn) adds n 2-fold perpendicular axes)")
    parser.add_argument("-s", "--steps", default=None, help="Maximum number of steps (iterations)")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
    parser.add_argument("-v", "--voxel", default=None, type=float, help="Set desired voxel size, setting resolution of map")
    parser.add_argument("-os","--oversampling", default=3., type=float, help="Sampling ratio")
    parser.add_argument("--ne", default=None, type=float, help="Number of electrons in object (default 10000, set to negative number to ignore.)")
    parser.add_argument("--seed", default=None, help="Random seed to initialize the map")
    parser.add_argument("-rc","-rc_on", "--recenter_on", dest="recenter", action="store_true", help="Recenter electron density when updating support. (default)")
    parser.add_argument("-rc_off", "--recenter_off", dest="recenter", action="store_false", help="Do not recenter electron density when updating support.")
    parser.add_argument("-rc_steps", "--recenter_steps", default=None, type=int, nargs='+', help="List of steps to recenter electron density.")
    parser.add_argument("-rc_mode", "--recenter_mode", default="com", type=str, help="Recenter based on either center of mass (com, default) or maximum density value (max)")
    parser.add_argument("-p","-p_on","--positivity_on", dest="positivity", action="store_true", help="Enforce positivity restraint inside support. (default)")
    parser.add_argument("-p_off","--positivity_off", dest="positivity", action="store_false", help="Do not enforce positivity restraint inside support.")
    parser.add_argument("-p_steps", "--positivity_steps", default=None, type=int, nargs='+', help=argparse.SUPPRESS) #help="List of steps to enforce positivity.")
    parser.add_argument("-rho", "--rho_start", default=None, type=str, help="Starting electron density map filename (for use in denss_refine.py only)")
    parser.add_argument("-support", "--support", "--support_start", dest="support_start", default=None, type=str, help=argparse.SUPPRESS) #help="Starting electron density map filename of initial support (for use in denss_refine.py only)")
    parser.add_argument("--add_noise", default=None, type=float, help="Add noise to starting density map. Uniformly distributed random density is added to each voxel, by default from 0 to 1. The argument is a scale factor to multiply this by.")
    parser.add_argument("-e","-e_on","--extrapolate_on", dest="extrapolate", action="store_true", help=argparse.SUPPRESS) # help="Extrapolate data by Porod law to high resolution limit of voxels. (default)")
    parser.add_argument("-e_off","--extrapolate_off", dest="extrapolate", action="store_false", help=argparse.SUPPRESS) #help="Do not extrapolate data by Porod law to high resolution limit of voxels.")
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
    parser.add_argument("-ec_max","--enforce_connectivity_max_features", default=1, type=int, help="Maximum number of features (i.e. disconnected blobs) allowed in support during enforce_connectivity step.")
    parser.add_argument("-cef", "--chi_end_fraction", default=0.001, type=float, help="Convergence criterion. Minimum threshold of chi2 std dev, as a fraction of the median chi2 of last 100 steps.")
    parser.add_argument("--write_xplor_format", default=False, action="store_true", help="Write out XPLOR map format (default only write MRC format).")
    parser.add_argument("--write_freq", default=100, type=int, help="How often to write out current density map (in steps, default 100).")
    parser.add_argument("--cutout_on", dest="cutout", action="store_true", help="When writing final map, cut out the particle to make smaller files.")
    parser.add_argument("--cutout_off", dest="cutout", action="store_false", help="When writing final map, do not cut out the particle to make smaller files (default).")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not display running statistics. (default False)")
    parser.add_argument("-gpu", "--gpu", dest="DENSS_GPU", action="store_true", help="Use GPU acceleration (requires CuPy). (default False)")
    parser.set_defaults(shrinkwrap=None)
    parser.set_defaults(shrinkwrap_old_method=None)
    parser.set_defaults(recenter=None)
    parser.set_defaults(positivity=None)
    parser.set_defaults(extrapolate=True)
    parser.set_defaults(enforce_connectivity=None)
    parser.set_defaults(cutout=False)
    parser.set_defaults(quiet = False)
    parser.set_defaults(DENSS_GPU = False)
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    if args.plot:
        #if plotting is enabled, try to import matplotlib
        #if import fails, set plotting to false
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            args.plot = False

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        if args.rho_start is not None:
            args.output = basename + "_refine"
        else:
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

    q, I, sigq, Ifit, file_dmax, isfit = denss.loadProfile(args.file, units=args.units)
    Iq = np.zeros((q.size,3))
    #for denss, I is actually the fit, since we want the smooth data
    #for reconstructions
    #also store the raw data for plotting
    Iq[:,0] = q
    Iq[:,1] = Ifit
    Iq[:,2] = sigq

    idx = np.where(I>0)
    qraw = np.copy(q)
    Iraw = np.copy(I)
    sigqraw = np.copy(sigq)
    idx = np.where(I>0)
    qraw = qraw[idx]
    Iraw = Iraw[idx]
    sigqraw = sigqraw[idx]

    if Iq.shape[0] < 3:
        print("Not enough data points (check that data has 3 columns: q, I, errors).")
        exit()

    Iq = denss.clean_up_data(Iq)
    is_raw_data = denss.check_if_raw_data(Iq)
    #now that we've cleaned up the data, reset the q, I, sigq arrays
    q = Iq[:,0]
    I = Iq[:,1]
    sigq = Iq[:,2]

    if args.dmax is not None and args.dmax >= 0:
        dmax = args.dmax
    elif file_dmax < 0:
        #if dmax from loadProfile() is -1, then dmax was not able
        #to be extracted from the file
        #in that case, estimate dmax directly from the data
        dmax, sasrec = denss.estimate_dmax(Iq)
    else:
        dmax = file_dmax
    D = dmax

    if is_raw_data:
        #in case a user gives raw experimental data, first, fit the data
        #using Sasrec and dmax
        sasrec = denss.Sasrec(Iq, D, alpha=0.0)
        ideal_chi2 = sasrec.calc_chi2()
        al = []
        chi2 = []
        #here, alphas are actually the exponents, since the range can
        #vary from 10^-10 upwards of 10^20. This should cover nearly all likely values
        alphas = np.arange(-10,20.)
        for alpha in alphas:
            #print("***** ALPHA ****** %.5e"%alpha)
            sasrec = denss.Sasrec(Iq, D, alpha=10.**alpha)
            r = sasrec.r
            pi = np.pi
            N = sasrec.N[:,None]
            In = sasrec.In[:,None]
            chi2value = sasrec.calc_chi2()
            al.append(alpha)
            chi2.append(chi2value)
        chi2 = np.array(chi2)

        #find optimal alpha value based on where chi2 begins to rise, to 10% above the ideal chi2 (where alpha=0)
        x = np.linspace(alphas[0],alphas[-1],1000)
        y = np.interp(x, alphas, chi2)
        chif = 2.0
        ali = np.argmin(y<=chif*ideal_chi2)
        opt_alpha = 10.0**(np.interp(chif*ideal_chi2,[y[ali+1],y[ali]],[x[ali+1],x[ali]])-1)
        alpha = opt_alpha

        sasrec = denss.Sasrec(Iq, D, alpha=alpha)
        #sasrec = denss.Sasrec(Iq, dmax, qc=None, extrapolate=False)
        #now, set the Iq values to be the new fitted q values
        q = sasrec.qc
        I = sasrec.Ic
        sigq = sasrec.Icerr

        #save fit, just like from denss_fit_data.py
        param_str = store_parameters_as_string(sasrec)
        #add column headers to param_str for output
        param_str += 'q, I, error, fit'
        #quick, interpolate the raw data, sasrec.I, to the new qc values, but be sure to 
        #put zeros in for the q values not measured behind the beamstop
        Iinterp = np.interp(sasrec.qc, sasrec.q, sasrec.I, left=0.0, right=0.0)
        np.savetxt(args.output+'_fitdata.fit', np.vstack((sasrec.qc, Iinterp, sasrec.Icerr, sasrec.Ic)).T,delimiter=' ',fmt='%.5e',header=param_str)

    #denss cannot deal with negative intensity values which sometimes happens with real data
    #if the fit goes negative, denss will fail
    #so first, remove all negative intensity values from the fit
    idx_pos = np.where(I>0)
    q = q[idx_pos]
    I = I[idx_pos]
    sigq = sigq[idx_pos]

    #allow ncs_steps to be either list of ints or string of list of ints
    if isinstance(args.ncs_steps, list):
        if len(args.ncs_steps) == 1:
            args.ncs_steps = np.fromstring(args.ncs_steps[0],sep=' ',dtype=int)
        else:
            args.ncs_steps = [int(x) for x in args.ncs_steps]

    if args.ncs_type[0].upper() == "D":
        args.ncs_type = "dihedral"
    else:
        args.ncs_type = "cyclical"

    if args.ncs_axis[0].upper() == "L" or args.ncs_axis[0] == "1":
        #if "1" or "long" or "LONG" or "L" or similar is given
        #assume the long axis, i.e. axis = 1
        args.ncs_axis = 1
    elif args.ncs_axis[0].upper() == "M" or args.ncs_axis[0] == "2":
        #if "2" or "middle" or "MIDDLE" or "M" or similar is given
        #assume the middle axis, i.e. axis = 2
        args.ncs_axis = 2
    elif args.ncs_axis[0].upper() == "S" or args.ncs_axis[0] == "3":
        #if "3" or "short" or "SHORT" or "S" or similar is given
        #assume the short axis, i.e. axis = 3
        args.ncs_axis = 3
    else:
        #default to long axis
        args.ncs_axis = 1

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
            shrinkwrap_minstep = 0
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,2502,500))
        elif args.mode[0].upper() == "S":
            args.mode = "SLOW"
            nsamples = 64
            shrinkwrap_minstep = 0
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
            shrinkwrap_minstep = 0
            ec_steps_to_add = np.array([1000])
            enforce_connectivity_steps = shrinkwrap_minstep + ec_steps_to_add
            recenter_steps = list(range(501,2502,500))
        elif args.mode[0].upper() == "S":
            args.mode = "SLOW"
            nsamples = 64
            shrinkwrap_minstep = 0
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

    if args.positivity_steps is not None:
        print("The positivity_steps option is currently not allowed.")
        # if not isinstance(args.positivity_steps, list):
        #     args.positivity_steps = [ positivity_steps ]

    if args.steps is not None:
        steps = args.steps

    oversampling = args.oversampling
    side = dmax * oversampling

    #allow user to give initial density map for denss_refine.py
    if args.rho_start is not None:
        rho_start, rho_side = denss.read_mrc(args.rho_start)
        rho_nsamples = rho_start.shape[0]
        rho_voxel = rho_side/rho_nsamples

        if (not np.isclose(rho_side, side) or
            not np.isclose(rho_voxel, voxel) or
            not np.isclose(rho_nsamples, nsamples)):
            print("rho_start density dimensions do not match given options.")
            print("Oversampling and voxel size adjusted to match rho_start dimensions.")
            side = rho_side
            voxel = rho_voxel
            oversampling = side/dmax
            nsamples = rho_nsamples
        #set args.rho_start to the actual density map, rather than the filename
        args.rho_start = rho_start
        if args.recenter is None:
            args.recenter = False
        if args.shrinkwrap_old_method is None:
            shrinkwrap_old_method = True
        else:
            shrinkwrap_old_method = args.shrinkwrap_old_method

    #allow user to give initial support, check for consistency with given grid parameters
    if args.support_start is not None:
        support_start, support_side = denss.read_mrc(args.support_start)
        support_nsamples = support_start.shape[0]
        support_voxel = support_side/support_nsamples

        if (not np.isclose(support_side, side) or
            not np.isclose(support_voxel, voxel) or
            not np.isclose(support_nsamples, nsamples)):
            print("Support dimensions do not match given options.")
            print("Ignoring support.")
            print("side (support, given): ", support_side, side)
            print("voxel (support, given): ", support_voxel, voxel)
            print("n (support, given): ", support_nsamples, nsamples)
            args.support_start = None
        else:
            args.support_start = support_start.astype(bool)

    if args.shrinkwrap is None:
        if args.support_start is not None:
            #assume if a user gives a support, that they do not want to
            #run shrinkwrap. However, allow the user to enable shrinkwrap explicitly
            #by setting the shrinkwrap option on
            args.shrinkwrap = False
            if args.enforce_connectivity is None:
                args.enforce_connectivity = False
            if args.recenter is None:
                args.recenter = False
        else:
            args.shrinkwrap = True

    if args.enforce_connectivity is None:
        args.enforce_connectivity = True
    if args.recenter is None:
        args.recenter = True
    if args.shrinkwrap_old_method is None:
        args.shrinkwrap_old_method = False

    if args.ne is not None:
        if args.ne <= 0.0:
            #if args.ne is negative, then assume that the user
            #does not want to set the number of electrons, and just
            #set it equal to the square root of the forward scattering I(0)
            args.ne = I[0]**0.5
    elif args.rho_start is not None:
        #if args.ne is not given, and args.rho_start is given,
        #then set ne to be the sum of the given density map
        dV = voxel**3
        args.ne = args.rho_start.sum() * dV
    else:
        #default to 10,000
        args.ne = 10000.0

    #now recollect all the edited options back into args
    args.nsamples = nsamples
    args.positivity = positivity
    args.shrinkwrap_minstep = shrinkwrap_minstep
    args.shrinkwrap_threshold_fraction = shrinkwrap_threshold_fraction
    args.shrinkwrap_sigma_start = shrinkwrap_sigma_start_in_vox
    args.shrinkwrap_sigma_end = shrinkwrap_sigma_end_in_vox
    args.enforce_connectivity_steps = enforce_connectivity_steps
    args.recenter_steps = recenter_steps
    args.oversampling = oversampling
    args.dmax = dmax
    args.voxel = voxel
    args.q = q
    args.I = I
    args.sigq = sigq
    args.qraw = qraw
    args.Iraw = Iraw
    args.sigqraw = sigqraw

    return args
