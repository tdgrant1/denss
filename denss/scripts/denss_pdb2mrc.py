#!/usr/bin/env python
#
#    denss_pdb2mrc.py
#    A tool for calculating simple electron density maps from pdb files.
#
#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tdgrant@buffalo.edu>
#    Copyright 2017-Present The Research Foundation for SUNY
#
#    Additional Authors:
#    Sarah Chamberlain
#    Stephen Moore
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function

import time


import denss

import numpy as np
from scipy import ndimage
import sys, argparse, os, logging
import copy
import time
from textwrap import wrap

def main():
    t = []
    fs = []
    t.append(time.time())
    fs.append("start")

    parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input (required).")
    parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
    parser.add_argument("--fast", dest="fast", action="store_true", help="Fast mode. Sets nsamples to 64 (increases voxel size), disables plotting, disables mrc writing, sets shell type to uniform.")
    parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use of experimental data")
    parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use of experimental data")
    parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
    parser.add_argument("-qmin", "--qmin", default=None, type=float, help="Minimum q value to use for fitting experimental data.")
    parser.add_argument("-qmax", "--qmax", default=None, type=float, help="Maximum q value to use for fitting experimental data.")
    parser.add_argument("-nq", "--nq", default=None, type=int, help="Number of q values to include in final scattering profile (only relevant for prediction with no experimental data, uses interpolation).")
    parser.add_argument("-qfile", "--qfile", default=None, type=str, help="Filename of ASCII text file containing desired q values to write as the first column (only relevant for prediction with no experimental data, uses interpolation).")
    parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=3*Dmax).")
    parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=1.0)")
    parser.add_argument("-n", "--nsamples", default=None, type=int, help="Desired number of samples (i.e. voxels) per axis (default=variable)")
    parser.add_argument("-b", "--use_b", dest="use_b", action="store_true", help="Include B-factors from atomic model (optional, default=False)")
    parser.add_argument("-B", "--global_B", default=None, type=float, help="Desired global B-factor (added to any individual B-factors if enabled))")
    parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (after scattering profile calculation, blur map by this width using gaussian kernel, similar to Chimera Volume Filter function).")
    parser.add_argument("-exH","--explicitH", dest="explicitH", action="store_true", help="Use hydrogens in pdb file (optional, default=True if H exists)")
    parser.add_argument("-imH","--implicitH", dest="explicitH", action="store_false", help=argparse.SUPPRESS) #help="Use implicit hydrogens approximation (optional, EXPERIMENTAL)")
    parser.add_argument("-recalc","--recalc","--recalculate_volumes", dest="recalculate_atomic_volumes", action="store_true", help="Calculate atomic volumes directly from coordinates rather than using lookup table (default=False)")
    parser.add_argument("--read_radii", default=False, action="store_true", help="Read adjusted per atom radii (for volume calculation) from Occupancy column of PDB file (default=False)")
    parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
    parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
    parser.add_argument("-iw_on", "--iw_on", "--ignore_waters", "--ignore_waters_on", dest="ignore_waters", action="store_true", help="Ignore waters (default=True).")
    parser.add_argument("-iw_off", "--iw_off", "--ignore_waters_off", dest="ignore_waters", action="store_false", help="Turn Ignore waters off (i.e., read the waters).")
    parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
    parser.add_argument("-exvol_type", "--exvol_type", default="gaussian", type=str, help="Type of excluded volume (gaussian (default) or flat)")
    parser.add_argument("-fit_off", "--fit_off", dest="fit_all", action="store_false", help="Do not fit either rho0 or shell contrast (optional)")
    parser.add_argument("-fit_rho0_on", "--fit_rho0_on", dest="fit_rho0", action="store_true", help="Fit rho0, the bulk solvent density (optional, default=True)")
    parser.add_argument("-fit_rho0_off", "--fit_rho0_off", dest="fit_rho0", action="store_false", help="Do not fit rho0, the bulk solvent density (optional, default=True)")
    parser.add_argument("-fit_shell_on", "--fit_shell_on", dest="fit_shell", action="store_true", help="Fit hydration shell parameters (optional, default=True)")
    parser.add_argument("-fit_shell_off", "--fit_shell_off", dest="fit_shell", action="store_false", help="Do not fit hydration shell parameters (optional, default=True)")
    parser.add_argument("-drho","--drho","-shell","-shell_contrast", "--shell_contrast", dest="shell_contrast", default=0.011, type=float, help="Initial mean contrast of hydration shell in e-/A^3 (default=0.011)")
    parser.add_argument("-shell_type", "--shell_type", default=None, type=str, help="Type of hydration shell (water (default) or uniform)")
    parser.add_argument("-shell_mrcfile", "--shell_mrcfile", default=None, type=str, help=argparse.SUPPRESS) #help="Filename of hydration shell mrc file (default=None)")
    parser.add_argument("-fit_radii", "--fit_radii", dest="fit_radii", action="store_true", help=argparse.SUPPRESS) #help="Fit atomic radii for excluded volume calculation (optional, default=False)")
    parser.add_argument("--radii_sf", default=None, type=float, nargs='+', help=argparse.SUPPRESS) #help="Atomic radii scale factors for excluded volume calculation (optional, ordered as [H C N O])")
    parser.add_argument("--set_radii_explicitly", default=None, type=str, help=argparse.SUPPRESS) #help="Set atomic radii explicitly for different atom types. Ignores --radii_sf. Format H:1.07:C:1.58:N:0.084:O:1.30"
    parser.add_argument("-fit_scale_on", "--fit_scale_on", dest="fit_scale", action="store_true", help="Include scale factor in least squares fit to data (optional, default=True)")
    parser.add_argument("-fit_scale_off", "--fit_scale_off", dest="fit_scale", action="store_false", help="Do not include offset in least squares fit to data.")
    parser.add_argument("-fit_offset_on", "--fit_offset_on", dest="fit_offset", action="store_true", help="Include offset in least squares fit to data (optional, default=False)")
    parser.add_argument("-fit_offset_off", "--fit_offset_off", dest="fit_offset", action="store_false", help="Do not include offset in least squares fit to data.")
    parser.add_argument("-p", "-penalty_weight", "--penalty_weight", default=None, type=float, help="Overall penalty weight for fitting parameters (default=0)")
    parser.add_argument("-ps", "-penalty_weights", "--penalty_weights", default=[1.0, 0.01], type=float, nargs='+', help="Individual penalty weights for each parameter (space separated listed of weights for [rho0,shell], default=1.0 0.01)")
    parser.add_argument("-min_method", "--min_method", "-minimization_method","--minimization_method", dest="method", default='Nelder-Mead', type=str, help="Minimization method (scipy.optimize method, default=Nelder-Mead).")
    parser.add_argument("-min_options", "--min_options", "-minimization_options","--minimization_options", dest="minopts", default='{"adaptive": True}', type=str, help="Minimization options (scipy.optimize options formatted as python dictionary, default=\"{'adaptive': True}\").")
    parser.add_argument("-write_on", "--write_on", action="store_true", dest="write_mrc_file", help="Write MRC file (default=True).")
    parser.add_argument("-write_off", "--write_off", action="store_false", dest="write_mrc_file", help="Do not write MRC file.")
    parser.add_argument("-write_extras", "--write_extras", action="store_true", default=False, help="Write out extra MRC files for invacuo, exvol, shell densities (default=False).")
    parser.add_argument("-write_pdb_on", "--write_pdb_on", action="store_true", dest="write_pdb", help="Write modified pdb file including recentering and atom radii to B-factor column (default=True).")
    parser.add_argument("-write_pdb_off", "--write_pdb_off", action="store_false", dest="write_pdb", help="Do not write modified pdb file.")
    parser.add_argument("-interp_on", "--interp_on", dest="Icalc_interpolation", action="store_true", help="Interpolate I_calc to experimental q grid (default).")
    parser.add_argument("-interp_off", "--interp_off", dest="Icalc_interpolation", action="store_false", help="Do not interpolate I_calc to experimental q grid .")
    parser.add_argument("--use_sasrec_during_fitting", default=False, action="store_true", help="Use Sasrec during fitting procedure for interpolation (more accurate, slower. Note: Sasrec is always used for the output .dat or .fit file, this is just for during the iterative fitting procedure.).")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
    parser.add_argument("--print_timings", default=False, action="store_true", help="Print timings for each step of the script.")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
    parser.add_argument("--write_shannon", dest="write_shannon", action="store_true", help=argparse.SUPPRESS) # help="Write a file containing only the Shannon intensities.")
    parser.set_defaults(fast = False)
    parser.set_defaults(ignore_waters = True)
    parser.set_defaults(center = True)
    parser.set_defaults(plot=None)
    parser.set_defaults(use_b=False)
    parser.set_defaults(explicitH=None)
    parser.set_defaults(recalculate_atomic_volumes=False)
    parser.set_defaults(fit_rho0=True)
    parser.set_defaults(fit_shell=True)
    parser.set_defaults(fit_all=True)
    parser.set_defaults(fit_radii=False)
    parser.set_defaults(fit_scale=True)
    parser.set_defaults(fit_offset=False)
    parser.set_defaults(Icalc_interpolation=True)
    parser.set_defaults(write_mrc_file=None)
    parser.set_defaults(write_pdb=None)
    parser.set_defaults(write_shannon=False)
    args = parser.parse_args()

    np.set_printoptions(linewidth=150,precision=10)

    t.append(time.time())
    fs.append("argparse")

    start = time.time()

    scriptname = os.path.basename(sys.argv[0])
    command = scriptname + ' ' + ' '.join(sys.argv[1:])

    fname_nopath = os.path.basename(args.file)
    basename, ext = os.path.splitext(fname_nopath)

    if args.output is None:
        output = basename + "_pdb"
    else:
        output = args.output

    logging.basicConfig(filename=output+'.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s') #, datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Command: %s', ' '.join(sys.argv))
    logging.info('DENSS Version: %s', denss.__version__)
    # logging.info('PDB filename: %s', args.file)

    t.append(time.time())
    fs.append("init log")

    pdb = denss.PDB(args.file, ignore_waters=args.ignore_waters)

    t.append(time.time())
    fs.append("read pdb")

    if args.read_radii:
        #if the file has unique radius in occupancy column, use it
        pdb.unique_radius = pdb.occupancy
        pdb.unique_volume = 4/3*np.pi*pdb.unique_radius**3

    #if pdb contains an H atomtype, set explicit hydrogens to True
    if args.explicitH is None and np.any(np.core.defchararray.find(pdb.atomtype,"H")!=-1):
        #if explicitH not set, and there are hydrogens, set explicitH to true
        args.explicitH = True
    elif args.explicitH is None:
        #if explicitH not set, and there are no hydrogens, set explicitH to false, to use implicit hydrogens
        args.explicitH = False
        print("#"*90)
        print("#  WARNING: use of implicit hydrogens is an experimental feature.                        #")
        print("#  Recommend adding explicit hydrogens using a tool such as Reduce, CHARMM, PyMOL, etc.  #")
        print("#"*90)

    if args.fast:
        # only set these values in fast mode if they aren't explicitly set in the command line
        if args.nsamples is None:
            args.nsamples = 64
        if args.write_mrc_file is None:
            args.write_mrc_file = False
        if args.shell_type is None:
            args.shell_type = "uniform"
        if args.plot is None:
            args.plot = False
        if args.write_pdb is None:
            args.write_pdb = False
    else:
        if args.write_mrc_file is None:
            args.write_mrc_file = True
        if args.plot is None:
            args.plot = True
        if args.write_pdb is None:
            args.write_pdb = True

    if args.plot:
        # if plotting is enabled, try to import matplotlib
        # if import fails, set plotting to false
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError as e:
            print("matplotlib import failed.")
            args.plot = False

    t.append(time.time())
    fs.append("import matplotlib")

    pdb2mrc = denss.PDB2MRC(
        pdb=pdb,
        ignore_waters=args.ignore_waters,
        explicitH=args.explicitH,
        modifiable_atom_types=None,
        center_coords=args.center,
        radii_sf=args.radii_sf,
        recalculate_atomic_volumes=args.recalculate_atomic_volumes,
        exvol_type=args.exvol_type,
        use_b=args.use_b,
        global_B=args.global_B,
        resolution=args.resolution,
        voxel=args.voxel,
        side=args.side,
        nsamples=args.nsamples,
        rho0=args.rho0,
        shell_contrast=args.shell_contrast,
        shell_mrcfile=args.shell_mrcfile,
        shell_type=args.shell_type,
        Icalc_interpolation=args.Icalc_interpolation,
        fit_scale=args.fit_scale,
        fit_offset=args.fit_offset,
        data_filename=args.data,
        data_units=args.units,
        n1=args.n1,
        n2=args.n2,
        qmin=args.qmin,
        qmax=args.qmax,
        penalty_weight=args.penalty_weight,
        penalty_weights=args.penalty_weights,
        fit_rho0=args.fit_rho0,
        fit_shell=args.fit_shell,
        fit_all=args.fit_all,
        min_method=args.method,
        min_opts=args.minopts,
        fast=args.fast,
        use_sasrec_during_fitting=args.use_sasrec_during_fitting,
        )

    t.append(time.time())
    fs.append("pdb2mrc init")

    #write the modified pdb file, and store the 
    #new unique volume/radius value in the occupancy
    #column, to prevent needing to recalculate if wanting
    #to run the fitting again.
    pdboutput = basename
    if not args.read_radii:
        pdboutput += '_out.pdb'
        pdbout = copy.deepcopy(pdb2mrc.pdb)
        pdbout.occupancy = pdb2mrc.pdb.unique_radius
        if args.write_pdb:
            pdbout.write(filename=pdboutput)

    t.append(time.time())
    fs.append("log stuff")

    pdb2mrc.scale_radii(radii_sf=args.radii_sf)
    t.append(time.time())
    fs.append("scale_radii")

    if args.set_radii_explicitly is not None:
        arg_list = args.set_radii_explicitly.split(":")
        atom_types = arg_list[::2]
        radii = np.array(arg_list[1::2], dtype=float)
        pdb2mrc.set_radii(atom_types, radii)
    pdb2mrc.make_grids()
    t.append(time.time())
    fs.append("make_grids")

    pdb2mrc.calculate_global_B()
    t.append(time.time())
    fs.append("calculate_global_B")

    print("Optimal Side length >= %.2f" % pdb2mrc.optimal_side)
    print("Optimal N samples   >= %d" % pdb2mrc.optimal_nsamples)
    print("Optimal Voxel size  <= %.4f" % pdb2mrc.optimal_voxel)
    print("Actual  Side length  = %.2f" % pdb2mrc.side)
    print("Actual  N samples    = %d" % pdb2mrc.n)
    print("Actual  Voxel size   = %.4f" % pdb2mrc.dx)
    print("Global B-factor      = %.4f" % pdb2mrc.global_B)

    pdb2mrc.calculate_invacuo_density()
    t.append(time.time())
    fs.append("calculate_invacuo_density")

    pdb2mrc.calculate_excluded_volume()
    t.append(time.time())
    fs.append("calculate_excluded_volume")

    pdb2mrc.calculate_hydration_shell()
    t.append(time.time())
    fs.append("calculate_hydration_shell")

    pdb2mrc.calculate_structure_factors()
    t.append(time.time())
    fs.append("calculate_structure_factors")

    if args.data is not None:
        pdb2mrc.load_data()
        t.append(time.time())
        fs.append("load_data")
        pdb2mrc.initialize_penalties(penalty_weight=args.penalty_weight)
        pdb2mrc.minimize_parameters(fit_radii=args.fit_radii)
    t.append(time.time())
    fs.append("minimize_parameters")

    print()
    print("Final parameter values:")
    for i in range(len(pdb2mrc.params)):
        print("%s : %.5e" % (pdb2mrc.param_names[i],pdb2mrc.params[i]))

    pdb2mrc.calc_rho_with_modified_params(pdb2mrc.params)
    pdb2mrc.shell_mean_density = np.mean(pdb2mrc.rho_shell[pdb2mrc.water_shell_idx])
    t.append(time.time())
    fs.append("calc_rho_with_modified_params")

    pdb2mrc.calc_I_with_modified_params(pdb2mrc.params)
    t.append(time.time())
    fs.append("calc_I_with_modified_params")
    if args.data:
        pdb2mrc.calc_chi2()
        fit = pdb2mrc.fit
        optimized_chi2 = pdb2mrc.optimized_chi2
        exp_scale_factor = pdb2mrc.exp_scale_factor
        offset = pdb2mrc.offset
        print("Scale factor: %.5e " % pdb2mrc.exp_scale_factor)
        print("Offset: %.5e " % pdb2mrc.offset)
        print("chi2 of fit:  %.5e " % pdb2mrc.optimized_chi2)

    print("Calculated average radii:")
    pdb2mrc.calculate_average_radii()
    t.append(time.time())
    fs.append("calculate_average_radii")
    for i in range(len(pdb2mrc.modifiable_atom_types)):
        print("%s: %.3f"%(pdb2mrc.modifiable_atom_types[i],pdb2mrc.mean_radius[i]))

    pdb2mrc.calculate_excluded_volume_in_A3()
    print("Calculated excluded volume: %.2f"%(pdb2mrc.exvol_in_A3))

    end = time.time()
    print("Total calculation time: %.3f seconds" % (end-start))

    if args.qfile is not None:
        Iq = np.genfromtxt(args.qfile)
        qc = Iq[:,0]
    else:
        qc = None

    # always use sasrec interpolation for final output calculated profile, since its more accurate
    pdb2mrc.save_Iq_calc(prefix=output, qmax=args.qmax, nq=args.nq, qc=qc, use_sasrec=True)

    if args.write_shannon:
        qcalc_max = pdb2mrc.Iq_calc[:,0].max()
        wshannon = np.pi/pdb2mrc.D
        nshannon = int(qcalc_max/wshannon)
        qshannon = np.linspace(0,nshannon*wshannon,nshannon)
        pdb2mrc.save_Iq_calc(prefix=output+'_Shannon', qc=qshannon)

    if args.data is not None:
        pdb2mrc.save_fit(prefix=output)

        if args.plot:
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)

            q = fit[:,0]
            Ie = fit[:,1]
            err = fit[:,2]
            Ic = fit[:,3]

            ax0.plot(q, Ie,'.',c='gray',label=pdb2mrc.data_filename)
            ax0.plot(q, Ic, '-',c='red',label=basename+'.fit \n' + r'$\chi^2 = $ %.2f'%optimized_chi2)
            resid = (Ie - Ic)/err
            ax1.plot(q, resid*0, 'k--')
            ax1.plot(q, resid, '.',c='red')

            ax0.semilogy()
            ax1.set_xlabel(r"q ($\AA^{-1}$)")
            ax0.set_ylabel("I(q)")
            ax1.set_ylabel(r"$\Delta I / \sigma$")
            fig.suptitle(output)
            #title is often long, so wrap it to multiple lines if needed
            title = "\n".join(wrap(command, 80))
            ax0.set_title(title)
            ax0.legend()
            plt.tight_layout()
            plt.savefig(output+'_fits.png',dpi=300)
            # plt.show()
            plt.close()
    t.append(time.time())
    fs.append("plotting")

    #after all the map calculations are done, apply any resolution-based blurring directly to the map
    #but this should not be included in the scattering profile calculation, so we do it here just
    #before writing the maps
    if args.resolution is not None:
        sigma = args.resolution / pdb2mrc.dx
        if sigma <= 0.5:
            print("Note: requested resolution is less than half a voxel, meaning nothing will be done for the Gaussian filter.")
        if args.write_mrc_file:
            ne = pdb2mrc.rho_insolvent.sum()
            pdb2mrc.rho_insolvent = ndimage.gaussian_filter(pdb2mrc.rho_insolvent, sigma, mode='wrap')
            pdb2mrc.rho_insolvent *= ne/pdb2mrc.rho_insolvent.sum()
        if args.write_extras:
            ne = pdb2mrc.rho_invacuo.sum()
            pdb2mrc.rho_invacuo = ndimage.gaussian_filter(pdb2mrc.rho_invacuo, sigma, mode='wrap')
            pdb2mrc.rho_invacuo *= ne/pdb2mrc.rho_invacuo.sum()

            ne = pdb2mrc.rho_exvol.sum()
            pdb2mrc.rho_exvol = ndimage.gaussian_filter(pdb2mrc.rho_exvol, sigma, mode='wrap')
            pdb2mrc.rho_exvol *= ne/pdb2mrc.rho_exvol.sum()

            ne = pdb2mrc.rho_shell.sum()
            pdb2mrc.rho_shell = ndimage.gaussian_filter(pdb2mrc.rho_shell, sigma, mode='wrap')
            pdb2mrc.rho_shell *= ne/pdb2mrc.rho_shell.sum()
    t.append(time.time())
    fs.append("gaussian_filter")

    if args.write_mrc_file:
        #write output
        print('Writing density map to %s_insolvent.mrc file.'%output)
        denss.write_mrc(pdb2mrc.rho_insolvent/pdb2mrc.dV,pdb2mrc.side,output+"_insolvent.mrc")
    if args.write_extras:
        denss.write_mrc(pdb2mrc.rho_invacuo/pdb2mrc.dV,pdb2mrc.side,output+"_invacuo.mrc")
        denss.write_mrc(pdb2mrc.rho_exvol/pdb2mrc.dV,pdb2mrc.side,output+"_exvol.mrc")
        denss.write_mrc(pdb2mrc.rho_shell/pdb2mrc.dV,pdb2mrc.side,output+"_shell.mrc")
    t.append(time.time())
    fs.append("write_mrc")

    if args.print_timings:
        print("%40s: %7s %7s"%("function","time","cumtime"))
        for i in range(1,len(t)):
            print("%40s: %7.3f %7.3f"%(fs[i],t[i]-t[i-1],t[i]-t[0]))


if __name__ == "__main__":
    main()


