#!/usr/bin/env python
#
#    denss.pdb2mrc.py
#    A tool for calculating simple electron density maps from pdb files.
#
#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2018 The Research Foundation for SUNY
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
from saxstats._version import __version__
import saxstats.saxstats as saxs
import numpy as np
from scipy import interpolate, ndimage
import sys, argparse, os
#copy particle pdb
import copy

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input.")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
parser.add_argument("-n", "--nsamples", default=64, type=int, help="Desired number of samples per axis (default=64)")
parser.add_argument("-m", "--mode", default="slow", type=str, help="Mode. Either fast (Simple Gaussian sphere), slow (4-term Gaussian using Cromer-Mann coefficients), or FFT (default=slow).")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement (slow mode) Gaussian sphere width sigma (fast mode) (default=3*voxel)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("--ignore_waters", dest="ignore_waters", action="store_true", help="Ignore waters.")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
parser.set_defaults(ignore_waters = False)
parser.set_defaults(center = True)
parser.set_defaults(plot=True)
args = parser.parse_args()

if args.plot:
    #if plotting is enabled, try to import matplotlib
    #if import fails, set plotting to false
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError as e:
        print("matplotlib import failed.")
        args.plot = False

if __name__ == "__main__":

    fname_nopath = os.path.basename(args.file)
    basename, ext = os.path.splitext(fname_nopath)

    if args.output is None:
        output = basename + "_pdb"
    else:
        output = args.output

    pdb = saxs.PDB(args.file)
    if args.center:
        pdboutput = basename+"_centered.pdb"
        pdb.coords -= pdb.coords.mean(axis=0)
        pdb.write(filename=pdboutput)

    if args.side is None:
        #roughly estimate maximum dimension
        #calculate max distance along x, y, z
        #take the maximum of the three
        #double that value to set the default side
        xmin = np.min(pdb.coords[:,0])
        xmax = np.max(pdb.coords[:,0])
        ymin = np.min(pdb.coords[:,1])
        ymax = np.max(pdb.coords[:,1])
        zmin = np.min(pdb.coords[:,2])
        zmax = np.max(pdb.coords[:,2])
        wx = xmax-xmin
        wy = ymax-ymin
        wz = zmax-zmin
        side = 2*np.max([wx,wy,wz])
    else:
        side = args.side

    if args.voxel is None:
        voxel = side / args.nsamples
    else:
        voxel = args.voxel

    halfside = side/2
    n = int(side/voxel)
    #want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n%2==1: n += 1
    dx = side/n
    dV = dx**3
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')

    xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))

    if args.resolution is None:
        resolution = 3*dx
    else:
        resolution = args.resolution

    if args.mode == "fast":
        rho = saxs.pdb2map_fastgauss(pdb,x=x,y=y,z=z,
                                    sigma=resolution,
                                    r=resolution*2,
                                    ignore_waters=args.ignore_waters)
    elif args.mode == "slow":
        #this slow mode uses the 4-term Gaussian with Cromer-Mann coefficients
        rho, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,ignore_waters=args.ignore_waters)
    elif args.mode == "read":
        #in case a user has already calculated the in vacuo density map and exvol density map
        #allow those to be read in
        rho, side = saxs.read_mrc("6lyz_rho_s256n256r1.mrc")
        support, side = saxs.read_mrc("6lyz_rho_s256n256r1_support.mrc")
        support = support.astype(bool)
    else:
        print("Note: Using FFT method results in severe truncation ripples in map.")
        print("This will also run a quick refinement of phases to attempt to clean this up.")
        rho, support = saxs.pdb2map_FFT(pdb,x=x,y=y,z=z,radii=None)
        rho = saxs.denss_3DFs(rho_start=rho,dmax=side,voxel=dx,oversampling=1.,shrinkwrap=False,support=support)
    print()

    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qz_ = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0]) - 1e-8
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep/2.
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()

    if args.data is not None:
        Iq_exp = np.genfromtxt(args.data, invalid_raise = False, usecols=(0,1,2))
        if len(Iq_exp.shape) < 2:
            print("Invalid data format. Data file must have 3 columns: q, I, errors.")
            exit()
        if Iq_exp.shape[1] < 3:
            print("Not enough columns (data must have 3 columns: q, I, errors).")
            exit()
        Iq_exp = Iq_exp[~np.isnan(Iq_exp).any(axis = 1)]
        #get rid of any data points equal to zero in the intensities or errors columns
        idx = np.where((Iq_exp[:,1]!=0)&(Iq_exp[:,2]!=0))
        Iq_exp = Iq_exp[idx]
        if args.units == "nm":
            Iq_exp[:,0] *= 0.1
        q_exp = Iq_exp[:,0]
        I_exp = Iq_exp[:,1]
        sigq_exp = Iq_exp[:,2]
        Iq_exp_orig = np.copy(Iq_exp)

    rho_s = args.rho0 * dV

    if args.mode == 'read':
        solv, _ = saxs.read_mrc('6lyz_withH_pdb_solv.mrc')
        rho, _ = saxs.read_mrc('6lyz_withH_pdb_rho.mrc')
    else:
        solv, supportsolv = saxs.pdb2map_simple_gauss_by_radius(pdb,x,y,z,cutoff=3.0,rho0=args.rho0,ignore_waters=True)
        # saxs.write_mrc(solv, side, '6lyz_withH_pdb_solv.mrc')
        # saxs.write_mrc(rho, side, '6lyz_withH_pdb_rho.mrc')

    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    latt_correction = (np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    diff = rho - solv
    F = saxs.myfftn(diff)
    F[F.real==0] = 1e-16
    I3D = saxs.myabs(F)**2
    I3D *= latt_correction
    q_calc = qbinsc
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')

    # np.savetxt(basename+'.pdb2mrc2sas.dat',Iq_calc,delimiter=' ',fmt='%.8e')

    if args.data is not None:
        #for saving the fit, use the full original experimental profile
        q_exp = Iq_exp_orig[:,0]
        I_exp = Iq_exp_orig[:,1]
        sigq_exp = Iq_exp_orig[:,2]
        #generate q values for data down to q=0 using q_calc values in case they don't exist
        #fill those unmeasured q values with zero intensities and errors
        q_exp_to_q0 = np.concatenate((q_calc[q_calc<q_exp.min()],q_exp,q_calc[q_calc>q_exp.max()]))
        idx_low = np.where(q_exp_to_q0<q_exp.min())
        idx_overlap = np.where((q_exp_to_q0>=q_exp.min())&(q_exp_to_q0<=q_exp.max()))
        idx_high = np.where(q_exp_to_q0>q_exp.max())
        I_exp_to_q0 = np.copy(q_exp_to_q0)
        I_exp_to_q0[idx_low] = 0.0
        I_exp_to_q0[idx_overlap] = I_exp
        I_exp_to_q0[idx_high] = 0.0
        sigq_exp_to_q0 = np.copy(q_exp_to_q0)
        sigq_exp_to_q0[idx_low] = 0.0
        sigq_exp_to_q0[idx_overlap] = sigq_exp
        sigq_exp_to_q0[idx_high] = 0.0

        I_calc_interp = I_calc_interpolator(q_exp_to_q0)
        exp_scale_factor = saxs._fit_by_least_squares(I_exp_to_q0[idx_overlap],I_calc_interp[idx_overlap])
        I_exp_to_q0 /= exp_scale_factor
        sigq_exp_to_q0 /= exp_scale_factor

        fit = np.vstack((q_exp_to_q0, I_exp_to_q0, sigq_exp_to_q0, I_calc_interp)).T
        header = '' #'q, I, error, fit ; chi2= %.3e'%optimized_chi2
        np.savetxt(basename+'.pdb2mrc2sas.fit', fit, delimiter=' ',fmt='%.5e',header=header)

        if args.plot:
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)

            ax0.plot(q_exp_to_q0,I_exp_to_q0,'.',c='gray',label='data')
            ax0.plot(q_exp_to_q0, I_calc_interp, '.-',c='red',label='denss')
            resid = (I_exp_to_q0 - I_calc_interp)/sigq_exp_to_q0
            ax1.plot(q_exp_to_q0, resid*0, 'k--')
            ax1.plot(q_exp_to_q0, resid, '.-',c='red')

            ax0.semilogy()
            ax1.set_xlabel(r"q ($\AA^{-1}$)")
            ax0.set_ylabel("I(q)")
            ax1.set_ylabel(r"$\Delta I / \sigma$")
            ax0.legend()
            plt.savefig(basename+'_fits.png',dpi=300)
            # plt.show()

    #write output
    saxs.write_mrc(rho,side,output+".mrc")
    saxs.write_mrc(diff,side,output+"_diff.mrc")
    #saxs.write_mrc(support*1.,side,output+"_support.mrc")






