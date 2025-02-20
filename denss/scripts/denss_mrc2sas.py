#!/usr/bin/env python
#
#    denss_mrc2sas.py
#    A tool for calculating a scattering profile from an electron density
#    map and fitting to experimental SWAXS data.
#
#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tdgrant@buffalo.edu>
#    Copyright 2023-Present The Research Foundation for SUNY
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
import os, argparse, sys
import numpy as np
from scipy import interpolate

import denss
from textwrap import wrap


def main():
    parser = argparse.ArgumentParser(description="A tool for calculating a scattering profile from an electron density map and fitting to experimental SWAXS data.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="Electron density filename (.mrc) (required)")
    parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional, has priority over -q options for qgrid interpolation).")
    parser.add_argument("-q", "--qfile", default=None, type=str, help="ASCII text filename to use for setting the calculated q values (like a SAXS .dat file, but just uses first column, optional, -data has priority for qgrid interpolation).")
    parser.add_argument("-qmax", "--qmax", default=None, type=float, help="Maximum q value for calculated intensities (optional)")
    parser.add_argument("-nq", "--nq", default=None, type=int, help="Number of data points in calculated intensity profile (optional)")
    parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use of experimental data")
    parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use of experimental data")
    parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
    parser.add_argument("-fit_scale_on", "--fit_scale_on", dest="fit_scale", action="store_true", help="Include scale factor in least squares fit to data (optional, default=True)")
    parser.add_argument("-fit_scale_off", "--fit_scale_off", dest="fit_scale", action="store_false", help="Do not include offset in least squares fit to data.")
    parser.add_argument("-fit_offset_on", "--fit_offset_on", dest="fit_offset", action="store_true", help="Include offset in least squares fit to data (optional, default=False)")
    parser.add_argument("-fit_offset_off", "--fit_offset_off", dest="fit_offset", action="store_false", help="Do not include offset in least squares fit to data.")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Plot the profile (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not plot the profile. (Default if Matplotlib does not exist)")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
    parser.set_defaults(plot=True)
    parser.set_defaults(fit_scale=True)
    parser.set_defaults(fit_offset=False)
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

    scriptname = os.path.basename(sys.argv[0])
    command = scriptname + ' ' + ' '.join(sys.argv[1:])

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    #read density map and calculate scattering profile
    rho, side = denss.read_mrc(args.file)
    halfside = side/2
    nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]
    n = nx
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qz_ = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0]) - 1e-8
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = denss.mybinmean(qr.ravel(), qblravel, xcount)

    #calculate scattering profile from density
    F = denss.myfftn(rho)
    I3D = denss.abs2(F)
    Imean = denss.mybinmean(I3D.ravel(), qblravel, xcount=xcount)
    Iq_calc = np.vstack((qbinsc, Imean, Imean*.01)).T

    if args.data is not None:
        #read experimental data
        q, I, sigq, Ifit, file_dmax, isfit = denss.loadProfile(args.data, units=args.units)
        Iq = np.vstack((q,I,sigq)).T
        Iq = Iq[~np.isnan(Iq).any(axis = 1)]
        #get rid of any data points equal to zero in the intensities or errors columns
        idx = np.where((Iq[:,1]!=0)&(Iq[:,2]!=0))
        Iq = Iq[idx]
        if args.units == "nm":
            Iq[:,0] *= 0.1
        Iq_exp_orig = np.copy(Iq)
        if args.n1 is None:
            args.n1 = 0
        if args.n2 is None:
            args.n2 = len(Iq[:,0])
        Iq = Iq[args.n1:args.n2]
    else:
        #if no experimental data given, create a new qgrid for interpolation
        if args.qfile is not None:
            #if qfile is given, this takes priority over qmax/nq options
            q = np.loadtxt(args.qfile,usecols=(0,))
        else:
            #let a user set a desired set of q values to be calculated
            #based on a given qmax and nq
            if args.qmax is not None:
                qmax = args.qmax
            else:
                qmax = np.max(qbinsc)
            if args.nq is not None:
                nq = args.nq
            else:
                nq = 501
            q = np.linspace(0,qmax,nq)
        #interpolate Iq_calc to desired qgrid
        I_interpolator = interpolate.interp1d(Iq_calc[:,0],Iq_calc[:,1],kind='cubic',fill_value='extrapolate')
        I = I_interpolator(q)
        err_interpolator = interpolate.interp1d(Iq_calc[:,0],Iq_calc[:,2],kind='cubic',fill_value='extrapolate')
        err = err_interpolator(q)
        Iq = np.vstack((q,I,err)).T

    qmax = np.min([Iq[:,0].max(),Iq_calc[:,0].max()])
    Iq = Iq[Iq[:,0]<=qmax]
    Iq_calc = Iq_calc[Iq_calc[:,0]<=qmax]

    #calculate fit with interpolation
    final_chi2, exp_scale_factor, offset, fit = denss.calc_chi2(Iq, Iq_calc, scale=args.fit_scale, offset=args.fit_offset, interpolation=True,return_sf=True,return_fit=True)
    np.savetxt(output+'.mrc2sas.dat', fit[:,[0,3,2]], delimiter=' ', fmt='%.5e',
        header='q(data),I(density),error(data)')
    np.savetxt(output+'.mrc2sas.fit', fit, delimiter=' ', fmt='%.5e',
        header='q(data),I(data),error(data),I(density); chi2=%.3f'%final_chi2)

    print("Chi2 = %.3f"%final_chi2)

    if args.plot:
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)

        q = fit[:,0]
        Ie = fit[:,1]
        err = fit[:,2]
        Ic = fit[:,3]

        ax0.plot(q, Ie,'.',c='gray',label=args.file)
        ax0.plot(q, Ic, '-',c='red',label=output+'.mrc2sas.fit \n' + r'$\chi^2 = $ %.2f'%final_chi2)
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
        plt.savefig(output+'_mrc2sas_fit.png',dpi=300)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()




