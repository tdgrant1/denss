#!/usr/bin/env python
#
#    denss.rho2dat.py
#    A tool for calculating simple scattering profiles
#    from MRC formatted electron density maps
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
import os, argparse, sys
import logging
import numpy as np
from scipy import ndimage, optimize, interpolate
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for calculating simple scattering profiles from MRC formatted electron density maps", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Electron density filename (.mrc)")
parser.add_argument("-dq", "--dq", default=None, type=float, help="Desired q spacing (pads map with zeros)")
parser.add_argument("-n", "--n", default=None, type=int, help="Desired number of samples (overrides --dq option)")
parser.add_argument("--ns", default=1, type=int, help="Sampling fraction (i.e. every ns'th element, must be integer, allows for reduced sampling for speed up at cost of resolution (i.e. qmax will be smaller))")
parser.add_argument("-t","--threshold", default=None, type=float, help="Minimum density threshold (sets lesser values to zero).")
parser.add_argument("-exvol", "--exvol", default=None, type=str, help="Electron density filename (.mrc) of excluded volume (optional).")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("-fit_rho0", "--fit_rho0", dest="fit_rho0", action="store_true", help="Fit rho0 (optional, default=False)")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Plot the profile (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not plot the profile. (Default if Matplotlib does not exist)")
parser.add_argument("--save_mrc", action="store_true", help="Save the modified MRC file.")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
parser.set_defaults(plot=True)
parser.set_defaults(fit_rho0=False)
args = parser.parse_args()

if args.plot:
    #if plotting is enabled, try to import matplotlib
    #if import fails, set plotting to false
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        args.plot = False

def score_from_solvent_parameters(solvent_params, rho_invacuo, rho_exvol, qblravel, latt_correction, qexp, Iexp, sigq):
    rho0 = solvent_params[0]
    rho = rho_invacuo - rho0 * rho_exvol
    F = saxs.myfftn(rho)
    I3D = np.abs(F)**2
    I3D *= latt_correction
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel)
    I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')
    I_calc_interp = I_calc_interpolator(q_exp)
    exp_scale_factor = saxs._fit_by_least_squares(I_calc_interp/sigq,Iexp/sigq)
    Iexp_tmp = np.copy(Iexp)
    sigq_tmp = np.copy(sigq)
    Iexp_tmp *= exp_scale_factor
    sigq_tmp *= exp_scale_factor
    chi2 = np.sum(((Iexp_tmp - I_calc_interp)/sigq_tmp)**2) / Iexp.size
    print(rho0, exp_scale_factor, chi2)
    return chi2

if __name__ == "__main__":

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename + '_rho'
    else:
        output = args.output

    rho, side = saxs.read_mrc(args.file)
    if args.exvol is not None:
        exvol, exvol_side = saxs.read_mrc(args.exvol)
        if rho.shape[0] != exvol.shape[0]:
            print("rho and exvol shape mismatch")
            print("rho.shape: ", rho.shape )
            print("exvol.shape: ", exvol.shape )
        if side != exvol_side:
            print("rho and exvol side mismatch")
            print("rho side: ", side )
            print("exvol side: ", exvol_side )

    if rho.shape[0]%2==1:
        rho = rho[:-1,:-1,:-1]
        if args.exvol is not None:
            exvol = exvol[:-1,:-1,:-1]
 
    #if nstmp%2==1: args.ns+=1
    rho = np.copy(rho[::args.ns, ::args.ns, ::args.ns])
    if args.exvol is not None:
        exvol = np.copy(exvol[::args.ns, ::args.ns, ::args.ns])

    if args.threshold is not None:
        rho[np.abs(rho)<=args.threshold] = 0

    halfside = side/2
    nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]
    n = nx
    voxel = side/n
    #want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    #store n for later use if needed
    n_orig = n
    dx = side/n
    dV = dx**3
    V = side**3
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
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep/2.
    q_calc = np.copy(qbinsc)
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins,qr,"right")
    qbin_labels -= 1
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)

    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    latt_correction = (np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    #assume rho is given as electron density, not electron count
    #convert from density to electron count for FFT calculation
    rho *= dV
    if args.exvol is not None:
        exvol *= dV

    rho0 = args.rho0 * dV
    invacuo = np.copy(rho)
    # rho = invacuo - rho0 * exvol

    #calculate scattering profile from density
    F = saxs.myfftn(rho)
    F[F.real==0] = 1e-16
    # I3D = saxs.abs2(F)
    I3D = np.abs(F)**2
    I3D *= latt_correction
    Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)

    # if args.plot: 
    #     plt.plot(qbinsc, Imean, '.-', label='Default dq = %.4f' % (2*np.pi/side))
    print('Default dq = %.4f' % (2*np.pi/side))

    if args.dq is not None or args.n is not None:

        #padded to get desired dq value (or near it)
        if args.n is not None:
            n = args.n
        else:
            current_dq = 2*np.pi/side
            desired_dq = args.dq
            if desired_dq > current_dq:
                print("desired dq must be smaller than dq calculated from map (which is %f)" % current_dq)
                print("Resetting desired dq to current dq...")
                desired_dq = current_dq
            #what side would give us desired dq?
            desired_side = 2*np.pi/desired_dq
            #what n, given the existing voxel size, would give us closest to desired_side
            desired_n = desired_side/voxel
            n = int(desired_n)
            if n%2==1: n+=1
        side = voxel*n
        halfside = side/2
        dx = side/n
        dV = dx**3
        V = side**3
        x_ = np.linspace(-halfside,halfside,n)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        df = 1/side
        print(n, 2*np.pi*df)
        qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
        qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
        qr = np.sqrt(qx**2+qy**2+qz**2)
        qmax = np.max(qr)
        qstep = np.min(qr[qr>0]) - 1e-8
        nbins = int(qmax/qstep)
        qbins = np.linspace(0,nbins*qstep,nbins+1)
        #create modified qbins and put qbins in center of bin rather than at left edge of bin.
        qbinsc = np.copy(qbins)
        qbinsc[1:] += qstep/2.
        q_calc = np.copy(qbinsc)
        #create an array labeling each voxel according to which qbin it belongs
        qbin_labels = np.searchsorted(qbins,qr,"right")
        qbin_labels -= 1
        qblravel = qbin_labels.ravel()
        rho_pad = np.zeros((n,n,n),dtype=np.float32)
        a = n//2-n_orig//2
        b = n//2+n_orig//2
        rho_pad[a:b,a:b,a:b] = rho
        rho = np.copy(rho_pad)
        invacuo_pad = np.zeros((n,n,n),dtype=np.float32)
        invacuo_pad[a:b,a:b,a:b] = invacuo
        invacuo = np.copy(invacuo)
        exvol_pad = np.zeros((n,n,n),dtype=np.float32)
        exvol_pad[a:b,a:b,a:b] = exvol
        exvol = np.copy(exvol_pad)
        invacuo *= dV
        if args.exvol is not None:
            exvol *= dV
        rho0 = args.rho0 * dV

    print(invacuo.sum(), exvol.sum())

    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    latt_correction = (np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    F = saxs.myfftn(rho)
    # I3D = saxs.abs2(F)**2
    I3D = np.abs(F)**2
    I3D *= latt_correction
    q_calc = qbinsc
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')

    if args.data is not None:
        Iq_exp = np.genfromtxt(args.data, invalid_raise = False, usecols=(0,3,2))
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

        if args.fit_rho0:
            params = [rho0]
            results = optimize.minimize(score_from_solvent_parameters, params,
                args = (invacuo, exvol, qblravel, latt_correction, q_exp, I_exp, sigq_exp),
                # bounds = [(0.0,0.6)],
                )
            rho0 = results.x[0]

    #combine invacuo, exvol, and shell
    rho = invacuo - rho0 * exvol  #+ drho_shell * shell

    F = saxs.myfftn(rho)
    # I3D = saxs.abs2(F)**2
    I3D = np.abs(F)**2
    I3D *= latt_correction
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    Imean = np.copy(I_calc)
    I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')

    if args.data:
        I_calc_interp = I_calc_interpolator(q_exp_to_q0)
        exp_scale_factor = saxs._fit_by_least_squares(I_exp_to_q0[idx_overlap],I_calc_interp[idx_overlap])
        I_exp_to_q0 /= exp_scale_factor
        sigq_exp_to_q0 /= exp_scale_factor

    qmax_to_use = np.max(qx_)
    print("qmax to use: %f" % qmax_to_use)
    qbinsc_to_use = qbinsc[qbinsc<qmax_to_use]
    Imean_to_use = Imean[qbinsc<qmax_to_use]

    # qbinsc = np.copy(qbinsc_to_use)
    # Imean = np.copy(Imean_to_use)

    Iq = np.vstack((qbinsc, Imean, Imean*.03*Imean[0]*.001)).T

    np.savetxt(output+'.dat', Iq, delimiter=' ', fmt='% .16e')

    if args.save_mrc:
        saxs.write_mrc(rho, side, output+'_mod.mrc')

    if args.plot:
        print('Actual dq = %.4f' % (2*np.pi/side))
        plt.plot(q_exp_to_q0, I_exp_to_q0, 'k.', label=args.data)
        plt.plot(qbinsc_to_use, Imean_to_use,'.-', label='Actual dq = %.4f' % (2*np.pi/side))
        plt.xlabel('q (1/A)')
        plt.ylabel('I(q)')
        plt.semilogy()
        plt.legend()
        plt.show()







