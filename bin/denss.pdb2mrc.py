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
from scipy import interpolate, ndimage, optimize
import sys, argparse, os
#copy particle pdb
import copy
import time

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input.")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
parser.add_argument("-n", "--nsamples", default=64, type=int, help="Desired number of samples per axis (default=64)")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement, default=3*voxel)")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-vdW", "--vdW", "-vdw", "--vdw", dest="vdW", default=None, nargs=4, type=float, help="van der Waals radii of H, C, N, O, respectively. (optional)")
parser.add_argument("-shell_density", "--shell_density", default=0.00, type=float, help="Contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shell_sigma", "--shell_sigma", default=0.0, type=float, help="Hydration shell thickness in A (default=3.0)")
parser.add_argument("-b", "--b", "--use_b", dest="use_b", action="store_true", help="Include B-factors in atomic model (optional, default=False)")
parser.add_argument("-fit_rho0", "--fit_rho0", dest="fit_rho0", action="store_true", help="Fit rho0, the bulk solvent density (optional, default=False)")
parser.add_argument("-fit_vdW", "--fit_vdW", "-fit_vdw", "--fit_vdw", dest="fit_vdW", action="store_true", help="Fit van der Waals radii (optional, default=False)")
parser.add_argument("-fit_shell", "--fit_shell", dest="fit_shell", action="store_true", help="Fit hydration shell parameters (optional, default=False)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("--ignore_waters", dest="ignore_waters", action="store_true", help="Ignore waters.")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use of experimental data")
parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use of experimental data")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
parser.set_defaults(ignore_waters = False)
parser.set_defaults(center = True)
parser.set_defaults(plot=True)
parser.set_defaults(use_b=False)
args = parser.parse_args()


np.set_printoptions(linewidth=150,precision=10)

if args.plot:
    #if plotting is enabled, try to import matplotlib
    #if import fails, set plotting to false
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError as e:
        print("matplotlib import failed.")
        args.plot = False

def calc_chi2(Iq_exp, Iq_calc, scale=True):
    q_exp = Iq_exp[:,0]
    I_exp = Iq_exp[:,1]
    sigq_exp = Iq_exp[:,2]
    q_calc = Iq_calc[:,0]
    I_calc = Iq_calc[:,1]
    I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic')
    I_calc_interp = I_calc_interpolator(q_exp)
    if scale:
        exp_scale_factor = saxs._fit_by_least_squares(I_calc_interp/sigq_exp,Iq_exp[:,1]/sigq_exp)
    else:
        exp_scale_factor = 1.0
    I_exp_tmp = np.copy(I_exp)
    sigq_exp_tmp = np.copy(sigq_exp)
    I_exp_tmp *= exp_scale_factor
    sigq_exp_tmp *= exp_scale_factor
    chi2 = 1/len(q_exp) * np.sum(((I_exp_tmp-I_calc_interp)/sigq_exp_tmp)**2)
    return chi2, exp_scale_factor

def calc_score(Iq_exp, Iq_calc):
    chi2, exp_scale_factor = calc_chi2(Iq_exp, Iq_calc)
    return chi2

def create_shell_idx(support,x,y,z,shell_thickness=3.0):
    """Given an initial support, generate a new binary support object containing
    the indices of a shell around the support."""
    dx = x[1,0,0]-x[0,0,0]
    iterations = np.ceil(shell_thickness/dx).astype(int)
    shell_idx = ndimage.binary_dilation(support,iterations=iterations)
    shell_idx[support] = False
    return shell_idx

def create_shell(support=None,x=None,y=None,z=None,shell_idx=None,shell_thickness=3.0,shell_density=0.364,shell_profile="gaussian",shell_sigma=3.0):
    """Given an initial support, generate a shell of density around the support.

    thickness - thickness of the support in angstroms
    shell_density - density of the support in e-/A^3 (default=0.03)
    shell_profile - either "gaussian" (default) or "uniform" (like crysol)
    shell_sigma - if shell_profile is gaussian, this is the width of the gaussian in A (default = 3.0).
            Note: this is different from thickness in that thickness dictates the initial width of the
            shell_idx that will be generated, whereas shell_sigma is specifically for the gaussian width.
    """
    if shell_idx is None:
        #allow user to give shell_idx if known
        if support is None:
            print("To create shell_idx, must give support,x,y,z parameters.")
            exit()
        else:
            shell_idx = create_shell_idx(support,x,y,z,shell_thickness)
    #first, create a uniform density shell equal to shell_density
    shell = shell_idx * shell_density
    shell_ne = shell.sum()
    #if the shell_profile is gaussian, convolve the uniform density shell with a gaussian.
    #then reset the total number of electrons to match what originally was there
    if shell_profile == "gaussian":
        shell = ndimage.gaussian_filter(shell,sigma=shell_sigma,mode='wrap')
        # shell *= shell_ne/shell.sum()
    return shell

def calc_exvol_with_modified_params(params,pdb,x,y,z,ignore_waters=True):
    rho0 = params[0]
    atom_types = ['H','C','N','O']
    vdWs = params[3:]
    for i in range(len(atom_types)):
        #set the vdW for each atom type in the temporary pdb
        pdb.radius[pdb.atomtype==atom_types[i]] = vdWs[i]
    exvol, supportexvol = saxs.pdb2map_simple_gauss_by_radius(pdb,x,y,z,rho0=rho0,ignore_waters=ignore_waters)
    return exvol

def calc_shell_with_modified_params(params,pdb,x,y,z):
    rho0 = params[0]
    shell_density = params[1]
    shell_sigma = params[2]
    shell_thickness = 3.0
    if (shell_density==0) or (shell_sigma==0):
        shell = np.zeros_like(x)
    else:
        support = saxs.pdb2support_fast(pdb,x,y,z,dr=1.7)
        shell_idx = saxs.pdb2support_fast(pdb,x,y,z,dr=1.7+shell_thickness)
        shell_idx[support] = False
        shell = create_shell(shell_idx=shell_idx,shell_density=shell_density,shell_sigma=shell_sigma)
    return shell

# def calc_shell_with_modified_params(params,pdb,x,y,z):
#     rho0 = params[0]
#     shell_density = params[1]
#     shell_sigma = params[2]
#     support = saxs.pdb2support_fast(pdb,x,y,z)
#     shell = create_shell(support,x,y,z,shell_density=shell_density,shell_sigma=shell_sigma)
#     return shell

def calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo):
    exvol = calc_exvol_with_modified_params(params,pdb,x,y,z)
    #subtract excluded volume density from rho_invacuo
    rho_sum = rho_invacuo - exvol
    #add hydration shell to density
    shell_density = params[1]
    shell_sigma = params[2]
    if (shell_density!=0) and (shell_sigma!=0):
        shell = calc_shell_with_modified_params(params,pdb,x,y,z)
        rho_sum += shell
    return rho_sum

def calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,qbinsc,qblravel,xcount):
    rho_sum = calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo)
    Iq = mrc2sas(rho_sum,qbinsc,qblravel,xcount)
    #perform B-factor sharpening in rec space to correct for sampling issues
    u = saxs.B2u(pdb.b.mean())
    u += pdb.resolution
    B = saxs.u2B(u)
    B *= -1
    Iq[:,1] *= np.exp(-2*B* (Iq[:,0] / (4*np.pi))**2)
    return Iq

def mrc2sas(rho,qbinsc,qblravel,xcount):
    F = saxs.myfftn(rho)
    I3D = saxs.abs2(F)
    I3D *= latt_correction
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, xcount=xcount)
    Iq = np.vstack((qbinsc, I_calc, I_calc*.01 + I_calc[0]*0.002)).T
    return Iq

def calc_score_with_modified_params(params,pdb,x,y,z,rho_invacuo,qbinsc,qblravel,xcount,Iq_exp):
    Iq_calc = calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,qbinsc,qblravel,xcount)
    chi2, exp_scale_factor = calc_chi2(Iq_exp, Iq_calc)
    print(exp_scale_factor, params, chi2)
    return chi2


if __name__ == "__main__":
    start = time.time()

    command = ' '.join(sys.argv)

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

    if args.vdW is not None:
        try:
            pdb.radius[pdb.atomtype=='H'] = args.vdW[0]
            pdb.radius[pdb.atomtype=='C'] = args.vdW[1]
            pdb.radius[pdb.atomtype=='N'] = args.vdW[2]
            pdb.radius[pdb.atomtype=='O'] = args.vdW[3]
        except Error as e:
            print("Error assigning van der Waals radii")
            print(e)
            exit()

    if not args.use_b:
        pdb.b *= 0

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

    if args.resolution is None and not args.use_b:
        resolution = 0.3
    elif args.resolution is not None:
        resolution = args.resolution
    else:
        resolution = 0.0

    #for now, add in resolution to pdb object to enable passing between functions easily.
    pdb.resolution = resolution

    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
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
    # qbinsc = np.copy(qbins)
    # qbinsc[1:] += qstep/2.
    qbinsc = saxs.mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    q_calc = np.copy(qbinsc)

    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    latt_correction = 1. #(np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    rho_s = args.rho0 * dV

    rho, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,use_b=args.use_b,ignore_waters=args.ignore_waters)
    solv, supportsolv = saxs.pdb2map_simple_gauss_by_radius(pdb,x,y,z,rho0=args.rho0,ignore_waters=True)

    rho_invacuo = rho

    diff = rho - solv
    # print(diff.sum())

    F = saxs.myfftn(diff)
    F[F.real==0] = 1e-16
    I3D = saxs.abs2(F)
    I3D *= latt_correction
    q_calc = qbinsc
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)

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
        Iq_exp_orig = np.copy(Iq_exp)

        if args.n1 is None:
            n1 = 0
        else:
            n1 = args.n1
        if args.n2 is None:
            n2 = len(Iq_exp[:,0])
        else:
            n2 = args.n2

        Iq_exp = Iq_exp[n1:n2]

        print(Iq_exp[:,0])

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

        #interpolate the calculated scattering profile which is usually pretty poorly sampled to
        #the experimental q values for residual calculations and chi2 calculations. Use cubic spline
        #interpolation for better approximation.
        I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')
        I_calc_interp = I_calc_interpolator(q_exp_to_q0)
        exp_scale_factor = saxs._fit_by_least_squares(I_exp_to_q0[idx_overlap],I_calc_interp[idx_overlap])
        I_exp_to_q0 /= exp_scale_factor
        sigq_exp_to_q0 /= exp_scale_factor


    rho0 = args.rho0

    #generate a set of bounds
    bounds = np.zeros((7,2))

    #don't bother fitting if none is requested (default)
    fit_params = False
    if args.fit_rho0:
        rho0_guess = args.rho0
        bounds[0,0] = 0
        bounds[0,1] = np.inf
        fit_params = True
    else:
        rho0_guess = args.rho0
        bounds[0,0] = args.rho0
        bounds[0,1] = args.rho0
    if args.fit_shell:
        shell_density_guess = args.shell_density
        shell_sigma_guess = args.shell_sigma
        bounds[1,0] = -np.inf #could be negative
        bounds[1,1] = np.inf
        bounds[2,0] = 0
        bounds[2,1] = np.inf
        fit_params = True
    else:
        shell_density_guess = args.shell_density
        shell_sigma_guess = args.shell_sigma
        bounds[1,0] = shell_density_guess
        bounds[1,1] = shell_density_guess
        bounds[2,0] = shell_sigma_guess
        bounds[2,1] = shell_sigma_guess
    if args.fit_vdW:
        # vdWs_guess = [1.07, 1.58, 0.84, 1.30] #from crysol paper
        # vdWs_guess = [1.20, 1.775, 1.50, 1.45] #online dictionary
        # vdWs_guess = [0.885, 1.704, 0.937, 1.357] #6lyz denss optimization
        vdWs_guess = [saxs.radius.get(key) for key in ['H','C','N','O']]
        bounds[3:,0] = 0.0 #minimum vdW of 0
        bounds[3:,1] = 30.0 #maximum vdW of 3.0
        fit_params = True
    else:
        # vdWs_guess = [1.07, 1.58, 0.84, 1.30] #from crysol paper
        # vdWs_guess = [1.20, 1.775, 1.50, 1.45] #online dictionary
        # vdWs_guess = [0.885, 1.704, 0.937, 1.357] #6lyz denss optimization
        vdWs_guess = [saxs.radius.get(key) for key in ['H','C','N','O']]
        bounds[3:,0] = vdWs_guess
        bounds[3:,1] = vdWs_guess


    params_guess = np.zeros(7)
    params_guess[0] = rho0_guess
    params_guess[1] = shell_density_guess
    params_guess[2] = shell_sigma_guess
    params_guess[3:] = vdWs_guess
    if fit_params:
        results = optimize.minimize(calc_score_with_modified_params, params_guess,
            args = (pdb,x,y,z,rho_invacuo,qbinsc,qblravel,xcount,Iq_exp),
            bounds = bounds,
            )
        optimized_params = results.x
        optimized_chi2 = results.fun
    else:
        optimized_params = [rho0_guess] + [shell_density_guess, shell_sigma_guess] + vdWs_guess
        optimized_chi2 = "None"

    if fit_params:
        params = optimized_params
    else:
        params = params_guess

    shell = calc_shell_with_modified_params(params,pdb,x,y,z)
    rho_insolvent = calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo)
    Iq_calc = calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,qbinsc,qblravel,xcount)

    #attempt to correct for b-factor and resolution by performing negative B factor sharpening in reciprocal space
    #use average B-factor
    # u = saxs.B2u(pdb.b.mean())
    # u += resolution
    # B = saxs.u2B(u)
    # B *= -1
    # #multiply B-factor by 2 since now in intensity space
    # Iq_calc[:,1] *= np.exp(-2*B* (Iq_calc[:,0] / (4*np.pi))**2)

    if args.data:
        #interpolate the calculated scattering profile which is usually pretty poorly sampled to
        #the experimental q values for residual calculations and chi2 calculations. Use cubic spline
        #interpolation for better approximation.
        q_calc = Iq_calc[:,0]
        I_calc = Iq_calc[:,1]
        I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')
        I_calc_interp = I_calc_interpolator(q_exp_to_q0)
        exp_scale_factor = saxs._fit_by_least_squares(I_exp_to_q0[idx_overlap],I_calc_interp[idx_overlap])
        I_exp_to_q0 /= exp_scale_factor
        sigq_exp_to_q0 /= exp_scale_factor

    end = time.time()
    print("Total calculation time: %s" % (end-start))

    np.savetxt(output+'.pdb2mrc2sas.dat',Iq_calc,delimiter=' ',fmt='%.8e',header=' '.join(str(x) for x in optimized_params))


    if args.data is not None:
        fit = np.vstack((q_exp_to_q0, I_exp_to_q0, sigq_exp_to_q0, I_calc_interp)).T
        header = '' #'q, I, error, fit ; chi2= %.3e'%optimized_chi2
        np.savetxt(output+'.pdb2mrc2sas.fit', fit, delimiter=' ',fmt='%.5e',header=header)

        if args.plot:
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)

            plotidx = np.where(q_exp_to_q0<=q_exp.max())
            # plotidx = np.where(q_exp_to_q0<=q_exp_to_q0.max())

            ax0.plot(q_exp_to_q0[plotidx],I_exp_to_q0[plotidx],'.',c='gray',label='data')
            ax0.plot(q_exp_to_q0[plotidx], I_calc_interp[plotidx], '.-',c='red',label='denss')
            resid = (I_exp_to_q0[plotidx] - I_calc_interp[plotidx])/sigq_exp_to_q0[plotidx]
            ax1.plot(q_exp_to_q0[plotidx], resid[plotidx]*0, 'k--')
            ax1.plot(q_exp_to_q0[plotidx], resid[plotidx], '.-',c='red')

            ax0.semilogy()
            ax1.set_xlabel(r"q ($\AA^{-1}$)")
            ax0.set_ylabel("I(q)")
            ax1.set_ylabel(r"$\Delta I / \sigma$")
            fig.suptitle(basename)
            ax0.set_title(command)
            ax0.legend()
            plt.savefig(basename+'_fits.png',dpi=300)
            plt.show()

    #write output
    saxs.write_mrc(rho/dV,side,output+"_invacuo.mrc")
    saxs.write_mrc(rho_insolvent/dV,side,output+"_insolvent.mrc")
    saxs.write_mrc(shell/dV,side,output+"_shellonly.mrc")
    print(rho.sum(), pdb.nelectrons.sum())






