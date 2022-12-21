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
from scipy import interpolate
import sys, argparse, os

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input.")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
parser.add_argument("-n", "--nsamples", default=64, type=int, help="Desired number of samples per axis (default=64)")
parser.add_argument("-m", "--mode", default="slow", type=str, help="Mode. Either fast (Simple Gaussian sphere), slow (accurate 5-term Gaussian using Cromer-Mann coefficients), or FFT (default=slow).")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement (slow mode) Gaussian sphere width sigma (fast mode) (default=3*voxel)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
# parser.add_argument("--solv", default=0.000, type=float, help="Desired Solvent Density (experimental, default=0.000 e-/A^3)")
parser.add_argument("--ignore_waters", dest="ignore_waters", action="store_true", help="Ignore waters.")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
parser.set_defaults(ignore_waters = False)
parser.set_defaults(center = True)
args = parser.parse_args()

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
        #for slow mode, set resolution to be zero as this will then just
        #be equivalent to no B-factor, using just the atomic form factor
        resolution = 3*dx
    else:
        resolution = args.resolution

    if args.mode == "fast":
        rho = saxs.pdb2map_fastgauss(pdb,x=x,y=y,z=z,
                                    sigma=resolution,
                                    r=resolution*2,
                                    ignore_waters=args.ignore_waters)
    elif args.mode == "slow":
        #this slow mode uses the 5-term Gaussian with Cromer-Mann coefficients
        rho, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,ignore_waters=args.ignore_waters)
        # saxs.write_mrc(rho, side, "6lyz_rho_s256n256r1.mrc")
        # saxs.write_mrc(support*1.0, side, "6lyz_rho_s256n256r1_support.mrc")
    elif args.mode == "read":
        rho, side = saxs.read_mrc("6lyz_rho_s256n256r1.mrc")
        support, side = saxs.read_mrc("6lyz_rho_s256n256r1_support.mrc")
        support = support.astype(bool)
    else:
        print("Note: Using FFT method results in severe truncation ripples in map.")
        print("This will also run a quick refinement of phases to attempt to clean this up.")
        rho, support = saxs.pdb2map_FFT(pdb,x=x,y=y,z=z,radii=None)
        rho = saxs.denss_3DFs(rho_start=rho,dmax=side,voxel=dx,oversampling=1.,shrinkwrap=False,support=support)
    print()

    #copy particle pdb
    import copy
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import ndimage

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

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # data = np.loadtxt('SASDCK8.dat',skiprows=12)
    # ax0.plot(data[:,0],data[:,1],'.',c='gray',label='data')
    # I0 = 6.38033e+02 #data[0,1]

    # foxs = np.loadtxt('6lyz.dat')
    # foxs = np.loadtxt('6lyz.pdb.dat',skiprows=2)
    # foxs[:,1] *= I0/foxs[0,1]
    # plt.plot(foxs[:,0],foxs[:,1],label='foxs (default)')

    #6lyz10.abs has default excluded volume, but no contrast for the shell
    # crysol = np.loadtxt('6lyz10.abs',skiprows=1)
    # crysol[:,1] *= I0 / crysol[0,1]
    # plt.plot(crysol[:,0],crysol[:,1],label='crysol (no shell)')

    #6lyz11.abs has been fit to the SASDCK8.dat data
    # crysol2 = np.loadtxt('6lyz11.abs',skiprows=1)
    # crysol2[:,1] *= I0 / crysol2[0,1]
    # ax0.plot(crysol2[:,0],crysol2[:,1],'b-',label='crysol (optimized)')
    # ax1.plot(data[:,0], data[:,0]*0, 'k--')
    # resid = (data[:,1] - np.interp(data[:,0],crysol2[:,0],crysol2[:,1]))/data[:,2]
    # ax1.plot(data[:,0], resid, 'b-')

    #6lyz14.abs has 0.0 solvent density (so no ex vol) and no shell.
    # crysol3 = np.loadtxt('6lyz14.abs',skiprows=1)
    # crysol3[:,1] *= I0 / crysol3[0,1]
    # ax0.plot(crysol3[:,0],crysol3[:,1],'g-',label='crysol (no exvol or shell)')
    # ax1.plot(data[:,0], data[:,0]*0, 'k--')
    # resid = (data[:,1] - np.interp(data[:,0],crysol3[:,0],crysol3[:,1]))/data[:,2]
    # ax1.plot(data[:,0], resid, 'g-')

    #6lyz15.abs has 0.334 solvent density and no shell.
    # crysol4 = np.loadtxt('6lyz15.abs',skiprows=1)
    # crysol4[:,1] *= I0 / crysol4[0,1]
    # ax0.plot(crysol4[:,0],crysol4[:,1],'m-',label='crysol (no shell)')
    # ax1.plot(data[:,0], data[:,0]*0, 'k--')
    # resid = (data[:,1] - np.interp(data[:,0],crysol4[:,0],crysol4[:,1]))/data[:,2]
    # ax1.plot(data[:,0], resid, 'm-')

    # debye = np.loadtxt('6lyz_withH_noexvol.pdb2sas.dat')
    # debye[:,1] *= I0 / debye[0,1]
    # ax0.plot(debye[:,0],debye[:,1],label='debye (in vacuo)')

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

    # rho_s = 0.334 * dV
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
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    I_calc_interpolator = interpolate.interp1d(qbinsc,I_calc,kind='cubic')
    I_calc_interp = I_calc_interpolator(q_exp)
    exp_scale_factor = saxs._fit_by_least_squares(I_exp,I_calc_interp)

    I_exp /= exp_scale_factor
    sigq_exp /= exp_scale_factor

    ax0.plot(q_exp,I_exp,'.',c='gray',label='data')
    ax0.plot(qbinsc, I_calc, '.-',c='red',label='denss')
    resid = (I_exp - I_calc_interp)/sigq_exp
    ax1.plot(q_exp, resid, '.-',c='red')

    # diff = solv
    # print(rho.sum(), solv.sum())
    # F = saxs.myfftn(diff)
    # F[F.real==0] = 1e-16
    # # I3D = saxs.abs2(F)
    # I3D = saxs.myabs(F)**2
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # # Imean *= I0 / Imean[0]
    # ax0.plot(qbinsc, Imean, 'r-',label='denss (solv)')
    # resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    # ax1.plot(data[:,0], resid, 'r-')
    # np.savetxt('6lyz_exvol.pdb2mrc2dat.dat',np.vstack((qbinsc,Imean,Imean*.01)).T)

    # sf = 1.0
    # diff = rho - solv*sf
    # F = saxs.myfftn(diff)
    # F[F.real==0] = 1e-16
    # # I3D = saxs.abs2(F)
    # I3D = saxs.myabs(F)**2
    # #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    # #according to Schmidt-Rohr, J Appl Cryst 2007
    # I3D *= (np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # Imean *= I0 / Imean[0]

    # ax0.plot(qbinsc, Imean, 'y-',label='denss (rho-solv*%s)'%sf)
    # resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    # ax1.plot(data[:,0], resid, 'y-')

    # sf = 1.02
    # diff = rho - solv*sf
    # F = saxs.myfftn(diff)
    # F[F.real==0] = 1e-16
    # # I3D = saxs.abs2(F)
    # I3D = saxs.myabs(F)**2
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # Imean *= I0 / Imean[0]
    # ax0.plot(qbinsc, Imean, 'g-',label='denss (rho-solv*%s)'%sf)
    # resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    # ax1.plot(data[:,0], resid, 'g-')

    # sf = 1.15
    # diff = rho - solv*sf
    # F = saxs.myfftn(diff)
    # F[F.real==0] = 1e-16
    # I3D = saxs.abs2(F)
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # Imean *= I0 / Imean[0]
    # ax0.plot(qbinsc, Imean, 'm-',label='denss (rho-solv*%s)'%sf)
    # resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    # ax1.plot(data[:,0], resid, 'm-')

    # sf = 1.18
    # diff = rho - solv*sf
    # F = saxs.myfftn(diff)
    # F[F.real==0] = 1e-16
    # I3D = saxs.abs2(F)
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # Imean *= I0 / Imean[0]
    # ax0.plot(qbinsc, Imean, 'c-',label='denss (rho-solv*%s)'%sf)
    # resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    # ax1.plot(data[:,0], resid, 'c-')

    # greens = plt.get_cmap('Greens')
    # drho = np.linspace(0.055,0.065,3) * dV
    # for i in range(len(drho)):
    #     rho_s = 0.334 * dV
    #     shell_thickness = 1.0
    #     iterations = int(shell_thickness/dx)+1
    #     particle_for_shell = np.zeros_like(support)
    #     particle_for_shell[rho>threshold_for_shell*rho.max()] = True
    #     shell_idx = ndimage.binary_dilation(particle_for_shell,iterations=iterations)
    #     shell_idx[particle_for_shell] = False
    #     sigma_shell = 1.0
    #     shell = ndimage.gaussian_filter(shell_idx*1.0,sigma=sigma_shell,mode='wrap')

    #     #higher shell contrast amount seems to improve high q fit, make low q fit worse
    #     # drho = 0.05 * dV #contrast of the hydration shell
    #     shell *= drho[i]
    #     particle_for_exvol = np.zeros_like(support)
    #     particle_for_exvol[rho>threshold_for_exvol*rho.max()] = True
    #     shell[particle_for_exvol] = 0
    #     #larger sigma makes low q worse, high q better
    #     sigma_exvol = 1.0
    #     exvol = ndimage.gaussian_filter(particle*1.0,sigma=sigma_exvol,mode='wrap')
    #     exvol *= rho_s
    #     #add hydration shell to protein
    #     rho_with_shell = rho + shell
    #     #subtract excluded solvent
    #     diff = rho_with_shell - exvol
    #     F = saxs.myfftn(diff)
    #     F[F.real==0] = 1e-16
    #     I3D = saxs.abs2(F)
    #     Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    #     Imean *= I0 / Imean[0]
    #     c = greens(drho[i]/drho.max())
    #     ax0.plot(qbinsc, Imean, '-',c=c,label='denss (rho_s=%.3f, drho=%.3f)'%(rho_s,drho[i]))
    #     resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    #     ax1.plot(data[:,0], resid, '-', c=c)

    # saxs.write_mrc(particle*1.0,side,'particle.mrc')
    # saxs.write_mrc(shell*1.0,side,'shell.mrc')

    ax0.semilogy()
    # ax0.set_xlim([.03,1.8])
    # ax1.set_xlim([.03,1.8])
    # ax0.set_ylim([0.3,I0*1.1])
    ax1.set_xlabel(r"q ($\AA^{-1}$)")
    ax0.set_ylabel("I(q)")
    ax1.set_ylabel(r"$\Delta I / \sigma$")
    ax0.legend()
    # fig.set_title('%s: side=%.2f, N=%d, dx=%.2f, res=%.1f'%(basename,side,n,dx,resolution))
    plt.savefig('fits.png',dpi=300)
    plt.show()


    # #subtract solvent density value
    # #first, identify which voxels have particle
    # support = np.zeros(rho.shape,dtype=bool)
    # #set a threshold for selecting which voxels have density
    # #say, some low percent of the maximum
    # #this becomes important for estimating solvent content
    # support[rho>=args.solv*dV] = True
    # rho[~support] = 0
    # #scale map to total number of electrons while still in vacuum
    # #to adjust for some small fraction of electrons just flattened
    # rho *= np.sum(pdb.nelectrons) / rho.sum()
    # #convert total electron count to density
    # rho /= dV
    # #now, subtract the solvent density from the particle voxels
    # rho[support] -= args.solv


    #use support, which is 0s and 1s, to simulate the effect of a 
    #constant bulk solvent, but after blurring at the resolution desired
    #need to convert resolution in angstroms used above into sigma for blurring
    #sigma is in pixels, but also some kind of quadratic relationship
    # from scipy import ndimage
    # import matplotlib.pyplot as plt
    # sigma_in_A = args.resolution**0.5
    # sigma_in_pix = sigma_in_A * dx
    # print(sigma_in_A, sigma_in_pix)
    # for sf in np.linspace(.1,1,10):
    #     solvent = ndimage.filters.gaussian_filter(support*1.0,sigma=0.5,mode='wrap')
    #     saxs.write_mrc(solvent,side,output+"_solvent_%s.mrc"%sf)
    #     diff = rho-solvent*sf
    #     saxs.write_mrc(diff,side,output+"_diff_%s.mrc"%sf)
    
    #     plt.hist(solvent.ravel(),bins=100)
    #     #plt.hist(diff.ravel(),bins=100)
    # plt.show()

    #write output
    saxs.write_mrc(rho,side,output+".mrc")
    saxs.write_mrc(diff,side,output+"_diff.mrc")
    #saxs.write_mrc(support*1.,side,output+"_support.mrc")






