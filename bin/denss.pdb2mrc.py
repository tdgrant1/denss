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
import sys, argparse, os

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="PDB filename")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
parser.add_argument("-n", "--nsamples", default=64, type=int, help="Desired number of samples per axis (default=64)")
parser.add_argument("-m", "--mode", default="slow", type=str, help="Mode. Either fast (Simple Gaussian sphere), slow (accurate 5-term Gaussian using Cromer-Mann coefficients), or FFT (default=slow).")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement (slow mode) Gaussian sphere width sigma (fast mode) (default=3*voxel)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("--solv", default=0.000, type=float, help="Desired Solvent Density (experimental, default=0.000 e-/A^3)")
parser.add_argument("--ignore_waters", dest="ignore_waters", action="store_true", help="Ignore waters.")
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
    solvpdb = copy.deepcopy(pdb)
    #change all atom types O, should update to water form factor in future
    solvpdb.atomtype = np.array(['C' for i in solvpdb.atomtype],dtype=np.dtype((str,2)))
    # solv, supportsolv = saxs.pdb2map_multigauss(solvpdb,x=x,y=y,z=z,resolution=resolution,ignore_waters=args.ignore_waters)
    #now we need to fit some parameters
    #maybe a simple scale factor would get us close?
    # from scipy import ndimage
    # saxs.write_mrc((rho)/dV,side,output+"_0.0.mrc")
    # for i in np.linspace(0.1,1.0,10):
    #     solv_blur = ndimage.filters.gaussian_filter(solv,sigma=i,mode='wrap')
    #     # rho -= solv_blur*0.5
    #     # rho /= dV
    #     for j in np.linspace(0.1,1.0,10):
    #         saxs.write_mrc((rho-solv_blur*j)/dV,side,output+"_%.1f_%.1f.mrc"%(i,j))
    #     saxs.write_mrc(solv_blur/dV,side,output+"_solv_%.1f.mrc"%i)
    # c1 = 0.5
    # rho -= solv
    # rho /= dV
    #really need a B-factor modification to fit probably, which in this case is resolution
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

    data = np.loadtxt('SASDCK8.dat',skiprows=12)
    ax0.plot(data[:,0],data[:,1],'k.',label='data')
    I0 = 6.38033e+02 #data[0,1]

    # foxs = np.loadtxt('6lyz.dat')
    # foxs = np.loadtxt('6lyz.pdb.dat',skiprows=2)
    # foxs[:,1] *= I0/foxs[0,1]
    # plt.plot(foxs[:,0],foxs[:,1],label='foxs (default)')

    #6lyz10.abs has default excluded volume, but no contrast for the shell
    # crysol = np.loadtxt('6lyz10.abs',skiprows=1)
    # crysol[:,1] *= I0 / crysol[0,1]
    # plt.plot(crysol[:,0],crysol[:,1],label='crysol (no shell)')

    #6lyz11.abs has been fit to the SASDCK8.dat data
    crysol2 = np.loadtxt('6lyz11.abs',skiprows=1)
    crysol2[:,1] *= I0 / crysol2[0,1]
    ax0.plot(crysol2[:,0],crysol2[:,1],'b-',label='crysol (optimized)')
    ax1.plot(data[:,0], data[:,0]*0, 'k--')
    resid = (data[:,1] - np.interp(data[:,0],crysol2[:,0],crysol2[:,1]))/data[:,2]
    ax1.plot(data[:,0], resid, 'b-')

    #6lyz14.abs has 0.0 solvent density (so no ex vol) and no shell.
    # crysol3 = np.loadtxt('6lyz14.abs',skiprows=1)
    # crysol3[:,1] *= I0 / crysol3[0,1]
    # ax0.plot(crysol3[:,0],crysol3[:,1],'g-',label='crysol (no exvol or shell)')
    # ax1.plot(data[:,0], data[:,0]*0, 'k--')
    # resid = (data[:,1] - np.interp(data[:,0],crysol3[:,0],crysol3[:,1]))/data[:,2]
    # ax1.plot(data[:,0], resid, 'g-')

    #6lyz15.abs has 0.334 solvent density and no shell.
    crysol4 = np.loadtxt('6lyz15.abs',skiprows=1)
    crysol4[:,1] *= I0 / crysol4[0,1]
    ax0.plot(crysol4[:,0],crysol4[:,1],'m-',label='crysol (no shell)')
    ax1.plot(data[:,0], data[:,0]*0, 'k--')
    resid = (data[:,1] - np.interp(data[:,0],crysol4[:,0],crysol4[:,1]))/data[:,2]
    ax1.plot(data[:,0], resid, 'm-')

    debye = np.loadtxt('6lyz.pdb2sas.dat')
    debye[:,1] *= I0 / debye[0,1]
    # ax0.plot(debye[:,0],debye[:,1],label='debye (in vacuo)')

    # F = np.fft.fftn(rho)
    # I3D = saxs.abs2(F)
    # Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    # Imean *= foxs[0,1] / Imean[0]
    # plt.plot(qbinsc, Imean, '-',label='denss (in vacuo)')

    # res = 3.0
    # solv, supportsolv = saxs.pdb2map_fastgauss(solvpdb,x=x,y=y,z=z,resolution=res,ignore_waters=args.ignore_waters)
    # for i in np.linspace(1,5,5):
    #     print(1./i)
    #     # solv, supportsolv = saxs.pdb2map_multigauss(solvpdb,x=x,y=y,z=z,resolution=res,ignore_waters=args.ignore_waters)
    #     #calculate scattering profile from density
    #     diff = rho - solv * 1./i
    #     F = np.fft.fftn(diff)
    #     F[np.abs(F)==0] = 1e-16
    #     I3D = saxs.abs2(F)
    #     Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    #     Imean *= foxs[0,1] / Imean[0]
    #     # plt.plot(qbinsc, Imean, '-',label='res=%.1f'%res)
    #     plt.plot(qbinsc, Imean, '-',label='fraction=%.1f'%(1./i))

    #some stuff for calculating excluded volume form factor term
    pdb.radius = np.zeros(pdb.natoms)
    for i in range(pdb.natoms):
        if len(pdb.atomtype[i])==1:
            atomtype = pdb.atomtype[i][0].upper()
        else:
            atomtype = pdb.atomtype[i][0].upper() + pdb.atomtype[i][1].lower()
        try:
            dr = saxs.radius[atomtype]
        except:
            try:
                dr = saxs.radius[atomtype[0]]
            except:
                dr = saxs.radius['C']
        pdb.radius[i] = dr

    from scipy import spatial
    shift = np.ones(3)*dx/2.
    particle = np.zeros_like(support)
    for i in range(pdb.natoms):
        xa, ya, za = pdb.coords[i] # for convenience, store up x,y,z coordinates of atom
        xmin = int(np.floor((xa-dr)/dx)) + n//2
        xmax = int(np.ceil((xa+dr)/dx)) + n//2
        ymin = int(np.floor((ya-dr)/dx)) + n//2
        ymax = int(np.ceil((ya+dr)/dx)) + n//2
        zmin = int(np.floor((za-dr)/dx)) + n//2
        zmax = int(np.ceil((za+dr)/dx)) + n//2
        #handle edges
        xmin = max([xmin,0])
        xmax = min([xmax,n])
        ymin = max([ymin,0])
        ymax = min([ymax,n])
        zmin = max([zmin,0])
        zmax = min([zmax,n])
        #now lets create a slice object for convenience
        slc = np.s_[xmin:xmax,ymin:ymax,zmin:zmax]
        nx = xmax-xmin
        ny = ymax-ymin
        nz = zmax-zmin
        #now lets create a column stack of coordinates for the cropped grid
        xyz = np.column_stack((x[slc].ravel(),y[slc].ravel(),z[slc].ravel()))
        #now calculate all distances from the atom to the minigrid points
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)
        #now, add any grid points within dr of atom to the env grid
        #first, create a dummy array to hold booleans of size dist.size
        tmpenv = np.zeros(dist.shape,dtype=np.bool_)
        #now, any elements that have a dist less than the atomic radius make true
        tmpenv[dist<=pdb.radius[i]] = True
        #now reshape for inserting into env
        tmpenv = tmpenv.reshape(nx,ny,nz)
        particle[slc] += tmpenv

    threshold = 1e-4
    threshold_for_exvol = 1e-5
    threshold_for_shell = 1e-6

    # radii = np.zeros(pdb.natoms)
    # for i in range(pdb.natoms):
    #     if len(pdb.atomtype[i])==1:
    #         atomtype = pdb.atomtype[i][0].upper()
    #     else:
    #         atomtype = pdb.atomtype[i][0].upper() + pdb.atomtype[i][1].lower()
    #     try:
    #         dr = saxs.radius[atomtype]
    #     except:
    #         try:
    #             dr = saxs.radius[atomtype[0]]
    #         except:
    #             dr = saxs.radius['C']
    #     radii[i] = dr

    # radii += 0.0
    # solv, supportsolv = saxs.pdb2map_fastgauss(solvpdb,x=x,y=y,z=z,resolution=radii,ignore_waters=args.ignore_waters)
    # solv, supportsolv = saxs.pdb2map_fastgauss(solvpdb,x=x,y=y,z=z,resolution=radii*1,ignore_waters=args.ignore_waters)
    # sigma1 = 0.5
    # solv1 = solv #ndimage.gaussian_filter(solv,sigma=sigma1,mode='wrap')
    # particle = np.zeros_like(support)
    # particle[solv1>threshold*rho.max()] = True
    # particle[rho>threshold*rho.max()] = True

    # shell_thickness = 4.0
    # # shell_radii = radii * shell_thickness/2.
    # # solv2, supportsolv = saxs.pdb2map_fastgauss(solvpdb,x=x,y=y,z=z,resolution=shell_radii,ignore_waters=args.ignore_waters)
    # sigma2 = 1.0
    # solv2 = ndimage.gaussian_filter(solv,sigma=sigma2,mode='wrap')
    # shell = np.zeros_like(support)
    # shell[solv1>threshold*.0001*rho.max()] = True
    # shell[particle] = False

    rho_s = 0.334 * dV
    # sigma = 1.0
    # #scale solv term to be number of electrons seen by excluded volume from crysol
    # ne_invacuo_crysol = (0.582674E+08)**0.5 #sqrt of I(0)
    # ne_exvol_crysol = (0.358891E+08)**0.5
    # ratio = ne_invacuo_crysol/ne_exvol_crysol
    # solv *= 1/ratio * rho.sum()/solv.sum()

    solv = np.zeros_like(rho)
    solv[particle] = rho_s

    sf = 0.0
    diff = rho - solv*sf
    print(rho.sum(), solv.sum())
    F = saxs.myfftn(diff)
    F[F.real==0] = 1e-16
    I3D = saxs.abs2(F)
    Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    Imean *= I0 / Imean[0]
    ax0.plot(qbinsc, Imean, 'r-',label='denss (rho-solv*%s)'%sf)
    resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    ax1.plot(data[:,0], resid, 'r-')

    sf = 1.0
    diff = rho - solv*sf
    F = saxs.myfftn(diff)
    F[F.real==0] = 1e-16
    I3D = saxs.abs2(F)
    Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    Imean *= I0 / Imean[0]
    ax0.plot(qbinsc, Imean, 'y-',label='denss (rho-solv*%s)'%sf)
    resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    ax1.plot(data[:,0], resid, 'y-')

    sf = 2.0
    diff = rho - solv*sf
    F = saxs.myfftn(diff)
    F[F.real==0] = 1e-16
    I3D = saxs.abs2(F)
    Imean = saxs.mybinmean(I3D.ravel(), qblravel, DENSS_GPU=False)
    Imean *= I0 / Imean[0]
    ax0.plot(qbinsc, Imean, 'g-',label='denss (rho-solv*%s)'%sf)
    resid = (data[:,1] - np.interp(data[:,0],qbinsc,Imean))/data[:,2]
    ax1.plot(data[:,0], resid, 'g-')

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


    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    I3D_mod = I3D * (np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2
    Imean_mod = saxs.mybinmean(I3D_mod.ravel(), qblravel, DENSS_GPU=False)
    Imean_mod *= I0 / Imean_mod[0]
    # plt.plot(qbinsc, Imean_mod, '.-',label='modified by sinc')

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
    ax0.set_xlim([.03,1.8])
    ax1.set_xlim([.03,1.8])
    ax0.set_ylim([0.3,I0*1.1])
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






