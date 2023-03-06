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
#    Email:  <tdgrant@buffalo.edu>
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
from textwrap import wrap

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input.")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
parser.add_argument("-n", "--nsamples", default=None, type=int, help="Desired number of samples per axis (default=None)")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement.)")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-vdW", "--vdW", "-vdw", "--vdw", dest="vdW", default=None, nargs='+', type=float, help="van der Waals radii of atom_types (for H, C, N, O, by default). (optional)")
parser.add_argument("-atom_types", "--atom_types", default=['H', 'C', 'N', 'O'], nargs='+', type=str, help="Atom types to allow modification of van der waals radii (space separated list, default = H C N O). (optional)")
parser.add_argument("-exvol_type", "--exvol_type", default="gaussian", type=str, help="Type of excluded volume (gaussian (default) or flat)")
parser.add_argument("-shell_contrast", "--shell_contrast", default=0.042, type=float, help="Initial contrast of hydration shell in e-/A^3 (default=0.042)")
parser.add_argument("-shell_invacuo", "-shell_invacuo_density_scale_factor", "--shell_invacuo_density_scale_factor", default=1.00, type=float, help="Contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shell_exvol","-shell_exvol_density_scale_factor", "--shell_exvol_density_scale_factor", default=1.00, type=float, help="Contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shell_type", "--shell_type", default="gaussian", type=str, help="Type of hydration shell (gaussian (default) or uniform)")
parser.add_argument("-shell_mrcfile", "--shell_mrcfile", default=None, type=str, help="Filename of hydration shell mrc file (default=None)")
parser.add_argument("-b", "--b", "--use_b", dest="use_b", action="store_true", help="Include B-factors in atomic model (optional, default=False)")
parser.add_argument("-fit_rho0", "--fit_rho0", dest="fit_rho0", action="store_true", help="Fit rho0, the bulk solvent density (optional, default=False)")
parser.add_argument("-fit_vdW", "--fit_vdW", "-fit_vdw", "--fit_vdw", dest="fit_vdW", action="store_true", help="Fit van der Waals radii (optional, default=False)")
parser.add_argument("-fit_shell", "--fit_shell", dest="fit_shell", action="store_true", help="Fit hydration shell parameters (optional, default=False)")
parser.add_argument("-p", "-penalty_weight", "--penalty_weight", default=100, type=float, help="Penalty weight for fitting parameters (default=1000)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("-iw", "--iw", "-iw_on", "--iw_on", "--ignore_waters", "--ignore_waters_on", dest="ignore_waters", action="store_true", help="Ignore waters (default=False.")
parser.add_argument("-iw_off", "--iw_off", "--ignore_waters_off", dest="ignore_waters", action="store_false", help="Turn Ignore waters off (i.e., read the waters).")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use of experimental data")
parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use of experimental data")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
parser.set_defaults(ignore_waters = True)
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
    I_calc_interp /= exp_scale_factor
    chi2 = 1/len(q_exp) * np.sum(((I_exp-I_calc_interp)/sigq_exp)**2)
    return chi2, exp_scale_factor

def calc_score(Iq_exp, Iq_calc):
    chi2, exp_scale_factor = calc_chi2(Iq_exp, Iq_calc)
    return chi2

def calc_exvol_with_modified_params(params,pdb,x,y,z,exvol_type="gaussian",ignore_waters=True):
    rho0 = params[0]
    atom_types = pdb.modified_atom_types #['H','C','N','O']
    vdWs = params[3:]
    for i in range(len(atom_types)):
        #set the vdW for each atom type in the temporary pdb
        # pdb.radius[pdb.atomtype==atom_types[i]] = vdWs[i]
        pdb.radius[pdb.atomtype==atom_types[i]] = vdWs[i] * pdb.modified_radius[pdb.atomtype==atom_types[i]]
    if exvol_type == "gaussian":
        #generate excluded volume assuming gaussian dummy atoms
        exvol, supportexvol = saxs.pdb2map_simple_gauss_by_radius(pdb,x,y,z,rho0=rho0,ignore_waters=ignore_waters)
    elif exvol_type == "flat":
        #generate excluded volume assuming flat solvent
        supportexvol = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=saxs.B2u(pdb.b))
        exvol = supportexvol * rho0
    return exvol

def calc_uniform_shell(pdb,x,y,z,thickness):
    dx = x[1,0,0]-x[0,0,0]
    dV = dx**3
    #create a one angstrom thick uniform layer around the particle
    #centered one water molecule radius away from the particle surface,
    #which means add the radius of a water molecule (1.4 A) to the radius of
    #the pdb surface atom (say 1.7 A), for a total of 1.4+1.7 from the pdb coordinates
    #since that is the center of the shell, and we want 1 A thick shell before blurring,
    #subtract 0.5 A from the inner support, and add 0.5 A for the outer support,
    #then subtract the inner support from the outer support
    #then blur that with a gaussian
    r_water = 1.4
    inner_support = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=r_water-thickness/2)
    outer_support = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=r_water+thickness/2)
    shell_idx = outer_support
    shell_idx[inner_support] = False
    shell = shell_idx * 1.0
    return shell

def calc_gaussian_shell(sigma,uniform_shell=None,pdb=None,x=None,y=None,z=None,thickness=None):
    if uniform_shell is None:
        #either give the uniform shell and skip this step, or give everything else to make it
        uniform_shell = calc_uniform_shell(pdb,x,y,z,thickness)
    shell = ndimage.gaussian_filter(uniform_shell,sigma=sigma,mode='wrap')
    return shell

def calc_shell_with_modified_params(params,shell_invacuo,shell_exvol):
    #modify hydration shell based on params
    shell_invacuo_density_scale_factor = params[1]
    shell_exvol_density_scale_factor = params[2]
    modified_shell_invacuo = shell_invacuo_density_scale_factor * shell_invacuo
    modified_shell_exvol = shell_exvol_density_scale_factor * shell_exvol
    shell_sum = modified_shell_invacuo - modified_shell_exvol
    return shell_sum, modified_shell_invacuo, modified_shell_exvol

def calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol):
    exvol = calc_exvol_with_modified_params(params,pdb,x,y,z,exvol_type=pdb.exvol_type)
    #subtract excluded volume density from rho_invacuo
    rho_sum = rho_invacuo - exvol
    #add hydration shell to density
    shell, _, _ = calc_shell_with_modified_params(params,shell_invacuo,shell_exvol)
    rho_sum += shell
    return rho_sum

def calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount):
    rho_sum = calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol)
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
    # I3D *= latt_correction
    I_calc = saxs.mybinmean(I3D.ravel(), qblravel, xcount=xcount)
    Iq = np.vstack((qbinsc, I_calc, I_calc*.01 + I_calc[0]*0.002)).T
    return Iq

def calc_penalty(params, params_target):
    nparams = len(params)
    params_weights = np.ones(nparams)
    #set the individual parameter penalty weights
    #to be 1/params_target, so that each penalty 
    #is weighted as a fraction of the target rather than an
    #absolute number.
    for i in range(nparams):
        if params_target[i] != 0:
            params_weights[i] = 1/params_target[i]
    #for now, ignore weights on shell parameters since we don't know them well
    params_weights[1:3] = 0
    #use quadratic loss function
    penalty = 1/nparams * np.sum((params_weights * (params - params_target))**2)
    return penalty

def calc_score_with_modified_params(params,params_target,penalty_weight,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount,Iq_exp):
    Iq_calc = calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount)
    chi2, exp_scale_factor = calc_chi2(Iq_exp, Iq_calc)
    penalty = penalty_weight * calc_penalty(params, params_target)
    score = chi2 + penalty
    print(exp_scale_factor, params, penalty, chi2)
    return score

def estimate_side_from_pdb(pdb):
    #roughly estimate maximum dimension
    #calculate max distance along x, y, z
    #take the maximum of the three
    #triple that value to set the default side
    #i.e. set oversampling to 3, like in denss
    xmin = np.min(pdb.coords[:,0]) - 1.7
    xmax = np.max(pdb.coords[:,0]) + 1.7
    ymin = np.min(pdb.coords[:,1]) - 1.7
    ymax = np.max(pdb.coords[:,1]) + 1.7
    zmin = np.min(pdb.coords[:,2]) - 1.7
    zmax = np.max(pdb.coords[:,2]) + 1.7
    wx = xmax-xmin
    wy = ymax-ymin
    wz = zmax-zmin
    side = 3*np.max([wx,wy,wz])
    return side


if __name__ == "__main__":
    start = time.time()

    command = ' '.join(sys.argv)

    fname_nopath = os.path.basename(args.file)
    basename, ext = os.path.splitext(fname_nopath)

    if args.output is None:
        output = basename + "_pdb"
    else:
        output = args.output

    pdb = saxs.PDB(args.file, ignore_waters=args.ignore_waters)
    if args.center:
        pdboutput = basename+"_centered.pdb"
        pdb.coords -= pdb.coords.mean(axis=0)
        pdb.write(filename=pdboutput)

    #allow setting of specific atom type radius
    # atom_types = ['H', 'C', 'N', 'O']
    atom_types = args.atom_types
    pdb.modified_atom_types = atom_types
    if args.vdW is not None:
        try:
            for i in range(len(atom_types)):
                pdb.radius[pdb.atomtype==atom_types[i]] = args.vdW[i]
        except Error as e:
            print("Error assigning van der Waals radii")
            print(e)
            exit()
        vdWs_guess = args.vdW
    else:
        vdWs_guess = [saxs.radius.get(key) for key in atom_types]

    pdb.generate_modified_pdb_radii()
    for i in range(len(atom_types)):
        #try using a scale factor for radii instead
        vdWs_guess[i] = 1.0 
        # print(pdb.radius[pdb.atomtype==atom_types[i]].mean())

    print(vdWs_guess)
    # print(pdb.modified_radius)

    pdb.exvol_type = args.exvol_type

    if not args.use_b:
        pdb.b *= 0

    #several of the calculations (such as shell generation) work best when
    #the voxel size is small, around 1 A or less, so assume that is the most 
    #important feature by default meaning that the side should be adjusted 
    #when giving nsamples, rather than adjusting the voxel size.
    #by default, if no options are given, the most accurate thing to do would 
    #be to estimate the side length as determined as above, assume the voxel 
    #size is close to 1A, and then adjust nsamples accordingly.
    #however, for large particles that might lead to prohibitively large nsamples
    #so if nsamples would default to being larger than 256, max out nsamples 
    #at 256, and reset the side length accordingly.
    #beyond that, the user will simply have to set them manually.

    if args.voxel is not None and args.nsamples is not None and args.side is not None:
        #if v, n, s are all given, voxel and nsamples dominates
        voxel = args.voxel
        nsamples = args.nsamples
        side = voxel * nsamples
    elif args.voxel is not None and args.nsamples is not None and args.side is None:
        #if v and n given, voxel and nsamples dominates
        voxel = args.voxel
        nsamples = args.nsamples
        side = voxel * nsamples
    elif args.voxel is not None and args.nsamples is None and args.side is not None:
        #if v and s are given, adjust side to match nearest integer value of n
        voxel = args.voxel
        side = args.side
        nsamples = np.ceil(side/voxel).astype(int)
        side = voxel * nsamples
    elif args.voxel is not None and args.nsamples is None and args.side is None:
        #if v is given, estimate side, calculate nsamples.
        voxel = args.voxel
        optimal_side = estimate_side_from_pdb(pdb)
        nsamples = np.ceil(optimal_side/voxel).astype(int)
        side = voxel * nsamples
        #if n > 256, adjust side length
        if nsamples > 256:
            nsamples = 256
            side = voxel * nsamples
        if side < optimal_side:
            print("side length may be too small and may result in undersampling errors.")
    elif args.voxel is None and args.nsamples is not None and args.side is not None:
        #if n and s are given, set voxel size based on those
        nsamples = args.nsamples
        side = args.side
        voxel = side / nsamples
        if voxel > 1.0:
            print("Warning: voxel size is greater than 1 A. This may lead to less accurate I(q) estimates at high q.")
    elif args.voxel is None and args.nsamples is not None and args.side is None:
        #if n is given, set voxel to 1, adjust side.
        nsamples = args.nsamples
        voxel = 1.0
        side = voxel * nsamples
        optimal_side = estimate_side_from_pdb(pdb)
        if side < optimal_side:
            print("side length may be too small and may result in undersampling errors.")
    elif args.voxel is None and args.nsamples is None and args.side is not None:
        #if s is given, set voxel to 1, adjust nsamples
        side = args.side
        voxel = 1.0
        nsamples = np.ceil(side/voxel).astype(int)
        if nsamples > 256:
            nsamples = 256
        voxel = side / nsamples
        if voxel > 1.0:
            print("Warning: voxel size is greater than 1 A. This may lead to less accurate I(q) estimates at high q.")
    elif args.voxel is None and args.nsamples is None and args.side is None:
        #if none given, set voxel to 1, estimate side length, adjust nsamples
        voxel = 1.0
        optimal_side = estimate_side_from_pdb(pdb)
        optimal_nsamples = np.ceil(optimal_side/voxel).astype(int)
        if optimal_nsamples > 256:
            nsamples = 256
        else:
            nsamples = optimal_nsamples
        side = voxel * nsamples
        if side < optimal_side:
            print("This must be a large particle. To ensure the highest accuracy, manually set")
            print("the -v option to 1 and the -s option to %.2f , " % optimal_side)
            print("which will set -n option to %d and thus may take a long time to calculate." % (optimal_nsamples))
            print("To avoid long computation times, the side length has been set to %.2f for now," % side)
            print("which may be too small and may result in undersampling errors.")



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
        resolution = 0.3 * dx 
    elif args.resolution is not None:
        resolution = args.resolution
    else:
        resolution = 0.0

    print("Side length: %.2f" % side)
    print("N samples:   %d" % n)
    print("Voxel size:  %.4f" % dx)

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
    qbinsc = saxs.mybinmean(qr.ravel(), qblravel, xcount=xcount, DENSS_GPU=False)
    q_calc = np.copy(qbinsc)

    #this multiplies the intensity by the form factor of a cube to correct for the discrete lattice
    #according to Schmidt-Rohr, J Appl Cryst 2007
    latt_correction = 1. #(np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    rho_invacuo, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,use_b=args.use_b,ignore_waters=args.ignore_waters)

    rho0 = args.rho0

    #calculate the volume of a shell of water diameter
    #this assumes a single layer of hexagonally packed water molecules on the surface
    r_water = 1.4 
    water_shell_idx = calc_uniform_shell(pdb,x,y,z,thickness=r_water).astype(bool)
    V_shell = water_shell_idx.sum() * dV
    N_H2O_in_shell = 2/(3**0.5) * V_shell / (2*r_water)**3
    V_H2O = 4/3*np.pi*r_water**3
    V_H2O_in_shell = N_H2O_in_shell * V_H2O
    print("Estimated number of H2O molecules in hydration shell: %d " % N_H2O_in_shell)
    print()

    shell_contrast = args.shell_contrast
    protein_with_shell_support = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=2*r_water)

    if args.shell_mrcfile is not None:
        #allow user to provide mrc filename to read in a custom shell
        shell_invacuo, sidex = saxs.read_mrc(args.shell_mrcfile)
        shell_exvol = x*0.0
        print(sidex, side)
        if (sidex != side) or (shell_invacuo.shape[0] != x.shape[0]):
            print("Error: shell_mrcfile does not match grid.")
            print("Use denss.mrcops.py to resample onto the desired grid.")
            exit()
    elif args.shell_type == "gaussian":
        #the default is gaussian type shell
        #generate initial hydration shell
        thickness = 1.0 
        invacuo_shell_sigma = 0.6
        exvol_shell_sigma = 1.0
        uniform_shell = calc_uniform_shell(pdb,x,y,z,thickness=thickness)
        shell_invacuo = calc_gaussian_shell(sigma=invacuo_shell_sigma,uniform_shell=uniform_shell)
        shell_exvol = calc_gaussian_shell(sigma=exvol_shell_sigma,uniform_shell=uniform_shell)

        #remove shell density that overlaps with protein
        print("hello")
        protein = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.modified_radius,probe=0.0)
        shell_invacuo[protein] = 0.0
        shell_exvol[protein] = 0.0

        ne_shell_exvol = V_H2O_in_shell * rho0 #10 electrons per water molecule
        shell_exvol *= ne_shell_exvol / shell_exvol.sum()
        #put back in units of density rather than electron count
        shell_exvol /= dV

        #estimate initial shell_invacuo scale based on contrast with shell_exvol using mean densities
        shell_exvol_mean_density = np.mean(shell_exvol[water_shell_idx])
        shell_invacuo_mean_density = shell_exvol_mean_density + shell_contrast
        #scale the mean density of the invacuo shell to match the desired mean density
        shell_invacuo *= shell_invacuo_mean_density / np.mean(shell_invacuo[water_shell_idx])

    elif args.shell_type == "uniform":
        #flat excluded volume in shell
        shell_exvol = water_shell_idx * rho0
        # shell_exvol *= 0
        shell_invacuo = water_shell_idx * (rho0 + shell_contrast)
    else:
        print("Error: no valid shell_type given. Disabling hydration shell.")
        shell_invacuo = x*0.0
        shell_exvol = x*0.0

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

        #for saving the fit, use the full original experimental profile
        q_exp = Iq_exp[:,0]
        I_exp = Iq_exp[:,1]
        sigq_exp = Iq_exp[:,2]
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

    param_names = ['rho0', 'shell_invacuo_scale_factor', 'shell_exvol_scale_factor'] + atom_types

    #generate a set of bounds
    bounds = np.zeros((len(param_names),2))

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
        shell_invacuo_density_scale_factor_guess = args.shell_invacuo_density_scale_factor
        shell_exvol_density_scale_factor_guess = args.shell_exvol_density_scale_factor
        bounds[1,0] = 0
        bounds[1,1] = np.inf
        bounds[2,0] = 0
        bounds[2,1] = np.inf
        fit_params = True
    else:
        shell_invacuo_density_scale_factor_guess = args.shell_invacuo_density_scale_factor
        shell_exvol_density_scale_factor_guess = args.shell_exvol_density_scale_factor
        bounds[1,0] = shell_invacuo_density_scale_factor_guess
        bounds[1,1] = shell_invacuo_density_scale_factor_guess
        bounds[2,0] = shell_exvol_density_scale_factor_guess
        bounds[2,1] = shell_exvol_density_scale_factor_guess
    if args.fit_vdW:
        # vdWs_guess = [1.07, 1.58, 0.84, 1.30] #from crysol paper
        # vdWs_guess = [1.20, 1.775, 1.50, 1.45] #online dictionary
        # vdWs_guess = [0.885, 1.704, 0.937, 1.357] #6lyz denss optimization
        # vdWs_guess = [saxs.radius.get(key) for key in ['H','C','N','O']]
        bounds[3:,0] = 0.0 #minimum vdW of 0
        bounds[3:,1] = 30.0 #maximum vdW of 3.0
        fit_params = True
    else:
        # vdWs_guess = [1.07, 1.58, 0.84, 1.30] #from crysol paper
        # vdWs_guess = [1.20, 1.775, 1.50, 1.45] #online dictionary
        # vdWs_guess = [0.885, 1.704, 0.937, 1.357] #6lyz denss optimization
        # vdWs_guess = [saxs.radius.get(key) for key in ['H','C','N','O']]
        bounds[3:,0] = vdWs_guess
        bounds[3:,1] = vdWs_guess

    params_guess = np.zeros(len(param_names))
    params_guess[0] = rho0_guess
    params_guess[1] = shell_invacuo_density_scale_factor_guess
    params_guess[2] = shell_exvol_density_scale_factor_guess
    params_guess[3:] = vdWs_guess
    penalty_weight = args.penalty_weight
    if fit_params:
        params_target = params_guess
        print(["scale_factor"], param_names, ["penalty"], ["chi2"])
        print("-"*100)
        results = optimize.minimize(calc_score_with_modified_params, params_guess,
            args = (params_target,penalty_weight,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount,Iq_exp),
            bounds = bounds,
            # method='Nelder-Mead',
            # method='L-BFGS-B', options={'eps':0.001},
            )
        optimized_params = results.x
        optimized_chi2 = results.fun
    else:
        optimized_params = [rho0_guess] + [shell_invacuo_density_scale_factor_guess,shell_exvol_density_scale_factor_guess] + vdWs_guess
        optimized_chi2 = "None"

    params = optimized_params

    print()
    print("Final parameter values:")
    for i in range(len(params)):
        print("%s : %.5e" % (param_names[i],params[i]))

    rho_insolvent = calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol)
    exvol = calc_exvol_with_modified_params(params,pdb,x,y,z)
    shell, modified_shell_invacuo, modified_shell_exvol = calc_shell_with_modified_params(params,shell_invacuo,shell_exvol)
    shell_invacuo_mean_density = np.mean(modified_shell_invacuo[water_shell_idx])
    shell_exvol_mean_density = np.mean(modified_shell_exvol[water_shell_idx])
    Iq_calc = calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount)

    print("Mean density of in vacuo shell: %.4f e-/A^3" % shell_invacuo_mean_density)
    print("Mean density of exvol shell:    %.4f e-/A^3" % shell_exvol_mean_density)
    if args.data:
        optimized_chi2, exp_scale_factor = calc_chi2(Iq_exp, Iq_calc)
        print("Scale factor: %.5e " % exp_scale_factor)
        print("Optimized chi2: %.5e " % optimized_chi2)

    for i in range(len(atom_types)):
        #try using a scale factor for radii instead
        vdWs_guess[i] = 1.0 
        # print(pdb.radius[pdb.atomtype==atom_types[i]].mean())

    pdb.V = 4/3*np.pi*pdb.modified_radius**3
    print(pdb.V.sum())
    print(pdb.V)
    print(4/3*np.pi*(1.7)**3)
    # protein = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=0.0)
    print(protein.sum()*dV)
    saxs.write_mrc((protein)*1.0,side,output+"_proteinsupport.mrc")

    if args.data:
        #interpolate the calculated scattering profile which is usually pretty poorly sampled to
        #the experimental q values for residual calculations and chi2 calculations. Use cubic spline
        #interpolation for better approximation.
        q_calc = Iq_calc[:,0]
        I_calc = Iq_calc[:,1]
        I_calc_interpolator = interpolate.interp1d(q_calc,I_calc,kind='cubic',fill_value='extrapolate')
        I_calc_interp = I_calc_interpolator(q_exp_to_q0)
        # exp_scale_factor = saxs._fit_by_least_squares(I_exp_to_q0[idx_overlap],I_calc_interp[idx_overlap])
        exp_scale_factor = saxs._fit_by_least_squares(I_calc_interp[idx_overlap]/sigq_exp_to_q0[idx_overlap],I_exp_to_q0[idx_overlap]/sigq_exp_to_q0[idx_overlap])
        I_exp_to_q0 *= exp_scale_factor
        sigq_exp_to_q0 *= exp_scale_factor

    end = time.time()
    print("Total calculation time: %.3f seconds" % (end-start))

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

            ax0.plot(q_exp_to_q0[plotidx],I_exp_to_q0[plotidx],'.',c='gray',label=args.data)
            ax0.plot(q_exp_to_q0[plotidx], I_calc_interp[plotidx], '.-',c='red',label=basename+'.pdb2mrc2sas.fit \n' + r'$\chi^2 = $ %.2f'%optimized_chi2)
            resid = (I_exp_to_q0[idx_overlap] - I_calc_interp[idx_overlap])/sigq_exp_to_q0[idx_overlap]
            ax1.plot(q_exp_to_q0[idx_overlap], resid*0, 'k--')
            ax1.plot(q_exp_to_q0[idx_overlap], resid, '.',c='red')

            ax0.semilogy()
            ax1.set_xlabel(r"q ($\AA^{-1}$)")
            ax0.set_ylabel("I(q)")
            ax1.set_ylabel(r"$\Delta I / \sigma$")
            fig.suptitle(basename)
            #title is often long, so wrap it to multiple lines if needed
            title = "\n".join(wrap(command, 80))
            ax0.set_title(title)
            ax0.legend()
            plt.tight_layout()
            plt.savefig(basename+'_fits.png',dpi=300)
            plt.show()

    #write output
    saxs.write_mrc(rho_invacuo,side,output+"_invacuo.mrc")
    saxs.write_mrc(exvol,side,output+"_exvol.mrc")
    saxs.write_mrc(shell,side,output+"_shell.mrc")
    saxs.write_mrc(rho_insolvent,side,output+"_insolvent.mrc")
    saxs.write_mrc((protein_with_shell_support)*1.0,side,output+"_supportwithshell.mrc")






