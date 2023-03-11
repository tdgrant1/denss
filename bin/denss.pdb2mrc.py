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
#    Copyright 2017-Present The Research Foundation for SUNY
#
#    Additional Authors:
#    Sarah Chamberlain
#    Stephen Moore
#    Jitendra Singh
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
import sys, argparse, os, logging
import copy
import time
from textwrap import wrap

parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input (Hint: use *_OccasRadius.pdb to reuse unique radii calculation if run previously to save time).")
parser.add_argument("-d", "--data", type=str, help="Experimental SAXS data file for input (3-column ASCII text file (q, I, err), optional).")
parser.add_argument("-n1", "--n1", default=None, type=int, help="First data point to use of experimental data")
parser.add_argument("-n2", "--n2", default=None, type=int, help="Last data point to use of experimental data")
parser.add_argument("-u", "--units", default="a", type=str, help="Angular units of experimental data (\"a\" [1/angstrom] or \"nm\" [1/nanometer]; default=\"a\"). If nm, will convert output to angstroms.")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=3*Dmax).")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=1.0)")
parser.add_argument("-n", "--nsamples", default=None, type=int, help="Desired number of samples (i.e. voxels) per axis (default=variable)")
parser.add_argument("-b", "--b", "--use_b", dest="use_b", action="store_true", help="Include B-factors in atomic model (optional, default=False)")
parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (additional B-factor-like atomic displacement.)")
parser.add_argument("-exH","-explicitH","--explicitH", dest="explicitH", action="store_true", help="Use hydrogens in pdb file (optional, default=False)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("-iw", "--iw", "-iw_on", "--iw_on", "--ignore_waters", "--ignore_waters_on", dest="ignore_waters", action="store_true", help="Ignore waters (default=False.")
parser.add_argument("-iw_off", "--iw_off", "--ignore_waters_off", dest="ignore_waters", action="store_false", help="Turn Ignore waters off (i.e., read the waters).")
parser.add_argument("-fit", "--fit","-fit_on", "--fit_on", "-fit_all", "--fit_all", dest="fit_all", action="store_true", help="Fit everything (optional, default=True)")
parser.add_argument("-fit_off", "--fit_off", dest="fit_all", action="store_false", help="Do not fit anything (optional, default=True)")
parser.add_argument("-rho0", "--rho0", default=0.334, type=float, help="Density of bulk solvent in e-/A^3 (default=0.334)")
parser.add_argument("-exvol_type", "--exvol_type", default="gaussian", type=str, help="Type of excluded volume (gaussian (default) or flat)")
parser.add_argument("-fit_rho0", "--fit_rho0","-fit_rho0_on", "--fit_rho0_on", dest="fit_rho0", action="store_true", help="Fit rho0, the bulk solvent density (optional, default=True)")
parser.add_argument("-fit_rho0_off", "--fit_rho0_off", dest="fit_rho0", action="store_false", help="Do not fit rho0, the bulk solvent density (optional, default=True)")
parser.add_argument("-fit_radii", "--fit_radii","-fit_radii_on", "--fit_radii_on", dest="fit_radii", action="store_true", help="Fit radii (optional, default=True)")
parser.add_argument("-fit_radii_off", "--fit_radii_off", dest="fit_radii", action="store_false", help="Do not fit  radii (optional, default=True)")
parser.add_argument("-radii_sf", "--radii_sf", dest="radii_sf", default=None, nargs='+', type=float, help="Scale factor for fitting radii of atom_types (space separated list, default=1.0 for each --atom_type). (optional)")
parser.add_argument("-radii", "--radii", dest="radii", default=None, nargs='+', type=float, help="Radii of atom_types (space separated list, default=None as unique radius for each atom is calculated by default based on volume).")
parser.add_argument("-atom_types", "--atom_types", default=['H', 'C', 'N', 'O'], nargs='+', type=str, help="Atom types to allow modification of radii (space separated list, default = H C N O). (optional)")
parser.add_argument("-recalc","--recalc","--recalc_radii","--recalculate", default=False, dest="recalculate_unique_radii", action="store_true", help="Recalculate unique radii even if *_OccasRadius.pdb file given (default=False)")
parser.add_argument("-fit_shell", "--fit_shell","-fit_shell_on", "--fit_shell_on", dest="fit_shell", action="store_true", help="Fit hydration shell parameters (optional, default=True)")
parser.add_argument("-fit_shell_off", "--fit_shell_off", dest="fit_shell", action="store_false", help="Do not fit hydration shell parameters (optional, default=True)")
parser.add_argument("-shell_contrast", "--shell_contrast", default=0.042, type=float, help="Initial contrast of hydration shell in e-/A^3 (default=0.042)")
parser.add_argument("-shin","--shin","-shell_invacuo", "-shell_invacuo_density_scale_factor", "--shell_invacuo_density_scale_factor", dest="shell_invacuo_density_scale_factor", default=1.00, type=float, help="Contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shex","--shex","-shell_exvol","-shell_exvol_density_scale_factor", "--shell_exvol_density_scale_factor", dest="shell_exvol_density_scale_factor", default=1.00, type=float, help="Contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shell_type", "--shell_type", default="gaussian", type=str, help="Type of hydration shell (gaussian (default) or uniform)")
parser.add_argument("-shell_mrcfile", "--shell_mrcfile", default=None, type=str, help="Filename of hydration shell mrc file (default=None)")
parser.add_argument("-p", "-penalty_weight", "--penalty_weight", default=100, type=float, help="Overall penalty weight for fitting parameters (default=100)")
parser.add_argument("-ps", "-penalty_weights", "--penalty_weights", default=[3.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], type=float, nargs='+', help="Individual penalty weights for each of the seven (or more depending on atom_types) parameters (space separated listed of weight for [rho0,shin,shex,H,C,N,O], default=1.0 0.0 0.0 1.0 1.0 1.0 1.0)")
parser.add_argument("-min_method", "--min_method", "-minimization_method","--minimization_method", dest="method", default='Nelder-Mead', type=str, help="Minimization method (scipy.optimize method, default=Nelder-Mead).")
parser.add_argument("-write_extras", "--write_extras", action="store_true", default=False, help="Write out extra MRC files for invacuo, exvol, shell densities and supports (default=False).")
parser.add_argument("-interp_on", "--interp_on", dest="Icalc_interpolation", action="store_true", help="Interpolate I_calc to experimental q grid (default).")
parser.add_argument("-interp_off", "--interp_off", dest="Icalc_interpolation", action="store_false", help="Do not interpolate I_calc to experimental q grid .")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
parser.set_defaults(ignore_waters = True)
parser.set_defaults(center = True)
parser.set_defaults(plot=True)
parser.set_defaults(use_b=False)
parser.set_defaults(explicitH=False)
parser.set_defaults(fit_rho0=True)
parser.set_defaults(fit_radii=True)
parser.set_defaults(fit_shell=True)
parser.set_defaults(fit_all=True)
parser.set_defaults(Icalc_interpolation=True)
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

if __name__ == "__main__":
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
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Command: %s', ' '.join(sys.argv))
    logging.info('DENSS Version: %s', __version__)
    logging.info('PDB filename: %s', args.file)
    logging.info('Data filename: %s', args.data)
    logging.info('First data point: %s', args.n1)
    logging.info('Last data point: %s', args.n2)
    logging.info('Use atomic B-factors: %s', args.use_b)
    logging.info('Use explicit Hydrogens: %s', args.explicitH)
    logging.info('Center PDB: %s', args.center)
    logging.info('Ignore waters: %s', args.ignore_waters)
    logging.info('Excluded volume type: %s', args.exvol_type)

    pdb = saxs.PDB(args.file, ignore_waters=args.ignore_waters)

    if not args.explicitH:
        pdb.add_ImplicitH()
        print("Implicit hydrogens used")

    #add a line here that will delete alternate conformations if they exist
    if 'B' in pdb.atomalt:
        pdb.remove_atomalt()

    #allow setting of specific atom type radius
    atom_types = args.atom_types
    pdb.modified_atom_types = atom_types

    suffix = "_OccasRadius"
    occasradius = False
    if basename[-len(suffix):] == suffix:
        occasradius = True
        if args.recalculate_unique_radii:
            pdb.calculate_unique_volume()
        else:
            #if the file has unique radius in occupancy column, use it
            pdb.unique_radius = pdb.occupancy
            pdb.unique_volume = 4/3*np.pi*pdb.unique_radius**3
    elif args.radii is not None:
        try:
            for i in range(len(atom_types)):
                #Use pdb.exvolHradius for implicitH use
                if atom_types[i]=='H' and not args.explicitH:
                    pdb.exvolHradius = args.radii[i]
                else:    
                    pdb.radius[pdb.atomtype==atom_types[i]] = args.radii[i]
                    #if args.radii is given, make that the unique_radius
                    pdb.unique_radius[pdb.atomtype==atom_types[i]] = args.radii[i]
                    pdb.unique_volume = 4/3*np.pi*pdb.unique_radius**3
        except Error as e:
            print("Error assigning radii")
            print(e)
            exit()
    else:
        pdb.calculate_unique_volume()

    if args.radii_sf is not None:
        radii_sf = args.radii_sf
    else:
        radii_sf = np.ones(len(atom_types))

    #write the modified pdb file, and store the 
    #new unique volume/radius value in the occupancy
    #column, to prevent needing to recalculate if wanting
    #to run the fitting again.
    pdboutput = basename #overwrite if already has suffix (to prevent repeatedly adding suffixes)
    if not occasradius:
        #write new file if it does not have the suffix
        pdboutput += suffix + '.pdb'
    else:
        pdboutput += '.pdb'
    if args.center:
        pdb.coords -= pdb.coords.mean(axis=0)
    pdbout = copy.deepcopy(pdb)
    pdbout.occupancy = pdb.unique_radius
    pdbout.write(filename=pdboutput)

    print("Initial calculated average unique radii:")
    logging.info("Initial calculated average unique radii:")
    for i in range(len(atom_types)):
        #try using a scale factor for radii instead
        if atom_types[i]=='H' and not args.explicitH:
            mean_radius = pdb.exvolHradius
        else:
            mean_radius = pdb.unique_radius[pdb.atomtype==atom_types[i]].mean()
        print("%s: %.3f"%(atom_types[i],mean_radius))
        logging.info("%s: %.3f"%(atom_types[i],mean_radius))

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
    #however, large particles might lead to prohibitively large nsamples
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
        optimal_side = saxs.estimate_side_from_pdb(pdb)
        nsamples = np.ceil(optimal_side/voxel).astype(int)
        side = voxel * nsamples
        #if n > 256, adjust side length
        if nsamples > 256:
            nsamples = 256
            side = voxel * nsamples
        if side < optimal_side:
            print("side length may be too small and may result in undersampling errors.")
        if side < 2/3*optimal_side:
            print("Disabling interpolation of I_calc due to severe undersampling.")
            args.Icalc_interpolation = False
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
        optimal_side = saxs.estimate_side_from_pdb(pdb)
        if side < optimal_side:
            print("side length may be too small and may result in undersampling errors.")
        if side < 2/3*optimal_side:
            print("Disabling interpolation of I_calc due to severe undersampling.")
            args.Icalc_interpolation = False
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
        optimal_side = saxs.estimate_side_from_pdb(pdb)
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
        if side < 2/3*optimal_side:
            print("Disabling interpolation of I_calc due to severe undersampling.")
            args.Icalc_interpolation = False

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

    logging.info('Side length (angstroms): %.2f', side)
    logging.info('N samples: %d', n)
    logging.info('Voxel size: (angstroms): %.4f', dx)

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

    rho_invacuo, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,use_b=args.use_b,ignore_waters=args.ignore_waters)

    rho0 = args.rho0

    #calculate the volume of a shell of water diameter
    #this assumes a single layer of hexagonally packed water molecules on the surface
    r_water = 1.4 
    water_shell_idx = saxs.calc_uniform_shell(pdb,x,y,z,thickness=r_water).astype(bool)
    V_shell = water_shell_idx.sum() * dV
    N_H2O_in_shell = 2/(3**0.5) * V_shell / (2*r_water)**3
    V_H2O = 4/3*np.pi*r_water**3
    V_H2O_in_shell = N_H2O_in_shell * V_H2O
    print("Estimated number of H2O molecules in hydration shell: %d " % N_H2O_in_shell)
    print()

    shell_contrast = args.shell_contrast
    protein_with_shell_support = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.vdW,probe=2*r_water)

    if args.shell_mrcfile is not None:
        #allow user to provide mrc filename to read in a custom shell
        shell_invacuo, sidex = saxs.read_mrc(args.shell_mrcfile)
        shell_invacuo *= dV #assume mrc file is in units of density, convert to electron count
        shell_exvol = x*0.0
        print(sidex, side)
        if (sidex != side) or (shell_invacuo.shape[0] != x.shape[0]):
            print("Error: shell_mrcfile does not match grid.")
            print("Use denss.mrcops.py to resample onto the desired grid.")
            exit()
    elif args.shell_type == "gaussian":
        #the default is gaussian type shell
        #generate initial hydration shell
        thickness = 1.0 #in angstroms
        invacuo_shell_sigma = 0.6 / dx #convert to pixel units
        exvol_shell_sigma = 1.0 / dx #convert to pixel units
        uniform_shell = saxs.calc_uniform_shell(pdb,x,y,z,thickness=thickness)
        shell_invacuo = saxs.calc_gaussian_shell(sigma=invacuo_shell_sigma,uniform_shell=uniform_shell)
        shell_exvol = saxs.calc_gaussian_shell(sigma=exvol_shell_sigma,uniform_shell=uniform_shell)

        #remove shell density that overlaps with protein
        protein = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.unique_radius,probe=0.0)
        shell_invacuo[protein] = 0.0
        shell_exvol[protein] = 0.0

        ne_shell_exvol = V_H2O_in_shell * rho0 #10 electrons per water molecule
        shell_exvol *= ne_shell_exvol / shell_exvol.sum()
        #put back in units of density rather than electron count
        #actually, keep all density maps in electron count, so comment out next line
        # shell_exvol /= dV

        #estimate initial shell_invacuo scale based on contrast with shell_exvol using mean densities
        shell_exvol_mean_density = np.mean(shell_exvol[water_shell_idx]) / dV
        shell_invacuo_mean_density = shell_exvol_mean_density + shell_contrast
        #scale the mean density of the invacuo shell to match the desired mean density
        shell_invacuo *= shell_invacuo_mean_density / np.mean(shell_invacuo[water_shell_idx])
        #now convert shell_invacuo to electron count units
        shell_invacuo *= dV

    elif args.shell_type == "uniform":
        #flat excluded volume in shell
        shell_exvol = water_shell_idx * rho0
        shell_exvol *= dV #convert to electron units
        shell_invacuo = water_shell_idx * (rho0 + shell_contrast)
        shell_invacuo *= dV #convert to electron units
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
    else:
        Iq_exp = None
        fit_params = False

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
    if args.fit_radii:
        bounds[3:,0] = 0.0 
        bounds[3:,1] = np.inf 
        fit_params = True
    else:
        bounds[3:,0] = radii_sf
        bounds[3:,1] = radii_sf

    if args.data is None:
        #cannot fit parameters without data
        fit_params = False

    if not args.fit_all:
        #disable all fitting if requested
        fit_params = False

    params_guess = np.zeros(len(param_names))
    params_guess[0] = rho0_guess
    params_guess[1] = shell_invacuo_density_scale_factor_guess
    params_guess[2] = shell_exvol_density_scale_factor_guess
    params_guess[3:] = radii_sf
    penalty_weight = args.penalty_weight
    penalty_weights = args.penalty_weights
    if fit_params:
        params_target = params_guess
        print(["scale_factor"], param_names, ["penalty"], ["chi2"])
        print("-"*100)
        results = optimize.minimize(saxs.calc_score_with_modified_params, params_guess,
            args = (params_target,penalty_weight,penalty_weights,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount,Iq_exp,args.Icalc_interpolation),
            bounds = bounds,
            method=args.method, options={'adaptive': True}
            # method='L-BFGS-B', options={'eps':0.001},
            )
        optimized_params = results.x
        optimized_chi2 = results.fun
    else:
        optimized_params = params_guess
        optimized_chi2 = "None"

    params = optimized_params

    logging.info('Estimated number of waters in shell: %d', N_H2O_in_shell)
    logging.info('Final Parameter Values:')

    print()
    print("Final parameter values:")
    for i in range(len(params)):
        print("%s : %.5e" % (param_names[i],params[i]))
        logging.info("%s : %.5e" % (param_names[i],params[i]))

    rho_insolvent = saxs.calc_rho_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol)
    exvol = saxs.calc_exvol_with_modified_params(params,pdb,x,y,z,exvol_type=pdb.exvol_type)
    shell, modified_shell_invacuo, modified_shell_exvol = saxs.calc_shell_with_modified_params(params,shell_invacuo,shell_exvol)
    shell_invacuo_mean_density = np.mean(modified_shell_invacuo[water_shell_idx])
    shell_exvol_mean_density = np.mean(modified_shell_exvol[water_shell_idx])
    Iq_calc = saxs.calc_Iq_with_modified_params(params,pdb,x,y,z,rho_invacuo,shell_invacuo,shell_exvol,qbinsc,qblravel,xcount)

    logging.info("Mean density of in vacuo shell: %.4f e-/A^3" % shell_invacuo_mean_density)
    logging.info("Mean density of exvol shell:    %.4f e-/A^3" % shell_exvol_mean_density)

    print("Mean density of in vacuo shell: %.4f e-/A^3" % shell_invacuo_mean_density)
    print("Mean density of exvol shell:    %.4f e-/A^3" % shell_exvol_mean_density)
    if args.data:
        optimized_chi2, exp_scale_factor, fit = saxs.calc_chi2(Iq_exp, Iq_calc,interpolation=args.Icalc_interpolation,return_sf=True,return_fit=True)
        print("Scale factor: %.5e " % exp_scale_factor)
        print("chi2 of fit:  %.5e " % optimized_chi2)
        logging.info("Scale factor: %.5e " % exp_scale_factor)
        logging.info("chi2 of fit:  %.5e " % optimized_chi2)

    print("Final calculated average unique radii:")
    logging.info("Final calculated average unique radii:")
    for i in range(len(atom_types)):
        #try using a scale factor for radii instead
        if atom_types[i]=='H' and not args.explicitH:
            mean_radius = pdb.exvolHradius
        else:
            mean_radius = pdb.radius[pdb.atomtype==atom_types[i]].mean()
        print("%s: %.3f"%(atom_types[i],mean_radius))
        logging.info("%s: %.3f"%(atom_types[i],mean_radius))

    end = time.time()
    print("Total calculation time: %.3f seconds" % (end-start))
    logging.info("Total calculation time: %.3f seconds" % (end-start))

    # header_dat = "Scale factor: "
    header = ' '.join('%s: %.5e ; '%(param_names[i],params[i]) for i in range(len(params)))
    header_dat = header + "\n q_calc I_calc err_calc"

    np.savetxt(output+'.pdb2mrc2sas.dat',Iq_calc,delimiter=' ',fmt='%.8e',header=header_dat)

    if args.data is not None:
        # fit = np.vstack((q_exp_to_q0, I_exp_to_q0, sigq_exp_to_q0, I_calc)).T
        header_fit = header + '\n q, I, error, fit ; chi2= %.3e'%optimized_chi2
        np.savetxt(output+'.pdb2mrc2sas.fit', fit, delimiter=' ',fmt='%.5e',header=header_fit)

        if args.plot:
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)

            q = fit[:,0]
            Ie = fit[:,1]
            err = fit[:,2]
            Ic = fit[:,3]
            plotidx = np.where(q<=q_exp.max())

            ax0.plot(q_exp_to_q0[idx_overlap],I_exp_to_q0[idx_overlap],'.',c='gray',label=args.data)
            ax0.plot(q, Ie,'.',c='gray')
            ax0.plot(q, Ic, '.-',c='red',label=basename+'.pdb2mrc2sas.fit \n' + r'$\chi^2 = $ %.2f'%optimized_chi2)
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
            plt.show()

    #write output
    saxs.write_mrc(rho_insolvent/dV,side,output+"_insolvent.mrc")
    if args.write_extras:
        saxs.write_mrc(rho_invacuo/dV,side,output+"_invacuo.mrc")
        saxs.write_mrc(exvol/dV,side,output+"_exvol.mrc")
        saxs.write_mrc(shell/dV,side,output+"_shell.mrc")
        saxs.write_mrc((protein)*1.0,side,output+"_proteinsupport.mrc")
        saxs.write_mrc((protein_with_shell_support)*1.0,side,output+"_supportwithshell.mrc")






