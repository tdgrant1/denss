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
parser.add_argument("-exH","-explicitH","--explicitH", dest="explicitH", action="store_true", help="Use hydrogens in pdb file (optional, default=True if H exists)")
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
parser.add_argument("-fit_shell", "--fit_shell","-fit_shell_on", "--fit_shell_on", dest="fit_shell", action="store_true", help="Fit hydration shell parameters (optional, default=True)")
parser.add_argument("-fit_shell_off", "--fit_shell_off", dest="fit_shell", action="store_false", help="Do not fit hydration shell parameters (optional, default=True)")
parser.add_argument("-drho","--drho","-shell_contrast", "--shell_contrast", dest="shell_contrast", default=0.03, type=float, help="Initial mean contrast of hydration shell in e-/A^3 (default=0.03)")
parser.add_argument("-shell_type", "--shell_type", default="gaussian", type=str, help="Type of hydration shell (gaussian (default) or uniform)")
parser.add_argument("-shell_mrcfile", "--shell_mrcfile", default=None, type=str, help="Filename of hydration shell mrc file (default=None)")
parser.add_argument("-p", "-penalty_weight", "--penalty_weight", default=0., type=float, help="Overall penalty weight for fitting parameters (default=0)")
parser.add_argument("-ps", "-penalty_weights", "--penalty_weights", default=[1.0, 0.0], type=float, nargs='+', help="Individual penalty weights for each of the seven (or more depending on atom_types) parameters (space separated listed of weight for [rho0,shell], default=1.0 0.0)")
parser.add_argument("-min_method", "--min_method", "-minimization_method","--minimization_method", dest="method", default='Nelder-Mead', type=str, help="Minimization method (scipy.optimize method, default=Nelder-Mead).")
parser.add_argument("-min_options", "--min_options", "-minimization_options","--minimization_options", dest="minopts", default='{"adaptive": True}', type=str, help="Minimization options (scipy.optimize options formatted as python dictionary, default=\"{'adaptive': True}\").")
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
parser.set_defaults(explicitH=True)
parser.set_defaults(fit_rho0=True)
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

    pdb = saxs.PDB(args.file, ignore_waters=args.ignore_waters)

    if args.explicitH:
        #only use explicitH if H exists in the pdb file
        #for atoms that are not waters
        if 'H' not in pdb.atomtype[pdb.resname!="HOH"]:
            args.explicitH = False

    logging.info('Use atomic B-factors: %s', args.use_b)
    logging.info('Use explicit Hydrogens: %s', args.explicitH)
    logging.info('Center PDB: %s', args.center)
    logging.info('Ignore waters: %s', args.ignore_waters)
    logging.info('Excluded volume type: %s', args.exvol_type)

    if not args.explicitH:
        pdb.add_ImplicitH()
        print("Implicit hydrogens used")

    #add a line here that will delete alternate conformations if they exist
    if 'B' in pdb.atomalt:
        pdb.remove_atomalt()

    #allow setting of specific atom type radius
    atom_types = ['H', 'C', 'N', 'O']
    pdb.modified_atom_types = atom_types

    suffix = "_OccasRadius"
    occasradius = False
    if basename[-len(suffix):] == suffix:
        occasradius = True
        #if the file has unique radius in occupancy column, use it
        pdb.unique_radius = pdb.occupancy
        pdb.unique_volume = 4/3*np.pi*pdb.unique_radius**3
        pdb.radius = np.copy(pdb.unique_radius)
    else:
        pdb.calculate_unique_volume(use_b=args.use_b)
        pdb.radius = np.copy(pdb.unique_radius)

    if args.center:
        pdb.coords -= pdb.coords.mean(axis=0)

    #write the modified pdb file, and store the 
    #new unique volume/radius value in the occupancy
    #column, to prevent needing to recalculate if wanting
    #to run the fitting again.
    #only write if not already an occasradius pdb file
    pdboutput = basename
    if not occasradius:
        #write new file if it does not have the suffix
        if not args.explicitH:
            pdboutput += '_noH' + suffix + '.pdb'
        else:
            pdboutput += suffix + '.pdb'

        pdbout = copy.deepcopy(pdb)
        pdbout.occupancy = pdb.unique_radius
        pdbout.write(filename=pdboutput)

    #some good guesses for initial radii scale factors
    radii_sf_dict = {'H':1.10113e+00, 
                     'C':1.24599e+00, 
                     'N':1.02375e+00, 
                     'O':1.05142e+00,
                     }
    radii_sf = np.ones(len(atom_types))
    for i in range(len(atom_types)):
        if atom_types[i] in radii_sf_dict.keys():
            radii_sf[i] = radii_sf_dict[atom_types[i]]
        else:
            radii_sf[i] = 1.0

    #adjust volumes if using implicit hydrogens
    for i in range(len(atom_types)):
        if not args.explicitH:
            if atom_types[i]!='H':
                #subtract some volume from the heavy atom for each hydrogen for overlap
                try:
                    pdb.unique_volume[pdb.atomtype==atom_types[i]] -= saxs.volume_of_hydrogen_overlap[atom_types[i]] * pdb.numH[pdb.atomtype==atom_types[i]]
                except:
                    #only currently have CNO calculated
                    pass
    pdb.unique_radius = saxs.sphere_radius_from_volume(pdb.unique_volume)
    pdb.radius = np.copy(pdb.unique_radius)

    for i in range(len(atom_types)):
        #Consider if using implicit hydrogens, to use hydrogen radius saved to pdb 
        if not args.explicitH:
            if atom_types[i]=='H':
                pdb.exvolHradius = saxs.radius['H'] * radii_sf[i]
            else:
                pdb.radius[pdb.atomtype==atom_types[i]] = radii_sf[i] * pdb.unique_radius[pdb.atomtype==atom_types[i]]
        else:
            #set the radii for each atom type in the temporary pdb
            pdb.radius[pdb.atomtype==atom_types[i]] = radii_sf[i] * pdb.unique_radius[pdb.atomtype==atom_types[i]]

    pdb.exvol_type = args.exvol_type

    if not args.use_b:
        pdb.b *= 0

    #set some defaults for the grid size
    #prioritize side length, since this has the biggest effect on the scattering
    #profile calculation accuracy and the interpolation later.
    optimal_side = saxs.estimate_side_from_pdb(pdb)
    optimal_voxel = 1.0
    optimal_nsamples = np.ceil(optimal_side/optimal_voxel).astype(int)
    nsamples_limit = 256

    if args.voxel is not None and args.nsamples is not None and args.side is not None:
        #if v, n, s are all given, side and nsamples dominates
        side = optimal_side
        nsamples = args.nsamples
        voxel = side / nsamples
    elif args.voxel is not None and args.nsamples is not None and args.side is None:
        #if v and n given, voxel and nsamples dominates
        voxel = args.voxel
        nsamples = args.nsamples
        side = voxel * nsamples
    elif args.voxel is not None and args.nsamples is None and args.side is not None:
        #if v and s are given, adjust voxel to match nearest integer value of n
        voxel = args.voxel
        side = args.side
        nsamples = np.ceil(side/voxel).astype(int)
        voxel = side / nsamples
    elif args.voxel is not None and args.nsamples is None and args.side is None:
        #if v is given, voxel thus dominates, so estimate side, calculate nsamples.
        voxel = args.voxel
        nsamples = np.ceil(optimal_side/voxel).astype(int)
        side = voxel * nsamples
        #if n > 256, adjust side length
        if nsamples > nsamples_limit:
            nsamples = nsamples_limit
            side = voxel * nsamples
    elif args.voxel is None and args.nsamples is not None and args.side is not None:
        #if n and s are given, set voxel size based on those
        nsamples = args.nsamples
        side = args.side
        voxel = side / nsamples
    elif args.voxel is None and args.nsamples is not None and args.side is None:
        #if n is given, set side, adjust voxel.
        nsamples = args.nsamples
        side = optimal_side
        voxel = side / nsamples
    elif args.voxel is None and args.nsamples is None and args.side is not None:
        #if s is given, set voxel, adjust nsamples, reset voxel if necessary
        side = args.side
        voxel = optimal_voxel
        nsamples = np.ceil(side/voxel).astype(int)
        if nsamples > nsamples_limit:
            nsamples = nsamples_limit
        voxel = side / nsamples
    elif args.voxel is None and args.nsamples is None and args.side is None:
        #if none given, set side and voxel, adjust nsamples, reset voxel if necessary
        side = optimal_side
        voxel = optimal_voxel
        nsamples = np.ceil(side/voxel).astype(int)
        if nsamples > nsamples_limit:
            nsamples = nsamples_limit
        voxel = side / nsamples

    #make some warnings for certain cases
    side_small_warning = """
        Side length may be too small and may result in undersampling errors."""
    side_way_too_small_warning = """
        Disabling interpolation of I_calc due to severe undersampling."""
    voxel_big_warning = """
        Voxel size is greater than 1 A. This may lead to less accurate I(q) estimates at high q."""
    nsamples_warning = """
        To avoid long computation times and excessive memory requirements, the number of voxels
        has been limited to 256 and the voxel size has been set to {v:.2f},
        which may be too large and lead to less accurate I(q) estimates at high q.""".format(v=voxel)
    optimal_values_warning = """
        To ensure the highest accuracy, manually set the -s option to {os:.2f} and
        the -v option to {ov:.2f}, which will set -n option to {on:d} and thus 
        may take a long time to calculate and use a lot of memory.
        If that requires too much computation, set -s first, and -n to the 
        limit you prefer (n=512 may approach an upper limit for many computers).
        """.format(os=optimal_side, ov=optimal_voxel, on=optimal_nsamples)

    warn = False
    if side < optimal_side:
        print(side_small_warning)
        warn = True
    if side < 2/3*optimal_side:
        print(side_way_too_small_warning)
        args.Icalc_interpolation = False
        warn = True
    if voxel > optimal_voxel:
        print(voxel_big_warning)
        warn = True
    if nsamples < optimal_nsamples:
        print(nsamples_warning)
        warn = True
    if warn:
        print(optimal_values_warning)

    print("Optimal Side length >= %.2f" % optimal_side)
    print("Optimal N samples   >= %d" % optimal_nsamples)
    print("Optimal Voxel size  <= %.4f" % optimal_voxel)

    logging.info('Optimal Side length (angstroms): %.2f', optimal_side)
    logging.info('Optimal N samples: %d', optimal_nsamples)
    logging.info('Optimal Voxel size: (angstroms): %.4f', optimal_voxel)

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
        resolution = 0.30 #this helps with voxel sampling issues 
    elif args.resolution is not None:
        resolution = args.resolution
    else:
        resolution = 0.0

    print("Actual  Side length  = %.2f" % side)
    print("Actual  N samples    = %d" % n)
    print("Actual  Voxel size   = %.4f" % dx)

    logging.info('Actual  Side length (angstroms): %.2f', side)
    logging.info('Actual  N samples: %d', n)
    logging.info('Actual  Voxel size: (angstroms): %.4f', dx)

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

    latt_correction = 1. #(np.sinc(qx/2/(np.pi)) * np.sinc(qy/2/(np.pi)) * np.sinc(qz/2/(np.pi)))**2

    rho_invacuo, support = saxs.pdb2map_multigauss(pdb,x=x,y=y,z=z,resolution=resolution,use_b=args.use_b,ignore_waters=args.ignore_waters)

    rho0 = args.rho0

    #calculate the volume of a shell of water diameter
    #this assumes a single layer of hexagonally packed water molecules on the surface
    r_water = 1.4 
    uniform_shell = saxs.calc_uniform_shell(pdb,x,y,z,thickness=r_water)
    water_shell_idx = uniform_shell.astype(bool)
    V_shell = water_shell_idx.sum() * dV
    N_H2O_in_shell = 2/(3**0.5) * V_shell / (2*r_water)**3
    V_H2O = 4/3*np.pi*r_water**3
    V_H2O_in_shell = N_H2O_in_shell * V_H2O
    # print("Estimated number of H2O molecules in hydration shell: %d " % N_H2O_in_shell)
    print()

    shell_contrast = args.shell_contrast
    protein_with_shell_support = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.vdW,probe=2*r_water)

    if args.shell_mrcfile is not None:
        #allow user to provide mrc filename to read in a custom shell
        rho_shell, sidex = saxs.read_mrc(args.shell_mrcfile)
        rho_shell *= dV #assume mrc file is in units of density, convert to electron count
        print(sidex, side)
        if (sidex != side) or (rho_shell.shape[0] != x.shape[0]):
            print("Error: shell_mrcfile does not match grid.")
            print("Use denss.mrcops.py to resample onto the desired grid.")
            exit()
    elif args.shell_type == "gaussian":
        #the default is gaussian type shell
        #generate initial hydration shell
        thickness = max(1.0,dx) #in angstroms
        shell_sigma = (resolution+dx) / dx #convert to pixel units

        uniform_shell = saxs.calc_uniform_shell(pdb,x,y,z,thickness=thickness)
        uniform_shell *= 1.0/uniform_shell.sum() #scale to one total
        rho_shell = saxs.calc_gaussian_shell(sigma=shell_sigma,uniform_shell=uniform_shell)

        #remove shell density that overlaps with protein
        protein = saxs.pdb2support_fast(pdb,x,y,z,radius=pdb.unique_radius,probe=0.0)
        rho_shell[protein] = 0.0

        #estimate initial shell scale based on contrast using mean density
        shell_mean_density = np.mean(rho_shell[water_shell_idx]) / dV
        #scale the mean density of the invacuo shell to match the desired mean density
        rho_shell *= shell_contrast / shell_mean_density
        #shell should still be in electron count units

    elif args.shell_type == "uniform":
        rho_shell = water_shell_idx * (shell_contrast)
        rho_shell *= dV #convert to electron units
    else:
        print("Error: no valid shell_type given. Disabling hydration shell.")
        rho_shell = x*0.0

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

    param_names = ['rho0', 'shell_contrast']

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
        shell_contrast_guess = args.shell_contrast
        bounds[1,0] = -np.inf
        bounds[1,1] = np.inf
        fit_params = True
    else:
        shell_contrast_guess = args.shell_contrast
        bounds[1,0] = shell_contrast_guess
        bounds[1,1] = shell_contrast_guess

    if args.data is None:
        #cannot fit parameters without data
        fit_params = False

    if not args.fit_all:
        #disable all fitting if requested
        fit_params = False


    #set up dictionary to pass around functions for convenience
    pdb2mrc_dict = {}
    pdb2mrc_dict['pdb'] = pdb
    pdb2mrc_dict['x'] = x
    pdb2mrc_dict['y'] = y
    pdb2mrc_dict['z'] = z
    pdb2mrc_dict['qbinsc'] = qbinsc
    pdb2mrc_dict['qblravel'] = qblravel
    pdb2mrc_dict['xcount'] = xcount
    pdb2mrc_dict['Iq_exp'] = Iq_exp
    pdb2mrc_dict['Icalc_interpolation'] = args.Icalc_interpolation
    pdb2mrc_dict['ignore_waters'] = args.ignore_waters
    pdb2mrc_dict['rho0'] = rho0_guess
    pdb2mrc_dict['exvol_type'] = args.exvol_type
    pdb2mrc_dict['shell_contrast'] = shell_contrast_guess

    #save an array of indices containing only desired q range for speed
    if args.data:
        qmax4calc = q_exp.max()*1.1
    else:
        qmax4calc = qx_.max()*1.1
    qidx = np.where((qr<=qmax4calc))
    pdb2mrc_dict['qidx'] = qidx

    params_guess = np.zeros(len(param_names))
    params_guess[0] = rho0_guess
    params_guess[1] = shell_contrast_guess

    pdb2mrc_dict['penalty_weight'] = args.penalty_weight
    pdb2mrc_dict['penalty_weights'] = args.penalty_weights

    #precalculate structure factors F
    #in vacuo, F_v
    F_invacuo = saxs.myfftn(rho_invacuo)
    #perform B-factor sharpening to correct for B-factor sampling workaround
    F_invacuo *= np.exp(-(-saxs.u2B(resolution))*(qr/(4*np.pi))**2)

    #exvol F_exvol
    rho_exvol = saxs.pdb2F_calc_exvol_with_modified_params(params_guess, pdb2mrc_dict)
    F_exvol = saxs.myfftn(rho_exvol)

    #shell invacuo F_shell
    F_shell = saxs.myfftn(rho_shell)

    pdb2mrc_dict['rho_invacuo'] = rho_invacuo
    pdb2mrc_dict['rho_exvol'] = rho_exvol
    pdb2mrc_dict['rho_shell'] = rho_shell
    pdb2mrc_dict['F_invacuo'] = F_invacuo
    pdb2mrc_dict['F_exvol'] = F_exvol
    pdb2mrc_dict['F_shell'] = F_shell

    if fit_params:
        params_target = params_guess
        pdb2mrc_dict['params_target'] = params_target
        print(["scale_factor"], param_names, ["penalty"], ["chi2"])
        print("-"*100)
        results = optimize.minimize(saxs.pdb2F_calc_score_with_modified_params, params_guess,
            args = (pdb2mrc_dict),
            bounds = bounds,
            method=args.method,
            options=eval(args.minopts),
            # method='L-BFGS-B', options={'eps':0.001},
            )
        optimized_params = results.x
        optimized_chi2 = results.fun
    else:
        optimized_params = params_guess
        optimized_chi2 = "None"

    params = optimized_params

    # logging.info('Estimated number of waters in shell: %d', N_H2O_in_shell)
    logging.info('Final Parameter Values:')

    print()
    print("Final parameter values:")
    for i in range(len(params)):
        print("%s : %.5e" % (param_names[i],params[i]))
        logging.info("%s : %.5e" % (param_names[i],params[i]))

    rho_insolvent = saxs.pdb2F_calc_rho_with_modified_params(params,pdb2mrc_dict)
    exvol = saxs.pdb2F_calc_exvol_with_modified_params(params,pdb2mrc_dict)
    shell = saxs.pdb2mrc_calc_shell_with_modified_params(params,pdb2mrc_dict)
    shell_mean_density = np.mean(shell[water_shell_idx])
    I_calc = saxs.pdb2F_calc_I_with_modified_params(params,pdb2mrc_dict)
    qbinsc = pdb2mrc_dict['qbinsc']
    Iq_calc = np.vstack((qbinsc, I_calc, I_calc*.01 + I_calc[0]*0.002)).T

    if args.data:
        optimized_chi2, exp_scale_factor, fit = saxs.calc_chi2(Iq_exp, Iq_calc,interpolation=args.Icalc_interpolation,return_sf=True,return_fit=True)
        print("Scale factor: %.5e " % exp_scale_factor)
        print("chi2 of fit:  %.5e " % optimized_chi2)
        logging.info("Scale factor: %.5e " % exp_scale_factor)
        logging.info("chi2 of fit:  %.5e " % optimized_chi2)

    print("Calculated average radii:")
    logging.info("Calculated average radii:")
    for i in range(len(atom_types)):
        #try using a scale factor for radii instead
        if atom_types[i]=='H' and not args.explicitH:
            mean_radius = pdb.exvolHradius
        else:
            mean_radius = pdb.radius[pdb.atomtype==atom_types[i]].mean()
        print("%s: %.3f"%(atom_types[i],mean_radius))
        logging.info("%s: %.3f"%(atom_types[i],mean_radius))

    print("Calculated excluded volume: %.2f"%(np.sum( 4/3*np.pi*pdb.radius**3) + 4/3*np.pi*pdb.exvolHradius**3*pdb.numH.sum()))
    logging.info("Calculated excluded volume: %.2f"%(np.sum( 4/3*np.pi*pdb.radius**3) + 4/3*np.pi*pdb.exvolHradius**3*pdb.numH.sum()))

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
            ax0.plot(q, Ic, '-',c='red',label=basename+'.pdb2mrc2sas.fit \n' + r'$\chi^2 = $ %.2f'%optimized_chi2)
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
        # saxs.write_mrc((protein)*1.0,side,output+"_proteinsupport.mrc")
        # saxs.write_mrc((protein_with_shell_support)*1.0,side,output+"_supportwithshell.mrc")






