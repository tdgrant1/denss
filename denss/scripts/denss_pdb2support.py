#!/usr/bin/env python
#
#    denss_pdb2support.py
#    A tool for calculating unitary electron density maps from pdb files.
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

import denss
import numpy as np
import sys, argparse, os
import time


def main():
    parser = argparse.ArgumentParser(description="A tool for calculating simple electron density maps from pdb files.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="Atomic model as a .pdb file for input.")
    parser.add_argument("-s", "--side", default=None, type=float, help="Desired side length of real space box (default=None).")
    parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired voxel size (default=None)")
    parser.add_argument("-n", "--nsamples", default=None, type=int, help="Desired number of samples per axis (default=None)")
    parser.add_argument("-r", "--resolution", default=None, type=float, help="Desired resolution (B-factor-like atomic displacement.)")
    parser.add_argument("-ongrid", "--ongrid", "--onGrid", dest='ongrid', default=None, type=str, help="Filename of mrc file to match grid size to. (default=None)")
    parser.add_argument("-vdW", "--vdW", "-vdw", "--vdw", dest="vdW", default=None, nargs='+', type=float, help="van der Waals radii of atom_types (for H, C, N, O, by default). (optional)")
    parser.add_argument("-atom_types", "--atom_types", default=['H', 'C', 'N', 'O'], nargs='+', type=str, help="Atom types to allow modification of van der waals radii (space separated list, default = H C N O). (optional)")
    parser.add_argument("-probe", "--probe", default=None, type=float, help="Probe distance (default=2.80, i.e. water diameter)")
    parser.add_argument("-b", "--b", "--use_b", dest="use_b", action="store_true", help="Include B-factors in atomic model (optional, default=False)")
    parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
    parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
    parser.add_argument("--ignore_waters", dest="ignore_waters", action="store_true", help="Ignore waters.")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix (default=basename_pdb)")
    parser.set_defaults(ignore_waters = False)
    parser.set_defaults(center = True)
    parser.set_defaults(plot=True)
    parser.set_defaults(use_b=False)
    args = parser.parse_args()

    start = time.time()

    command = ' '.join(sys.argv)

    fname_nopath = os.path.basename(args.file)
    basename, ext = os.path.splitext(fname_nopath)

    if args.output is None:
        output = basename + "_pdb"
    else:
        output = args.output

    pdb = denss.PDB(args.file,ignore_waters=args.ignore_waters)
    if args.center:
        pdboutput = basename+"_centered.pdb"
        pdb.coords -= pdb.coords.mean(axis=0)
        pdb.write(filename=pdboutput)

    #allow setting of specific atom type radius
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
        vdWs = args.vdW
    else:
        vdWs = [denss.vdW.get(key) for key in atom_types]
        pdb.radius = pdb.vdW

    if not args.use_b:
        pdb.b *= 0

    if args.ongrid is not None:
        rho, (a,b,c) = denss.read_mrc(args.ongrid, returnABC=True)
        #check that grid is a cube
        if not np.allclose(rho.shape, rho.shape[0]) or not np.allclose([a,b,c], a):
            print("mrc file for --ongrid option is not a cube. Please resample using denss_mrcops.py first.")
            print(rho.shape, (a,b,c))
            exit()
        else:
            nsamples = rho.shape[0]
            side = a
            voxel = side / nsamples
    #set some sane defaults for the grid size
    elif args.voxel is not None and args.nsamples is not None and args.side is not None:
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
        optimal_side = denss.estimate_side_from_pdb(pdb)
        nsamples = np.ceil(optimal_side/voxel).astype(int)
        side = voxel * nsamples
        #if n > 256, adjust side length
        if nsamples > 256:
            nsamples = 256
            side = voxel * nsamples
        if side < optimal_side:
            print("Warning: side length may be too small and may result in undersampling errors.")
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
        optimal_side = denss.estimate_side_from_pdb(pdb)
        if side < optimal_side:
            print("Warning: side length may be too small and may result in undersampling errors.")
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
        optimal_side = denss.estimate_side_from_pdb(pdb)
        nsamples = np.ceil(optimal_side/voxel).astype(int)
        if nsamples > 256:
            nsamples = 256
        side = voxel * nsamples
        if side < optimal_side:
            print("This must be a large particle. To ensure the highest accuracy, manually set")
            print("the -v option to 1 and the -s option to %.2f" % optimal_side)
            print("This will set -n option to %d and thus may take a long time to calculate." % nsamples)
            print("To avoid long computation times, the side length has been set to %.2f for now," % side)
            print("which may be too small and may result in undersampling errors.")

    halfside = side/2
    n = int(side/voxel)
    #want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n%2==1: n += 1
    dx = side/n
    dV = dx**3
    # x_ = np.linspace(-halfside,halfside,n)
    x_ = np.linspace(-(n//2)*dx,(n//2-1)*dx,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')

    xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))

    if args.resolution is None and not args.use_b:
        resolution = 0.3 * dx 
    elif args.resolution is not None:
        resolution = args.resolution
    else:
        resolution = 0.0

    r_water = 1.4 
    if args.probe is None:
        probe = 2 * r_water
    else:
        probe = args.probe

    print("Side length: %.2f" % side)
    print("N samples:   %d" % n)
    print("Voxel size:  %.4f" % dx)

    #for now, add in resolution to pdb object to enable passing between functions easily.
    pdb.resolution = resolution

    support = denss.pdb2support_fast(pdb,x,y,z,radius=pdb.radius,probe=probe)

    #write output
    denss.write_mrc(support*1.0,side,output+"_support.mrc")


if __name__ == "__main__":
    main()











