#!/usr/bin/env python
#
#    pdb2mrc.py
#    A tool for calculating simple electron density maps from pdb files

#    Part of the DENSS package
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2017 The Research Foundation for SUNY
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

from saxstats._version import __version__
import saxstats.saxstats as saxs
import numpy as np
import sys, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="PDB filename")
parser.add_argument("-s", "--side", default=100., type=float, help="Desired length real space box side")
parser.add_argument("-v", "--voxel", default=5., type=float, help="Desired voxel size")
parser.add_argument("-r", "--resolution", default=10.0, type=float, help="Desired resolution (i.e. Gaussian width sigma)")
parser.add_argument("-c", "--center", action="store_true", help="Center molecule before calculating map. (default False)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
args = parser.parse_args()


if args.output is None:
    basename, ext = os.path.splitext(args.file)
    output = basename + "_pdb"
else:
    output = args.output

side = args.side
voxel = args.voxel
halfside = side/2
n = int(side/voxel)
#want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
if n%2==1: n += 1
dx = side/n
x_ = np.linspace(-halfside,halfside,n)
x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')

xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
pdb = saxs.PDB(args.file)
if args.center:
    pdb.coords -= pdb.coords.mean(axis=0)

rho = saxs.pdb2map_gauss(pdb,xyz=xyz,sigma=args.resolution)
print
saxs.write_mrc(rho,side,output+".mrc")






