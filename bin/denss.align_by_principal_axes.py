#!/usr/bin/env python
#
#    denss.align_by_principal_axes.py
#    A tool for aligning an electron density map to another electron density
#    map based only on alignment of principal axes (no minimization).
#
#    Part of DENSS
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Authors: Thomas D. Grant, Nhan D. Nguyen
#    Email:  <tgrant@hwi.buffalo.edu>, <ndnguyen20@wabash.edu>
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

import os, sys, logging
import numpy as np
from scipy import ndimage
import argparse
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for aligning an electron density map to another electron density map based only on alignment of principal axes (no minimization).", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="MRC file for alignment to reference principal axes.")
parser.add_argument("-ref", "--ref", default = None, type=str, help="Reference (.mrc or .pdb file (map will be calculated from PDB))")
parser.add_argument("-o", "--output", default = None, type=str, help="output filename prefix")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB reference (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB reference.")
parser.add_argument("-r", "--resolution", default=15.0, type=float, help="Desired resolution (i.e. Gaussian width sigma) of map calculated from PDB file.")
parser.set_defaults(center = True)
args = parser.parse_args()

if __name__ == "__main__":

    if args.output is None:
        basename, ext = os.path.splitext(args.file)
        output = basename+"_alignedbyPA"
    else:
        output = args.output

    logging.basicConfig(filename=output+'.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', __version__)
    logging.info('Map filename(s): %s', args.file)
    logging.info('Reference filename: %s', args.ref)

    rho, side = saxs.read_mrc(args.file)

    if args.ref is None:
        print "Need reference file (.mrc or .pdb)"
        sys.exit(1)
    else:
        if args.ref.endswith('.pdb'):
            logging.info('Center PDB reference: %s', args.center)
            logging.info('PDB reference map resolution: %.2f', args.resolution)
            refbasename, refext = os.path.splitext(args.ref)
            refoutput = refbasename+"_centered.pdb"
            refside = side
            voxel = (refside/movrho.shape)[0]
            halfside = refside/2
            n = int(refside/voxel)
            dx = refside/n
            x_ = np.linspace(-halfside,halfside,n)
            x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
            xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
            pdb = saxs.PDB(args.ref)
            if args.center:
                pdb.coords -= pdb.coords.mean(axis=0)
                pdb.write(filename=refoutput)
            refrho = saxs.pdb2map_gauss(pdb,xyz=xyz,sigma=args.resolution)
            refrho = refrho*np.sum(movrho)/np.sum(refrho)
            saxs.write_mrc(refrho,side,filename=refbasename+'_pdb.mrc')
        if args.ref.endswith('.mrc'):
            refrho, refside = saxs.read_mrc(args.ref)
        if (not args.ref.endswith('.mrc')) and (not args.ref.endswith('.pdb')):
            print "Invalid reference filename given. .mrc or .pdb file required"
            sys.exit(1)

    aligned = saxs.principal_axis_alignment(refrho,rho)

    saxs.write_mrc(aligned, side, output+'.mrc')
    print "%s.mrc written. " % (output,)

    logging.info('END')










