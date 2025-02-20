#!/usr/bin/env python
#
#    denss_align_and_average.py
#    A tool for aligning and averaging multiple electron density maps.
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

from __future__ import print_function
import sys, os, argparse
import numpy as np

import denss


def main():
    parser = argparse.ArgumentParser(description="A tool for selecting the enantiomer with the same handedness as the reference.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="MRC file")
    parser.add_argument("-ref", "--ref", default = None, type=str, help="Reference filename (.mrc file, required)")
    parser.add_argument("-o", "--output", type=str, help="output filename prefix")
    parser.add_argument("-j", "--cores", type=int, default = 1, help="Number of cores used for parallel processing. (default: 1)")
    parser.set_defaults(enan = True)
    parser.set_defaults(center = True)
    args = parser.parse_args()


    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename + "_enan"
    else:
        output = args.output

    rho, side = denss.read_mrc(args.file)

    if args.ref is not None:
        #allow input of reference for enantiomer selection
        if args.ref.endswith('.mrc'):
            refrho, refside = denss.read_mrc(args.ref)
        else:
            print("Invalid reference filename given. .mrc file required")
            sys.exit(1)

        rho_nsamples = rho.shape[0]
        rho_voxel = side/rho_nsamples
        ref_nsamples = refrho.shape[0]
        ref_voxel = refside/ref_nsamples

        if (not np.isclose(side, refside) or
            not np.isclose(rho_voxel, ref_voxel) or
            not np.isclose(rho_nsamples, ref_nsamples)):
            print("reference density dimensions do not match given options.")
            sys.exit()
    else:
        print("reference file required")
        sys.exit(1)

    print(" Selecting best enantiomer...")
    try:
        best_enan, score = denss.select_best_enantiomer(refrho=refrho, rho=rho)
        denss.write_mrc(best_enan, side, output+'.mrc')
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()

