#!/usr/bin/env python
#
#    denss_generate_reference.py
#    A tool for aligning and averaging multiple electron density maps.
#
#    Part of DENSS
#    DENSS: DENsity from Solution Scattering
#    A tool for calculating an electron density map from solution scattering data
#
#    Tested using Anaconda / Python 2.7
#
#    Authors: Thomas D. Grant
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
import sys, os, argparse
import numpy as np

import denss


def main():
    parser = argparse.ArgumentParser(
        description="A tool for generating a reference from multiple electron density maps.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version", version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--files", type=str, nargs="+", help="List of MRC files")
    parser.add_argument("-o", "--output", type=str, help="output filename prefix")
    parser.add_argument("-j", "--cores", type=int, default=1,
                        help="Number of cores used for parallel processing. (default: 1)")
    parser.add_argument("-en_on", "--enantiomer_on", action="store_true", dest="enan",
                        help="Generate and select best enantiomers (default). ")
    parser.add_argument("-en_off", "--enantiomer_off", action="store_false", dest="enan",
                        help="Do not generate and select best enantiomers.")
    parser.add_argument("--thorough_alignment_on", action="store_true", dest="thorough_alignment",
                        help="Perform thorough alignment (uses Nelder-Mead) (default: on). ")
    parser.add_argument("--thorough_alignment_off", action="store_false", dest="thorough_alignment",
                        help="Do not perform thorough alignment (uses simple gradient descent). ")

    parser.set_defaults(enan=True)
    parser.set_defaults(thorough_alignment=True)
    args = parser.parse_args()

    if args.output is None:
        fname_nopath = os.path.basename(args.files[0])
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    nmaps = len(args.files)

    allrhos = []
    sides = []
    for file in args.files:
        rho, side = denss.read_mrc(file)
        allrhos.append(rho)
        sides.append(side)
    allrhos = np.array(allrhos)
    sides = np.array(sides)

    if nmaps < 2:
        print("Not enough maps to generate reference. Please input more maps again...")
        sys.exit(1)

    print(" Generating reference...")
    try:
        # Pass the arguments correctly using keywords
        # We only need the first returned value (refrho)
        refrho, _, _ = denss.iterative_average(
            allrhos,
            cores=args.cores,
            thorough=args.thorough_alignment,
            enan=args.enan
        )

        if refrho is None:
            print("Reference generation was aborted.")
            sys.exit(1)

        denss.write_mrc(refrho, sides[0], output + "_reference.mrc")
        print("Reference map saved to: {}_reference.mrc".format(output))

    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()