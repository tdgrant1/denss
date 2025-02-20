#!/usr/bin/env python
#
#    denss_align2xyz.py
#    A tool for aligning an electron density map such that its principal
#    axes of inertia are aligned with the x,y,z axes.
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
import os, sys, logging
import argparse

import denss

def main():
    parser = argparse.ArgumentParser(description="A tool for aligning an electron density map such that its principal axes of inertia are aligned with the x,y,z axes.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="List of MRC files for alignment to reference.")
    parser.add_argument("-o", "--output", default = None, type=str, help="output filename prefix")
    args = parser.parse_args()

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename+"_aligned2xyz"
    else:
        output = args.output

    logging.basicConfig(filename=output+'.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Command: %s', ' '.join(sys.argv))
    #logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', denss.__version__)
    logging.info('Map filename(s): %s', args.file)

    rho, side = denss.read_mrc(args.file)

    aligned = denss.align2xyz(rho)

    denss.write_mrc(aligned, side, output+'.mrc')
    print("%s.mrc written. " % (output,))

    logging.info('END')


if __name__ == "__main__":
    main()







