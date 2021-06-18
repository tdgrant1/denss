#!/usr/bin/env python
#
#    denss.mrcops.py
#    A tool for calculating simple scattering profiles
#    from MRC formatted electron density maps
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
import os, argparse, sys, imp
import logging
import numpy as np
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for performing basic operations on MRC formatted electron density maps", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--file", type=str, help="Electron density filename (.mrc)")
parser.add_argument("-v", "--voxel", default=None, type=float, help="Desired length of voxel of map (resamples map, before any padding)")
parser.add_argument("-n", "--n", default=None, type=int, help="Desired number of samples (creates cubic map by padding with zeros or clipping, after any resampling)")
parser.add_argument("-s", "--side", default=None, type=float, help="Desired length of side of map (creates cubic map by padding with zeros or clipping, after any resampling)")
parser.add_argument("-t","--threshold", default=None, type=float, help="Minimum density threshold (given as e-/A^3; sets lesser values to zero).")
parser.add_argument("-ne","--ne", default=None, type=float, help="Desired number of electrons in map.")
parser.add_argument("-zflip","--zflip", action="store_true", help="Generate the enantiomer by flipping map over Z axis.")
parser.add_argument("-u", "--units", default=None, type=str, help="Change units (\"a\": [from nm to angstrom] or \"nm\": [from angstrom to nanometer])")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
args = parser.parse_args()

if __name__ == "__main__":

    if args.output is None:
        basename, ext = os.path.splitext(args.file)
        output = basename + '_resampled'
    else:
        output = args.output

    rho, (a,b,c) = saxs.read_mrc(args.file,returnABC=True)
    #float32 is not precise enough for some calculations, convert to float64
    rho = rho.astype(np.float64)
    vx, vy, vz = np.array((a,b,c))/np.array(rho.shape)
    #assume electron density map is given in density (e-/A^3), not number of electrons
    #multiply by voxel volume to get total number of electrons
    V = a*b*c
    dV = vx*vy*vz
    ne = np.sum(rho) * dV

    print("Original number of electrons:", ne)
    print("prezoom")
    print("Shape:  ", rho.shape)
    print("Sides:  ", a, b, c)
    print("Voxels: ", vx, vy, vz)

    #change units. By default sf (scale factor) is 1.0 so no change
    sf = 1.0
    if args.units == "a":
        sf = 10.0
    if args.units == "nm":
        sf = 0.1
    a *= sf
    b *= sf
    c *= sf
    vx *= sf
    vy *= sf
    vz *= sf
    V = a*b*c
    dV = vx*vy*vz

    if args.voxel is None:
        voxel = min((vx, vy, vz))
    else:
        voxel = args.voxel

    if args.voxel is not None:
        #only resample if voxel option is defined by user
        rho = saxs.zoom_rho(rho,(vx,vy,vz),voxel)
        #the zooming isn't exact due to integer voxels
        #so reset voxel sizes here
        vx, vy, vz = np.array((a,b,c))/np.array(rho.shape)
        dV = vx*vy*vz

    print("postzoom")
    print("Shape:  ", rho.shape)
    print("Sides:  ", a, b, c)
    print("Voxels: ", vx, vy, vz)

    if args.side is None:
        newside = max((a,b,c))
    else:
        newside = args.side

    if args.n is not None:
        n = args.n
    else:
        n = max((int(newside/vx),int(newside/vy),int(newside/vz)))
        if n%2==1: n+=1

    newshape = (n, n, n)

    if (args.side is not None) or (args.n is not None):
        rho = saxs.pad_rho(rho,newshape)
        a,b,c = vx * newshape[0], vy * newshape[1], vz*newshape[2]
    print("postpad")
    print("Shape:  ", rho.shape)
    print("Sides:  ", a, b, c)
    print("Voxels: ", vx, vy, vz)


    if args.threshold is not None:
        #note that threshold must be given as density in e-/A^3
        rho[rho < args.threshold] = 0

    #rescale map to total number of electrons desired (args.ne if set, otherwise original number)
    if args.ne is not None:
        ne = args.ne
    
    #now that we're in density, rather than total e-, must also multiply by total volume
    #won't keep this in the args.ne condition since we want to at least rescale to the 
    #original total number of electrons even if args.ne is not given
    rho *= ne/np.sum(rho) #total number of electrons
    rho /= dV #convert to density

    if args.zflip:
        rho = rho[:,:,::-1]

    print("Final number of electrons:", np.sum(rho)*dV)

    saxs.write_mrc(rho,(a,b,c),filename=output+'.mrc')














