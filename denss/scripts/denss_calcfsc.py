#!/usr/bin/env python
#
#    denss_calcfsc.py
#    A tool for calculating the Fourier Shell Correlation
#    between two pre-aligned MRC formatted electron density maps
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
import os, argparse, sys
import numpy as np

import denss

def main():
    parser = argparse.ArgumentParser(description="A tool for calculating the Fourier Shell Correlation between two pre-aligned MRC formatted electron density maps", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--file", type=str, help="Electron density filename (.mrc)")
    parser.add_argument("-ref", "--ref", type=str, help="Reference electron density filename (.mrc)")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Plot the profile (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not plot the profile. (Default if Matplotlib does not exist)")
    parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    if args.plot:
        #if plotting is enabled, try to import matplotlib
        #if import fails, set plotting to false
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            args.plot = False

    if args.output is None:
        fname_nopath = os.path.basename(args.file)
        basename, ext = os.path.splitext(fname_nopath)
        output = basename + '_fsc'
    else:
        output = args.output


    rho, side = denss.read_mrc(args.file)
    refrho, refside = denss.read_mrc(args.ref)
    if rho.shape[0] != refrho.shape[0]:
        print("Shape of rho and ref are not equal.")
        sys.exit()
    if side != refside:
        print("Side length of rho and ref are not equal.")
        sys.exit()

    fsc = denss.calc_fsc(rho,refrho,side)
    rscc = denss.real_space_correlation_coefficient(rho,refrho)
    print("RSCC: %.3e"%rscc)
    resn, x, y, resx = denss.fsc2res(fsc, return_plot=True)
    if np.min(fsc[:,1]) > 0.5:
        print("Resolution: < %.1f A (maximum possible)" % resn)
    else:
        print("Resolution: %.1f A" % resn)

    np.savetxt(output+'.dat', fsc, delimiter=' ', fmt='% .5e', header="1/resolution, FSC; Resolution=%.1f A" % resn)

    if args.plot:
        plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')
        plt.plot(fsc[:,0],fsc[:,1],'o')
        plt.plot(x,y,'k-')
        plt.plot([resx],[0.5],'ro',label='Resolution = '+str(resn)+r'$\mathrm{\AA}$')
        plt.legend()
        plt.xlabel('Resolution (1/$\mathrm{\AA}$)')
        plt.ylabel('Fourier Shell Correlation')
        plt.savefig(output+'.png',dpi=150)
        plt.close()


if __name__ == "__main__":
    main()




