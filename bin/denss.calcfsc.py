#!/usr/bin/env python
#
#    denss.calcfsc.py
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

import os, argparse, sys, imp
import logging
import numpy as np
from scipy import ndimage
from saxstats._version import __version__
import saxstats.saxstats as saxs
try:
    imp.find_module('matplotlib')
    import matplotlib.pyplot as plt
    matplotlib_found = True
except ImportError:
    matplotlib_found = False

parser = argparse.ArgumentParser(description="A tool for calculating the Fourier Shell Correlation between two pre-aligned MRC formatted electron density maps", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f1", "--file1", type=str, help="Electron density filename (.mrc)")
parser.add_argument("-f2", "--file2", type=str, help="Electron density filename (.mrc)")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Plot the profile (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not plot the profile. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
if matplotlib_found:
    parser.set_defaults(plot=True)
else:
    parser.set_defaults(plot=False)
args = parser.parse_args()

if __name__ == "__main__":

    if args.output is None:
        basename, ext = os.path.splitext(args.file1)
        output = basename + '_fsc'
    else:
        output = args.output


    rho1, side1 = saxs.read_mrc(args.file1)
    rho2, side2 = saxs.read_mrc(args.file2)
    if rho1.shape[0] != rho2.shape[0]:
        print "Shape of rho1 and rho2 are not equal."
        sys.exit()
    if side1 != side2:
        print "Side length of rho1 and rho2 are not equal."
        sys.exit()

    fsc = saxs.calc_fsc(rho1,rho2,side1)

    np.savetxt(output+'.dat', fsc, delimiter=' ', fmt='% .5e')

    x = np.linspace(fsc[0,0],fsc[-1,0],100)
    y = np.interp(x, fsc[:,0], fsc[:,1])
    resi = np.argmin(y>=0.5)
    resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
    resn = round(float(1./resx),1)
    print "Resolution: %.1f" % resn, u'\u212B'.encode('utf-8')

    if args.plot:
        plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')
        plt.plot(fsc[:,0],fsc[:,1],'o')
        plt.plot(x,y,'k-')
        plt.plot([resx],[0.5],'ro',label='Resolution = '+str(resn)+r'$\mathrm{\AA}$')
        plt.legend()
        plt.xlabel('Resolution (1/$\mathrm{\AA}$)')
        plt.ylabel('Fourier Shell Correlation')
        plt.savefig(output,ext='png',dpi=150)
        plt.close()







