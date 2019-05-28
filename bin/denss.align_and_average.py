#!/usr/bin/env python
#
#    denss.align_and_average.py
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

import sys, os, argparse, logging
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from saxstats._version import __version__
import saxstats.saxstats as saxs

parser = argparse.ArgumentParser(description="A tool for aligning and averaging multiple electron density maps.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=__version__))
parser.add_argument("-f", "--files", type=str, nargs="+", help="List of MRC files")
parser.add_argument("-ref", "--ref",default = None, type=str, help="Reference filename (.mrc or .pdb file, optional)")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
parser.add_argument("-r", "--resolution", default=15.0, type=float, help="Desired resolution (i.e. Gaussian width sigma) of map calculated from PDB file.")
parser.add_argument("-o", "--output", type=str, help="output filename prefix")
parser.add_argument("-j", "--cores", type=int, default = 1, help="Number of cores used for parallel processing. (default: 1)")
parser.add_argument("-en_on", "--enantiomer_on", action = "store_true", dest="enan", help="Generate and select best enantiomers (default). ")
parser.add_argument("-en_off", "--enantiomer_off", action = "store_false", dest="enan", help="Do not generate and select best enantiomers.")
parser.set_defaults(enan = True)
parser.set_defaults(center = True)
args = parser.parse_args()

if __name__ == "__main__":

    if args.output is None:
        basename, ext = os.path.splitext(args.files[0])
        output = basename
    else:
        output = args.output

    logging.basicConfig(filename=output+'_final.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', __version__)
    logging.info('Map filename(s): %s', args.files)
    logging.info('Reference filename: %s', args.ref)
    logging.info('Enantiomer selection: %s', args.enan)

    nmaps = len(args.files)

    allrhos = []
    sides = []
    for file in args.files:
        rho, side = saxs.read_mrc(file)
        allrhos.append(rho)
        sides.append(side)
    allrhos = np.array(allrhos)
    sides = np.array(sides)

    if nmaps<2:
        print "Not enough maps to align. Please input more maps again..."
        sys.exit(1)

    if args.ref is not None:
        #allow input of reference structure
        if args.ref.endswith('.pdb'):
            logging.info('Center PDB reference: %s', args.center)
            logging.info('PDB reference map resolution: %.2f', args.resolution)
            refbasename, refext = os.path.splitext(args.ref)
            refoutput = refbasename+"_centered.pdb"
            refside = sides[0]
            voxel = (refside/allrhos[0].shape)[0]
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
            refrho = refrho*np.sum(allrhos[0])/np.sum(refrho)
            saxs.write_mrc(refrho,sides[0],filename=refbasename+'_pdb.mrc')
        if args.ref.endswith('.mrc'):
            refrho, refside = saxs.read_mrc(args.ref)
        if (not args.ref.endswith('.mrc')) and (not args.ref.endswith('.pdb')):
            print "Invalid reference filename given. .mrc or .pdb file required"
            sys.exit(1)

    if args.enan:
        print " Selecting best enantiomers..."
        try:
            if args.ref:
                allrhos, scores = saxs.select_best_enantiomers(allrhos, refrho=refrho, cores=args.cores)
            else:
                allrhos, scores = saxs.select_best_enantiomers(allrhos, refrho=allrhos[0], cores=args.cores)
        except KeyboardInterrupt:
            sys.exit(1)

    if args.ref is None:
        print " Generating reference..."
        try:
            refrho = saxs.binary_average(allrhos, args.cores)
        except KeyboardInterrupt:
            sys.exit(1)

    print " Aligning all maps to reference..."
    try:
        aligned, scores = saxs.align_multiple(refrho, allrhos, args.cores)
    except KeyboardInterrupt:
        sys.exit(1)

    #filter rhos with scores below the mean - 2*standard deviation.
    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean - 2*std
    filtered = np.empty(len(scores),dtype=str)
    print
    print "Mean of correlation scores: %.3f" % mean
    print "Standard deviation of scores: %.3f" % std
    for i in range(nmaps):
        if scores[i] < threshold:
            filtered[i] = 'Filtered'
        else:
            filtered[i] = ' '
        basename, ext = os.path.splitext(args.files[i])
        ioutput = basename+"_aligned"
        saxs.write_mrc(aligned[i], sides[0], ioutput+'.mrc')
        print "%s.mrc written. Score = %0.3f %s " % (ioutput,scores[i],filtered[i])
        logging.info('Correlation score to reference: %s.mrc %.3f %s', ioutput, scores[i], filtered[i])

    aligned = aligned[scores>threshold]
    average_rho = np.mean(aligned,axis=0)

    logging.info('Mean of correlation scores: %.3f', mean)
    logging.info('Standard deviation of the scores: %.3f', std)
    logging.info('Total number of input maps for alignment: %i',allrhos.shape[0])
    logging.info('Number of aligned maps accepted: %i', aligned.shape[0])
    logging.info('Correlation score between average and reference: %.3f', 1/saxs.rho_overlap_score(average_rho, refrho))
    saxs.write_mrc(average_rho, sides[0], output+'_avg.mrc')
    logging.info('END')

    """
    #split maps into 2 halves--> enan, align, average independently with same refrho
    avg_rho1 = np.mean(aligned[::2],axis=0)
    avg_rho2 = np.mean(aligned[1::2],axis=0)
    fsc = saxs.calc_fsc(avg_rho1,avg_rho2,sides[0])
    np.savetxt(output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="qbins, FSC")
    """
    #rather than compare two halves, average all fsc's to the reference
    fscs = []
    for map in range(len(aligned)):
        fscs.append(saxs.calc_fsc(aligned[map],refrho,sides[0]))
    fscs = np.array(fscs)
    fsc = np.mean(fscs,axis=0)
    np.savetxt(output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="1/resolution, FSC")
    x = np.linspace(fsc[0,0],fsc[-1,0],100)
    y = np.interp(x, fsc[:,0], fsc[:,1])
    resi = np.argmin(y>=0.5)
    resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
    resn = round(float(1./resx),1)
    print "Resolution: %.1f" % resn, u'\u212B'.encode('utf-8')

    logging.info('Resolution: %.1f '+ u'\u212B'.encode('utf-8'), resn )
    logging.info('END')




