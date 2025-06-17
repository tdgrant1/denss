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
import sys, os, argparse, logging
import numpy as np

import denss

def main():
    parser = argparse.ArgumentParser(description="A tool for aligning and averaging multiple electron density maps.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version", action="version",version="%(prog)s v{version}".format(version=denss.__version__))
    parser.add_argument("-f", "--files", type=str, nargs="+", help="List of MRC files")
    parser.add_argument("-ref", "--ref",default = None, type=str, help="Reference filename (.mrc or .pdb file, optional)")
    parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center PDB (default).")
    parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center PDB.")
    parser.add_argument("-en_on", "--enantiomer_on", action = "store_true", dest="enan", help="Generate and select best enantiomers (default). ")
    parser.add_argument("-en_off", "--enantiomer_off", action = "store_false", dest="enan", help="Do not generate and select best enantiomers.")
    parser.add_argument("-r", "--resolution", default=15.0, type=float, help="Desired resolution (i.e. Gaussian width sigma) of map calculated from PDB file.")
    parser.add_argument("--ignore_pdb_waters", dest="ignore_waters", action="store_true", help="Ignore waters if PDB file given.")
    parser.add_argument("-j", "--cores", type=int, default = 1, help="Number of cores used for parallel processing. (default: 1)")
    parser.add_argument("-o", "--output", type=str, help="output filename prefix")
    parser.set_defaults(enan = True)
    parser.set_defaults(center = True)
    parser.set_defaults(ignore_waters = False)
    args = parser.parse_args()

    __spec__ = None

    if args.output is None:
        fname_nopath = os.path.basename(args.files[0])
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    logging.basicConfig(filename=output+'_final.log',level=logging.INFO,filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('BEGIN')
    logging.info('Command: %s', ' '.join(sys.argv))
    #logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', denss.__version__)
    logging.info('Map filename(s): %s', args.files)
    logging.info('Reference filename: %s', args.ref)
    logging.info('Enantiomer selection: %s', args.enan)

    nmaps = len(args.files)

    allrhos = []
    sides = []
    for file in args.files:
        rho, side = denss.read_mrc(file)
        allrhos.append(rho)
        sides.append(side)
    allrhos = np.array(allrhos)
    sides = np.array(sides)

    if nmaps<2:
        print("Not enough maps to align. Please input more maps again...")
        sys.exit(1)

    if args.ref is not None:
        #allow input of reference structure
        if args.ref.endswith('.pdb'):
            logging.info('Center PDB reference: %s', args.center)
            logging.info('PDB reference map resolution: %.2f', args.resolution)
            reffname_nopath = os.path.basename(args.ref)
            refbasename, refext = os.path.splitext(reffname_nopath)
            refoutput = refbasename+"_centered.pdb"
            refside = sides[0]
            voxel = (refside/allrhos[0].shape)[0]
            halfside = refside/2
            n = int(refside/voxel)
            dx = refside/n
            x_ = np.linspace(-halfside,halfside,n)
            x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
            xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
            pdb = denss.PDB(args.ref)
            if args.center:
                pdb.coords -= pdb.coords.mean(axis=0)
                pdb.write(filename=refoutput)
            pdb2mrc = denss.PDB2MRC(
                pdb=pdb,
                center_coords=False, #done above
                voxel=dx,
                side=refside,
                nsamples=n,
                ignore_warnings=True,
                )
            pdb2mrc.scale_radii()
            pdb2mrc.make_grids()
            pdb2mrc.calculate_global_B()
            pdb2mrc.calculate_invacuo_density()
            pdb2mrc.calculate_excluded_volume()
            pdb2mrc.calculate_hydration_shell()
            pdb2mrc.calculate_structure_factors()
            pdb2mrc.calc_rho_with_modified_params(pdb2mrc.params)
            refrho = pdb2mrc.rho_insolvent
            refrho = refrho*np.sum(allrhos[0])/np.sum(refrho)
            denss.write_mrc(refrho,pdb2mrc.side,filename=refbasename+'_pdb.mrc')
        if args.ref.endswith('.mrc'):
            refrho, refside = denss.read_mrc(args.ref)
        if (not args.ref.endswith('.mrc')) and (not args.ref.endswith('.pdb')):
            print("Invalid reference filename given. .mrc or .pdb file required")
            sys.exit(1)

    if args.enan:
        print(" Selecting best enantiomers...")
        try:
            if args.ref:
                allrhos, scores = denss.select_best_enantiomers(allrhos, refrho=refrho, cores=args.cores)
            else:
                allrhos, scores = denss.select_best_enantiomers(allrhos, refrho=allrhos[0], cores=args.cores)
        except KeyboardInterrupt:
            sys.exit(1)

    if args.ref is None:
        print(" Generating reference...")
        try:
            refrho = denss.binary_average(allrhos, args.cores)
            denss.write_mrc(refrho, sides[0], output+"_reference.mrc")
        except KeyboardInterrupt:
            sys.exit(1)

    print(" Aligning all maps to reference...")
    try:
        aligned, scores = denss.align_multiple(refrho, allrhos, args.cores)
    except KeyboardInterrupt:
        sys.exit(1)

    #filter rhos with scores below the mean - 2*standard deviation.
    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean - 2*std
    filtered = np.empty(len(scores),dtype=str)
    print()
    print("Mean of correlation scores: %.3e" % mean)
    print("Standard deviation of scores: %.3e" % std)
    for i in range(nmaps):
        if scores[i] < threshold:
            filtered[i] = 'Filtered'
        else:
            filtered[i] = ' '
        fname_nopath = os.path.basename(args.files[i])
        basename, ext = os.path.splitext(fname_nopath)
        ioutput = basename+"_aligned"
        denss.write_mrc(aligned[i], sides[0], ioutput+'.mrc')
        print("%s.mrc written. Score = %0.3e %s " % (ioutput,scores[i],filtered[i]))
        logging.info('Correlation score to reference: %s.mrc %.3e %s', ioutput, scores[i], filtered[i])

    idx_keep = np.where(scores>threshold)
    kept_ids = np.arange(nmaps)[idx_keep]
    aligned = aligned[idx_keep]
    average_rho = np.mean(aligned,axis=0)

    logging.info('Mean of correlation scores: %.3e', mean)
    logging.info('Standard deviation of the scores: %.3e', std)
    logging.info('Total number of input maps for alignment: %i',allrhos.shape[0])
    logging.info('Number of aligned maps accepted: %i', aligned.shape[0])
    logging.info('Correlation score between average and reference: %.3e', -denss.rho_overlap_score(average_rho, refrho))
    denss.write_mrc(average_rho, sides[0], output+'_avg.mrc')
    logging.info('END')


    #rather than compare two halves, average all fsc's to the reference
    fscs = []
    resns = []
    for calc_map in range(len(aligned)):
        fsc_map = denss.calc_fsc(aligned[calc_map],refrho,sides[0])
        fscs.append(fsc_map)
        resn_map = denss.fsc2res(fsc_map)
        resns.append(resn_map)

    fscs = np.array(fscs)

    #save a file containing all fsc curves
    fscs_header = ['res(1/A)']
    for i in kept_ids:
        ioutput = output+"_"+str(i)+"_aligned"
        fscs_header.append(ioutput)
    #add the resolution as the first column
    fscs_for_file = np.vstack((fscs[0,:,0],fscs[:,:,1])).T
    np.savetxt(output+'_allfscs.dat',fscs_for_file,delimiter=" ",fmt="%.5e",header=",".join(fscs_header))

    resns = np.array(resns)
    fsc = np.mean(fscs,axis=0)
    resn, x, y, resx = denss.fsc2res(fsc, return_plot=True)
    resn_sd = np.std(resns)
    if np.min(fsc[:,1]) > 0.5:
        print("Resolution: < %.1f +- %.1f A (maximum possible)" % (resn,resn_sd))
    else:
        print("Resolution: %.1f +- %.1f A " % (resn,resn_sd))

    np.savetxt(output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="1/resolution, FSC; Resolution=%.1f +- %.1f A" % (resn,resn_sd))

    logging.info('Resolution: %.1f A', resn )
    logging.info('END')


if __name__ == "__main__":
    main()

