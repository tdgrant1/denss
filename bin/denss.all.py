#!/usr/bin/env python
#
#    denss.all.py
#    Generate, align, and average many electron density maps from solution scattering data.
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


import multiprocessing
import logging
import sys
import argparse
import os
import copy
import time
from functools import partial

import numpy as np

import saxstats.denssopts as dopts
from saxstats._version import __version__
import saxstats.saxstats as saxs

#have to run parser twice, first just to get filename for loadProfile
#then have to run it after deciding what the correct dmax should be
#so that the voxel size, box size, nsamples, etc are set correctly
initparser = argparse.ArgumentParser(description="Generate, align, and average many electron density maps from solution scattering data.", formatter_class=argparse.RawTextHelpFormatter)
initparser.add_argument("-nm", "--nmaps",default = 20,type =int, help="Number of maps to be generated (default 20)")
initparser.add_argument("-j", "--cores", type=int, default = 1, help="Number of cores used for parallel processing. (default: 1)")
initparser.add_argument("-en_on", "--enantiomer_on", action = "store_true", dest="enan", help="Generate and select best enantiomers (default). ")
initparser.add_argument("-en_off", "--enantiomer_off", action = "store_false", dest="enan", help="Do not generate and select best enantiomers.")
initparser.add_argument("-ref", "--ref", default=None, type=str, help="Input reference model (.mrc or .pdb file, optional).")
initparser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center reference PDB map.")
initparser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center reference PDB map (default).")
initparser.add_argument("-r", "--resolution", default=15.0, type=float, help="Resolution of map calculated from reference PDB file (default 15 angstroms).")
initparser.set_defaults(enan = True)
initparser.set_defaults(center = True)
initargs = dopts.parse_arguments(initparser, gnomdmax=None)

q, I, sigq, dmax, isout = saxs.loadProfile(initargs.file, units=initargs.units)

if not initargs.force_run:
    if min(q) != 0.0:
        print "CAUTION: Minimum q value = %f " % min(q)
        print "is not 0.0. It is STRONGLY recommended to include"
        print "I(q=0) in your given scattering profile. You can use"
        print "denss.fit_data.py to calculate a scattering profile fit"
        print "which will include I(q=0), or you can also use the GNOM"
        print "program from ATSAS to create a .out file."
        print
        print "If you are positive you would like to continue, "
        print "rerun with the --force_run option."
        sys.exit()

if dmax <= 0:
    dmax = None

parser = argparse.ArgumentParser()
parser.add_argument("-nm", "--nmaps",default = 20,type =int, help="Number of maps to be generated (default 20)")
parser.add_argument("-j", "--cores", type=int, default = 1, help="Number of cores used for parallel processing. (default: 1)")
parser.add_argument("-en_on", "--enantiomer_on", action = "store_true", dest="enan", help="Generate and select best enantiomers (default). ")
parser.add_argument("-en_off", "--enantiomer_off", action = "store_false", dest="enan", help="Do not generate and select best enantiomers.")
parser.add_argument("-ref", "--ref", default=None, type=str, help="Input reference model (.mrc or .pdb file, optional).")
parser.add_argument("-c_on", "--center_on", dest="center", action="store_true", help="Center reference PDB map.")
parser.add_argument("-c_off", "--center_off", dest="center", action="store_false", help="Do not center reference PDB map (default).")
parser.add_argument("-r", "--resolution", default=15.0, type=float, help="Resolution of map calculated from reference PDB file (default 15 angstroms).")
parser.set_defaults(enan = True)
parser.set_defaults(center = True)
superargs = dopts.parse_arguments(parser, gnomdmax=dmax)

args = copy.copy(superargs)
del args.units
del args.cores
del args.enan
del args.ref
del args.nmaps
del args.file
del args.plot
del args.nsamples
del args.mode
del args.resolution
del args.center
del args.force_run

def multi_denss(niter, **kwargs):
    try:
        # Processing keyword args for compatibility with RAW GUI
        kwargs['path'] = '.'

        kwargs['output'] = kwargs['output'] +'_'+str(niter)
        np.random.seed(niter+int(time.time()))
        kwargs['seed'] = np.random.randint(2**31-1)
        kwargs['quiet'] = True

        if niter<=superargs.nmaps-1:
            sys.stdout.write( "\r Running denss job: %i / %i " % (niter+1,superargs.nmaps))
            sys.stdout.flush()

        fname = kwargs['output']+'.log'
        logger = logging.getLogger("")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(fname)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        kwargs['my_logger'] = logger

        logging.info('BEGIN')
        logging.info('Script name: %s', sys.argv[0])
        logging.info('DENSS Version: %s', __version__)
        logging.info('Data filename: %s', superargs.file)
        logging.info('Output prefix: %s', kwargs['output'])
        logging.info('Mode: %s', superargs.mode)
        result = saxs.denss(**kwargs)
        logging.info('END')
        return result
        time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":

    if superargs.nmaps<2:
        print "Not enough maps to align"
        sys.exit(1)

    basename, ext = os.path.splitext(superargs.file)
    if (superargs.output is None) or (superargs.output == basename):
        output = basename
    else:
        output = superargs.output

    out_dir = output
    dirn = 0
    while os.path.isdir(out_dir):
        out_dir = output + "_" + str(dirn)
        dirn += 1

    print out_dir
    os.mkdir(out_dir)
    output = out_dir+'/'+out_dir
    args.output = output
    superargs.output = output

    fname = output+'_final.log'
    superlogger = logging.getLogger("")
    superlogger.setLevel(logging.INFO)
    fh = logging.FileHandler(fname)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    superlogger.addHandler(fh)

    logging.info('BEGIN')
    logging.info('Script name: %s', sys.argv[0])
    logging.info('DENSS Version: %s', __version__)
    logging.info('Data filename: %s', superargs.file)
    logging.info('Enantiomer selection: %r', superargs.enan)

    denss_inputs = {'I':I,'sigq':sigq,'q':q}

    for arg in vars(args):
        denss_inputs[arg]= getattr(args, arg)

    pool = multiprocessing.Pool(superargs.cores)

    try:
        mapfunc = partial(multi_denss, **denss_inputs)
        denss_outputs = pool.map(mapfunc, range(superargs.nmaps))
        print "\r Finishing denss job: %i / %i" % (superargs.nmaps,superargs.nmaps)
        sys.stdout.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.close()
        sys.exit(1)

    qdata = denss_outputs[0][0]
    Idata = denss_outputs[0][1]
    sigqdata = denss_outputs[0][2]
    qbinsc = denss_outputs[0][3]
    all_Imean = [denss_outputs[i][4] for i in np.arange(superargs.nmaps)]
    header = ['q','I','error']
    fit = np.zeros(( len(qbinsc),superargs.nmaps+3 ))
    fit[:len(qdata),0] = qdata
    fit[:len(Idata),1] = Idata
    fit[:len(sigqdata),2] = sigqdata

    for map in range(superargs.nmaps):
        fit[:len(all_Imean[0]),map+3] = all_Imean[map]
        header.append("I_fit_"+str(map))

    np.savetxt(output+'_map.fit',fit,delimiter=" ",fmt="%.5e", header=" ".join(header))
    chi_header, rg_header, supportV_header = zip(*[('chi_'+str(i), 'rg_'+str(i),'supportV_'+str(i)) for i in range(superargs.nmaps)])
    all_chis = np.array([denss_outputs[i][5] for i in np.arange(superargs.nmaps)])
    all_rg = np.array([denss_outputs[i][6] for i in np.arange(superargs.nmaps)])
    all_supportV = np.array([denss_outputs[i][7] for i in np.arange(superargs.nmaps)])

    np.savetxt(output+'_chis_by_step.fit',all_chis.T,delimiter=" ",fmt="%.5e",header=",".join(chi_header))
    np.savetxt(output+'_rg_by_step.fit',all_rg.T,delimiter=" ",fmt="%.5e",header=",".join(rg_header))
    np.savetxt(output+'_supportV_by_step.fit',all_supportV.T,delimiter=" ",fmt="%.5e",header=",".join(supportV_header))

    allrhos = np.array([denss_outputs[i][8] for i in np.arange(superargs.nmaps)])
    sides = np.array([denss_outputs[i][9] for i in np.arange(superargs.nmaps)])

    if superargs.ref is not None:
        #allow input of reference structure
        if superargs.ref.endswith('.pdb'):
            refside = sides[0]
            voxel = (refside/allrhos[0].shape)[0]
            halfside = refside/2
            n = int(refside/voxel)
            dx = refside/n
            x_ = np.linspace(-halfside,halfside,n)
            x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
            xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
            pdb = saxs.PDB(superargs.ref)
            if superargs.center:
                pdb.coords -= pdb.coords.mean(axis=0)
            refrho = saxs.pdb2map_gauss(pdb,xyz=xyz,sigma=superargs.resolution)
            refrho = refrho*np.sum(allrhos[0])/np.sum(refrho)
        if superargs.ref.endswith('.mrc'):
            refrho, refside = saxs.read_mrc(superargs.ref)

    if superargs.enan:
        print
        print " Selecting best enantiomers..."
        try:
            allrhos, scores = saxs.select_best_enantiomers(allrhos, cores=superargs.cores)
        except KeyboardInterrupt:
            sys.exit(1)

    if superargs.ref is None:
        print
        print " Generating reference..."
        try:
            refrho = saxs.binary_average(allrhos, superargs.cores)
        except KeyboardInterrupt:
            sys.exit(1)

    print
    print " Aligning all maps to reference..."
    try:
        aligned, scores = saxs.align_multiple(refrho, allrhos, superargs.cores)
    except KeyboardInterrupt:
        sys.exit(1)

    #filter rhos with scores below the mean - 2*standard deviation.
    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean - 2*std
    filtered = np.empty(len(scores),dtype=str)
    print "Mean of correlation scores: %.3f" % mean
    print "Standard deviation of scores: %.3f" % std
    for i in range(superargs.nmaps):
        if scores[i] < threshold:
            filtered[i] = 'Filtered'
        else:
            filtered[i] = ' '
        ioutput = output+"_"+str(i)+"_aligned"
        saxs.write_mrc(aligned[i], sides[0], ioutput+".mrc")
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

    """
    #split maps into 2 halves--> enan, align, average independently with same refrho
    avg_rho1 = np.mean(aligned[::2],axis=0)
    avg_rho2 = np.mean(aligned[1::2],axis=0)
    fsc = saxs.calc_fsc(avg_rho1,avg_rho2,sides[0])
    np.savetxt(output+'_fsc.dat',fsc,delimiter=" ",fmt="%.5e",header="qbins, FSC")
    """
    #rather than compare two halves, average all fsc's to the reference
    fscs = []
    for calc_map in range(len(aligned)):
        fscs.append(saxs.calc_fsc(aligned[calc_map],refrho,sides[0]))
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
