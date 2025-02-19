#!/usr/bin/env python
#
#    denss_all.py
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

from __future__ import print_function
import multiprocessing
import logging
import sys
import argparse
import os, shutil
import copy
import time
from functools import partial

import numpy as np

import denss
from denss import options as dopts


def multi_denss(niter, superargs_dict, args_dict):
    try:
        # time.sleep(1)

        # Processing keyword args for compatibility with RAW GUI
        args_dict['path'] = '.'

        args_dict['output'] = args_dict['output'] + '_' + str(niter)
        np.random.seed(niter + int(time.time()))
        args_dict['seed'] = np.random.randint(2 ** 31 - 1)
        args_dict['quiet'] = True

        if niter <= superargs_dict['nmaps'] - 1:
            sys.stdout.write("\r Running denss job: %i / %i " % (niter + 1, superargs_dict['nmaps']))
            sys.stdout.flush()

        fname = args_dict['output'] + '.log'
        logger = logging.getLogger(args_dict['output'])
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(fname)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        args_dict['my_logger'] = logger

        logger.info('BEGIN')
        logger.info('Script name: %s', sys.argv[0])
        logger.info('DENSS Version: %s', denss.__version__)
        logger.info('Data filename: %s', superargs_dict['file'])
        logger.info('Output prefix: %s', args_dict['output'])
        logger.info('Mode: %s', superargs_dict['mode'])
        result = denss.reconstruct_abinitio_from_scattering_profile(**args_dict)
        logger.info('END')
        return result

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass


def main():
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
    superargs = dopts.parse_arguments(parser)

    #these are arguments specifically for the denss() function
    #it cannot contain keyword arguments that are not listed
    #in the denss() function, so remove any of those here
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
    del args.shrinkwrap_sigma_start_in_A
    del args.shrinkwrap_sigma_end_in_A
    del args.shrinkwrap_sigma_start_in_vox
    del args.shrinkwrap_sigma_end_in_vox

    __spec__ = None

    if superargs.nmaps<2:
        print("Not enough maps to align")
        sys.exit(1)

    fname_nopath = os.path.basename(superargs.file)
    basename, ext = os.path.splitext(fname_nopath)
    if (superargs.output is None) or (superargs.output == basename):
        output = basename
    else:
        output = superargs.output

    out_dir = output
    dirn = 0
    while os.path.isdir(out_dir):
        out_dir = output + "_" + str(dirn)
        dirn += 1

    print(out_dir)
    os.mkdir(out_dir)
    output = out_dir+'/'+output
    args.output = output
    superargs.output = output

    #for convenience and record keeping, make a copy of the input file in the output directory
    shutil.copy(superargs.file, out_dir)

    fname = output+'_final.log'
    superlogger = logging.getLogger(output+'_final')
    superlogger.setLevel(logging.INFO)
    fh = logging.FileHandler(fname)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    superlogger.addHandler(fh)

    superlogger.info('BEGIN')
    superlogger.info('Command: %s', ' '.join(sys.argv))
    superlogger.info('DENSS Version: %s', denss.__version__)
    superlogger.info('Data filename: %s', superargs.file)
    superlogger.info('Enantiomer selection: %r', superargs.enan)

    denss_inputs = {'I':superargs.I,'sigq':superargs.sigq,'q':superargs.q}

    for arg in vars(args):
        denss_inputs[arg]= getattr(args, arg)

    pool = multiprocessing.Pool(superargs.cores)

    superlogger.info('Starting DENSS runs')

    try:
        #mapfunc = partial(multi_denss, **denss_inputs)
        # denss_outputs = pool.map(mapfunc, list(range(superargs.nmaps)))
        mapfunc = partial(multi_denss, superargs_dict=vars(superargs), args_dict=vars(args))
        denss_outputs = pool.map(mapfunc, list(range(superargs.nmaps)))
        print("\r Finishing denss job: %i / %i" % (superargs.nmaps,superargs.nmaps))
        sys.stdout.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.close()
        sys.exit(1)

    superlogger.info('Finished DENSS runs')

    qdata = denss_outputs[0][0]
    Idata = denss_outputs[0][1]
    sigqdata = denss_outputs[0][2]
    qbinsc = denss_outputs[0][3]
    all_Imean = [denss_outputs[i][4] for i in np.arange(superargs.nmaps)]
    all_fits = [denss_outputs[i][-2] for i in np.arange(superargs.nmaps)]
    header = ['q','I','error']
    fit = np.zeros(( all_fits[0].shape[0],superargs.nmaps+3 ))
    fit[:,0] = all_fits[0][:,0]
    fit[:,1] = all_fits[0][:,1]
    fit[:,2] = all_fits[0][:,2]

    for map in range(superargs.nmaps):
        fit[:,map+3] = all_fits[map][:,3]
        header.append("I_fit_"+str(map))

    np.savetxt(output+'_map.fit',fit,delimiter=" ",fmt="%.5e", header=" ".join(header))
    chi_header, rg_header, supportV_header = list(zip(*[('chi_'+str(i), 'rg_'+str(i),'supportV_'+str(i)) for i in range(superargs.nmaps)]))
    all_chis = np.array([denss_outputs[i][5] for i in np.arange(superargs.nmaps)])
    all_rg = np.array([denss_outputs[i][6] for i in np.arange(superargs.nmaps)])
    all_supportV = np.array([denss_outputs[i][7] for i in np.arange(superargs.nmaps)])
    final_chis = np.zeros(superargs.nmaps)
    final_rgs = np.zeros(superargs.nmaps)
    final_supportVs = np.zeros(superargs.nmaps)
    for i in range(superargs.nmaps):
        final_rgs[i] = all_rg[i,all_rg[i]>0][-1]
        final_chis[i] = all_chis[i,all_chis[i]>0][-1]
        final_supportVs[i] = all_supportV[i,all_supportV[i]>0][-1]
    superlogger.info('Average Rg...............: %3.3f +- %3.3f', np.mean(final_rgs), np.std(final_rgs))
    superlogger.info('Average Chi2.............: %.3e +- %.3e', np.mean(final_chis), np.std(final_chis))
    superlogger.info('Average Support Volume...: %3.3f +- %3.3f', np.mean(final_supportVs), np.std(final_supportVs))

    np.savetxt(output+'_chis_by_step.fit',all_chis.T,delimiter=" ",fmt="%.5e",header=",".join(chi_header))
    np.savetxt(output+'_rg_by_step.fit',all_rg.T,delimiter=" ",fmt="%.5e",header=",".join(rg_header))
    np.savetxt(output+'_supportV_by_step.fit',all_supportV.T,delimiter=" ",fmt="%.5e",header=",".join(supportV_header))

    allrhos = np.array([denss_outputs[i][8] for i in np.arange(superargs.nmaps)])
    sides = np.array([denss_outputs[i][9] for i in np.arange(superargs.nmaps)])

    if superargs.ref is not None:
        #allow input of reference structure
        if superargs.ref.endswith('.pdb'):
            reffname_nopath = os.path.basename(superargs.ref)
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
            pdb = denss.PDB(superargs.ref)
            if superargs.center:
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
        if superargs.ref.endswith('.mrc'):
            refrho, refside = denss.read_mrc(superargs.ref)

    if superargs.enan:
        print()
        print(" Selecting best enantiomers...")
        superlogger.info('Selecting best enantiomers')
        try:
            allrhos, scores = denss.select_best_enantiomers(allrhos, cores=superargs.cores)
        except KeyboardInterrupt:
            sys.exit(1)
        for i in range(superargs.nmaps):
            ioutput = output+"_"+str(i)+"_enan"
            denss.write_mrc(allrhos[i], sides[0], ioutput+".mrc")

    if superargs.ref is None:
        print()
        print(" Generating reference...")
        superlogger.info('Generating reference')
        try:
            refrho = denss.binary_average(allrhos, superargs.cores)
            denss.write_mrc(refrho, sides[0], output+"_reference.mrc")
        except KeyboardInterrupt:
            sys.exit(1)

    print()
    print(" Aligning all maps to reference...")
    superlogger.info('Aligning all maps to reference')
    try:
        aligned, scores = denss.align_multiple(refrho, allrhos, superargs.cores)
    except KeyboardInterrupt:
        sys.exit(1)

    #filter rhos with scores below the mean - 2*standard deviation.
    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean - 2*std
    filtered = np.empty(len(scores),dtype=str)
    print("Mean of correlation scores: %.3f" % mean)
    print("Standard deviation of scores: %.3f" % std)
    for i in range(superargs.nmaps):
        if scores[i] < threshold:
            filtered[i] = 'Filtered'
        else:
            filtered[i] = ' '
        ioutput = output+"_"+str(i)+"_aligned"
        denss.write_mrc(aligned[i], sides[0], ioutput+".mrc")
        print("%s.mrc written. Score = %0.3f %s " % (ioutput,scores[i],filtered[i]))
        superlogger.info('Correlation score to reference: %s.mrc %.3f %s', ioutput, scores[i], filtered[i])

    idx_keep = np.where(scores>threshold)
    kept_ids = np.arange(superargs.nmaps)[idx_keep]
    aligned = aligned[idx_keep]
    average_rho = np.mean(aligned,axis=0)

    superlogger.info('Mean of correlation scores: %.3f', mean)
    superlogger.info('Standard deviation of the scores: %.3f', std)
    superlogger.info('Total number of input maps for alignment: %i',allrhos.shape[0])
    superlogger.info('Number of aligned maps accepted: %i', aligned.shape[0])
    superlogger.info('Correlation score between average and reference: %.3f', -denss.rho_overlap_score(average_rho, refrho))
    superlogger.info('Mean Density of Avg Map (all voxels): %3.5f', np.mean(average_rho))
    superlogger.info('Std. Dev. of Density (all voxels): %3.5f', np.std(average_rho))
    superlogger.info('RMSD of Density (all voxels): %3.5f', np.sqrt(np.mean(np.square(average_rho))))
    denss.write_mrc(average_rho, sides[0], output+'_avg.mrc')

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

    superlogger.info('Resolution = %.1f +- %.1f A' % (resn,resn_sd))
    superlogger.info('END')

    if superargs.plot:
        import matplotlib.pyplot as plt
        plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')
        for i in range(len(aligned)):
            plt.plot(fscs[i,:,0],fscs[i,:,1],'k--',alpha=0.1)
        plt.plot(fsc[:,0],fsc[:,1],'bo-')
        #plt.plot(x,y,'k-')
        plt.plot([resx],[0.5],'ro',label=r'Resolution = %.2f $\mathrm{\AA}$'%resn)
        plt.legend()
        plt.xlabel(r'Resolution (1/$\mathrm{\AA}$)')
        plt.ylabel('Fourier Shell Correlation')
        pltoutput = os.path.splitext(output)[0]
        print(pltoutput)
        plt.savefig(pltoutput+'_fsc.png',dpi=150)
        plt.close()


if __name__ == "__main__":
    print("Entering main() function of denss_all.py")
    main()