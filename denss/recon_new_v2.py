import sys
import re
import os
import json
import struct
import logging
from functools import partial
import multiprocessing
import datetime, time
from time import sleep
import warnings
import pickle

import numpy as np
from scipy import ndimage, interpolate, spatial, special, optimize, signal, stats, fft
from functools import reduce

# load some dictionaries
from denss.resources import resources




# PYFFTW = False

from .core import mybinmean, myrfftn, abs2, mysqrt, mysum, myirfftn, write_mrc
from .core import shrinkwrap_by_density_value, PYFFTW, shrinkwrap_by_volume, mystd, mymean, grid_center

def reconstruct_abinitio_from_scattering_profile_PA(q, I, sigq, dmax, qraw=None, Iraw=None, sigqraw=None,
                                                 ne=None, voxel=5., oversampling=3., recenter=True, recenter_steps=None,
                                                 recenter_mode="com", positivity=True, positivity_steps=None, extrapolate=True, output="map",
                                                 steps=None, seed=None, rho_start=None, support_start=None, add_noise=None,
                                                 shrinkwrap=True, shrinkwrap_old_method=False, shrinkwrap_sigma_start=3,
                                                 shrinkwrap_sigma_end=1.5, shrinkwrap_sigma_decay=0.99, shrinkwrap_threshold_fraction=0.2,
                                                 shrinkwrap_iter=20, shrinkwrap_minstep=100, chi_end_fraction=0.01,
                                                 write_xplor_format=False, write_freq=100, enforce_connectivity=True,
                                                 enforce_connectivity_steps=[500], enforce_connectivity_max_features=1, cutout=True, quiet=False, ncs=0,
                                                 ncs_steps=[500], ncs_axis=1, ncs_type="cyclical", abort_event=None, my_logger=logging.getLogger(),
                                                 path='.', gui=False, DENSS_GPU=False,
                                                 PA_outdir=None, PA_cont=True, PA_dparams=[1], PA_files=None, PA_initmrc=None):
    """Calculate electron density from scattering data."""
    print('')
    print('!!')
    print('Using V2 PA DENSS recon')
    print('!!')

    ### PART 1
    ### Load in the multiple sas profiles 

    ### NEW
    PA_qs = []
    PA_Is = []
    PA_sigqs = []
    for PA_fname in PA_files:
        PA_loaded_array = np.loadtxt(PA_fname)
        PA_qs.append(PA_loaded_array[:,0])
        PA_Is.append(PA_loaded_array[:,1])
        PA_sigqs.append(PA_loaded_array[:,2])


    ### PART 2
    ### abort if needed and run with cuda if set. Also set the file path.

    if abort_event is not None:
        if abort_event.is_set():
            my_logger.info('Aborted!')
            return []

    if DENSS_GPU and CUPY_LOADED:
        DENSS_GPU = True
    elif DENSS_GPU:
        if gui:
            my_logger.info("GPU option set, but CuPy failed to load")
        else:
            print("GPU option set, but CuPy failed to load")
        DENSS_GPU = False

    fprefix = os.path.join(path, output)



    ### PART 3
    ### set up the varibles ranges. How many qbins? over what range? rbins?

    D = dmax

    side = oversampling * D
    halfside = side / 2

    n = int(side / voxel)
    # want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n % 2 == 1:
        n += 1
    # store n for later use if needed
    nbox = n

    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = np.linspace(-(n // 2) * dx, (n // 2 - 1) * dx, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    df = 1 / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qz_ = np.fft.rfftfreq(x_.size) * n * df * 2 * np.pi
    qx, qy, qz = np.meshgrid(qx_, qx_, qz_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0]) - 1e-8  # subtract a tiny bit to deal with floating point error
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)

    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    qbl = qbin_labels
    qblravel = qbin_labels.ravel()
    xcount = np.bincount(qblravel)

    # calculate qbinsc as average of q values in shell
    qbinsc = mybinmean(qr.ravel(), qblravel, xcount)

    # allow for any range of q data
    qdata = qbinsc[np.where((qbinsc >= q.min()) & (qbinsc <= q.max()))]

    ### PART 4
    ### In the original function, the intensity values are interpolated over the
    ### qbined values, the original q and I varibles aren't used.
    ### for each intensity value you read, interpolate it over qdata.


    ### NEW 
    PA_Idatas = []
    for PA_q, PA_I in zip(PA_qs, PA_Is):
        PA_Idatas.append(np.interp(qdata, PA_q, PA_I ))


    ### PART 5
    ### For some reason, the qdata we calculate might be bigger then the data we supplied.
    ### in this case, we want to extend the intensity function into this q range.

    if extrapolate:
        qextend = qbinsc[qbinsc >= qdata.max()]
        Iextend = qextend ** -4
        Iextend = Iextend / Iextend[0] * Idata[-1]
        qdata = np.concatenate((qdata, qextend[1:]))
        Idata = np.concatenate((Idata, Iextend[1:]))

    # create list of qbin indices just in region of data for later F scaling
    qbin_args = np.in1d(qbinsc, qdata, assume_unique=True)
    qba = qbin_args  # just for brevity when using it later

    sigqdata = np.interp(qdata, q, sigq)


    ### PART 5
    ### I don't know what this scale factor is doing but I will comment it out

    ### OLD
    ### scale_factor = ne ** 2 / Idata[0]

    #### PART 5
    #### Work out how many step are in the recon

    if steps == 'None' or steps is None or int(steps) < 1:
        stepsarr = np.concatenate((enforce_connectivity_steps, [shrinkwrap_minstep]))
        maxec = np.max(stepsarr)
        steps = int(shrinkwrap_iter * (
                    np.log(shrinkwrap_sigma_end / shrinkwrap_sigma_start) / np.log(shrinkwrap_sigma_decay)) + maxec)
        # add enough steps for convergence after shrinkwrap is finished
        # something like 7000 seems reasonable, likely will finish before that on its own
        # then just make a round number when using defaults
        steps += 7621
    else:
        steps = int(steps)

    ### PART 6
    ### We need to copy Imean for each d param
    ### raw variables are None by default, so just assign the variables without None check in original code


    ### NEW
    PA_Imeans = []
    for _ in PA_Is:
        PA_Imeans.append(np.zeros(len(qbins)))

    qraw = q
    PA_Iraws = PA_Is
    PA_sigqraws = PA_sigqs


    ### PART 7
    ### chi is calculated for each sas profile,
    ### rg and support is calculated for each real model (only one of those)


    ### NEW
    PA_chi = np.zeros( (len(PA_Is), steps + 1))



    rg = np.zeros((steps + 1), dtype=np.complex128)
    supportV = np.zeros((steps + 1))
    if support_start is not None:
        support = np.copy(support_start)
    else:
        support = np.ones(x.shape, dtype=bool)

    ### PART 7
    ### generate random seed 
    if seed is None:
        # Have to reset the random seed to get a random in different from other processes
        prng = np.random.RandomState()
        seed = prng.randint(2 ** 31 - 1)
    else:
        seed = int(seed)

    prng = np.random.RandomState(seed)


    ### PART 8
    ### load in (or generate) initial density


    ### NEW ####TODO
    if rho_start is not None:
        print('Loading a new rho start!!')
        rho = rho_start
        if add_noise is not None:
            noise_factor = rho.max() * add_noise
            noise = prng.random_sample(size=x.shape) * noise_factor
            rho += noise
    else:
        rho = prng.random_sample(size=x.shape)
    newrho = np.zeros_like(rho)





    ### PART 9
    ### Calculate a bunch of stuff that I don't know

    sigma = shrinkwrap_sigma_start

    # calculate the starting shrinkwrap volume as the volume of a sphere
    # of radius Dmax, i.e. much larger than the particle size
    swbyvol = True
    swV = V / 2.0
    Vsphere_Dover2 = 4. / 3 * np.pi * (D / 2.) ** 3
    swVend = Vsphere_Dover2
    swV_decay = 0.9
    first_time_swdensity = True
    threshold = shrinkwrap_threshold_fraction
    # erode will make take five outer edge pixels of the support, like a shell,
    # and will make sure no negative density is in that region
    # this is to counter an artifact that occurs when allowing for negative density
    # as the negative density often appears immediately next to positive density
    # at the edges of the object. This ensures (i.e. biases) only real negative density
    # in the interior of the object (i.e. more than five pixels from the support boundary)
    # thus we only need this on when in membrane mode, i.e. when positivity=False
    if shrinkwrap_old_method or positivity:
        erode = False
    else:
        erode = True
        erosion_width = int(20 / dx)  # this is in pixels
        if erosion_width == 0:
            # make minimum of one pixel
            erosion_width = 1

    # for icosahedral symmetry, just set ncs to 1 to trigger ncs averaging
    if ncs_type == "icosahedral":
        ncs = 1

    ### PART 10
    ### bunch of logging stuff was here that I deleted

    my_logger.info('blah blah')


    ### PART 11
    ### removing PYFFTW  DENSS_GPU stuff, by default it is false

    # if PYFFTW:
        # ...
    # if DENSS_GPU:
        #  ...




    ### PART 13
    ### The actual iteration loop

    for j in range(steps):
        if abort_event is not None:
            if abort_event.is_set():
                my_logger.info('Aborted!')
                return []

        ### PART 14
        ### for each sas curve, do the modulus constraint


        ### NEW
        ### make an array to hold the reconstructed densities for each d
        PA_rhods = np.zeros( (len(PA_dparams), rho.shape[0], rho.shape[1], rho.shape[2]))

        for PA_i_dparam, PA_dparam in enumerate(PA_dparams):


            ### for this d param, get rho to the power of d (handling the negatives)
            PA_neg_loc = np.where(rho<0)
            PA_rhod = np.abs(rho)**PA_dparam
            PA_rhod[PA_neq_loc] *= -1


            F = myrfftn(PA_rhod, DENSS_GPU=DENSS_GPU)

            # sometimes, when using denss_refine.py with non-random starting rho,
            # the resulting Fs result in zeros in some locations and the algorithm to break
            # here just make those values to be 1e-16 to be non-zero
            F[np.abs(F) == 0] = 1e-16

            # APPLY RECIPROCAL SPACE RESTRAINTS
            # calculate spherical average of intensities from 3D Fs
            # I3D = myabs(F, DENSS_GPU=DENSS_GPU)**2
            I3D = abs2(F)
            Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount, DENSS_GPU=DENSS_GPU)

            # scale Fs to match data
            factors = mysqrt(Idata / Imean, DENSS_GPU=DENSS_GPU)
            # do not scale bins outside of desired range
            # so set those factors to 1.0
            factors[~qba] = 1.0
            F *= factors[qbin_labels]


            ### There was a try: except: statement here. I have just left the except part.
            PA_chi[PA_i_dparam][j] = mysum(((Imean[qba] - PA_Idatas[PA_i_dparam][qba]) / PA_sigqdata[PA_i_dparam][qba]) ** 2, DENSS_GPU=DENSS_GPU) / PA_Idatas[PA_i_dparam][qba].size


            # APPLY REAL SPACE RESTRAINTS
            PA_rhods[PA_i_dparam] = myirfftn(F, DENSS_GPU=DENSS_GPU).real


            ### invert the powering we did at the start of the loop
            PA_neg_loc = np.where(PA_rhods[PA_i_dparam]<0)
            PA_rhods[PA_i_dparam] = np.abs(PA_rhods[PA_i_dparam])**(1/PA_dparam)
            PA_rhods[PA_i_dparam][PA_neg_loc] *= -1



        ### average back the unscaled rhos back together
        rhoprime = np.mean(PA_rhods, axis=0)
        


        ### PART 15
        ### Now that we have remade a version of rhoprime,
        ### everything down the line should hold

        ### you are here!!



        # use Guinier's law to approximate quickly
        rg_j = calc_rg_by_guinier_first_2_points(qbinsc, Imean, DENSS_GPU=DENSS_GPU)

        rg[j] = rg_j

        # Error Reduction
        newrho *= 0
        newrho[support] = rhoprime[support]

        if not DENSS_GPU and j % write_freq == 0:
            if write_xplor_format:
                write_xplor(rhoprime / dV, side, fprefix + "_current.xplor")
            write_mrc(rhoprime / dV, side, fprefix + "_current.mrc")

        # enforce positivity by making all negative density points zero.
        if positivity:  # and j in positivity_steps:
            newrho[newrho < 0] = 0.0

        # apply non-crystallographic symmetry averaging
        if ncs != 0 and j in ncs_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            if ncs_type == "icosahedral":
                newrho, shift = center_rho_roll(newrho, recenter_mode="com", maxfirst=True, return_shift=True)
                support = np.roll(np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
            else:
                newrho = align2xyz(newrho)
            if DENSS_GPU:
                newrho = cp.array(newrho)

        if ncs != 0 and j in [stepi + 1 for stepi in ncs_steps]:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
            if ncs_type == "icosahedral":
                rotations = get_icosahedral_matrices()
                newrho_total = np.copy(newrho)
                for R in rotations[1:]:  # Skip identity (first matrix)
                    sym = transform_rho(newrho, R=R, mode='constant')
                    newrho_total += sym
                newrho = newrho_total / len(rotations)
            else:
                if ncs_axis == 1:
                    axes = (1, 2)  # longest
                if ncs_axis == 2:
                    axes = (0, 2)  # middle
                if ncs_axis == 3:
                    axes = (0, 1)  # shortest
                degrees = 360. / ncs
                newrho_total = np.copy(newrho)
                if ncs_type == "dihedral":
                    # first, rotate original about perpendicular axis by 180
                    # then apply n-fold cyclical rotation
                    d2fold = ndimage.rotate(newrho, 180, axes=axes, reshape=False)
                    newrhosym = np.copy(newrho) + d2fold
                    newrhosym /= 2.0
                    newrho_total = np.copy(newrhosym)
                else:
                    newrhosym = np.copy(newrho)
                for nrot in range(1, ncs):
                    sym = ndimage.rotate(newrhosym, degrees * nrot, axes=axes, reshape=False)
                    newrho_total += np.copy(sym)
                newrho = newrho_total / ncs

            # run shrinkwrap after ncs averaging to get new support
            if shrinkwrap_old_method:
                # run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                swN = int(swV / dV)
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" % threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" % threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            if DENSS_GPU:
                newrho = cp.array(newrho)

        if recenter and j in recenter_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            # cannot run center_rho_roll() function since we want to also recenter the support
            # perhaps we should fix this in the future to clean it up
            if recenter_mode == "max":
                rhocom = np.unravel_index(newrho.argmax(), newrho.shape)
            else:
                rhocom = np.array(ndimage.measurements.center_of_mass(np.abs(newrho)))
            gridcenter = grid_center(newrho)  # (np.array(newrho.shape)-1.)/2.
            shift = gridcenter - rhocom
            shift = np.rint(shift).astype(int)
            newrho = np.roll(np.roll(np.roll(newrho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
            support = np.roll(np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        # update support using shrinkwrap method
        if shrinkwrap and j >= shrinkwrap_minstep and j % shrinkwrap_iter == 1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            if shrinkwrap_old_method:
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                swN = int(swV / dV)
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                    swV *= swV_decay
                else:
                    threshold = shrinkwrap_threshold_fraction
                    if first_time_swdensity:
                        if not quiet:
                            if gui:
                                my_logger.info("switched to shrinkwrap by density threshold = %.4f" % threshold)
                            else:
                                print("\nswitched to shrinkwrap by density threshold = %.4f" % threshold)
                        first_time_swdensity = False
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            if sigma > shrinkwrap_sigma_end:
                sigma = shrinkwrap_sigma_decay * sigma

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        # run erode when shrinkwrap is run
        if erode and shrinkwrap and j > shrinkwrap_minstep and j % shrinkwrap_iter == 1:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)
                support = cp.asnumpy(support)

            # eroded is the region of the support _not_ including the boundary pixels
            # so it is the entire interior. erode_region is _just_ the boundary pixels
            eroded = ndimage.binary_erosion(support, np.ones((erosion_width, erosion_width, erosion_width)))
            # get just boundary voxels, i.e. where support=True and eroded=False
            erode_region = np.logical_and(support, ~eroded)
            # set all negative density in boundary pixels to zero.
            newrho[(newrho < 0) & (erode_region)] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        if enforce_connectivity and j in enforce_connectivity_steps:
            if DENSS_GPU:
                newrho = cp.asnumpy(newrho)

            # first run shrinkwrap to define the features
            if shrinkwrap_old_method:
                # run the old method
                absv = True
                newrho, support = shrinkwrap_by_density_value(newrho, absv=absv, sigma=sigma, threshold=threshold,
                                                              recenter=recenter, recenter_mode=recenter_mode)
            else:
                # end this stage of shrinkwrap when the volume is less than a sphere of radius D/2
                swN = int(swV / dV)
                if swbyvol and swV > swVend:
                    newrho, support, threshold = shrinkwrap_by_volume(newrho, absv=True, sigma=sigma, N=swN,
                                                                      recenter=recenter, recenter_mode=recenter_mode)
                else:
                    newrho, support = shrinkwrap_by_density_value(newrho, absv=True, sigma=sigma, threshold=threshold,
                                                                  recenter=recenter, recenter_mode=recenter_mode)

            # label the support into separate segments based on a 3x3x3 grid
            struct = ndimage.generate_binary_structure(3, 3)
            labeled_support, num_features = ndimage.label(support, structure=struct)
            sums = np.zeros((num_features))
            num_features_to_keep = np.min([num_features, enforce_connectivity_max_features])
            if not quiet:
                if not gui:
                    print("EC: %d -> %d " % (num_features, num_features_to_keep))

            # find the feature with the greatest number of electrons
            for feature in range(num_features + 1):
                sums[feature - 1] = np.sum(newrho[labeled_support == feature])
            big_feature = np.argmax(sums) + 1
            # order the indices of the features in descending order based on their sum/total density
            sums_order = np.argsort(sums)[::-1]
            sums_sorted = sums[sums_order]
            # now grab the actual feature numbers (rather than the indices)
            features_sorted = sums_order + 1

            # remove features from the support that are not the primary feature
            # support[labeled_support != big_feature] = False
            # reset support to zeros everywhere
            # then progressively add in regions of support up to num_features_to_keep
            support *= False
            for feature in range(num_features_to_keep):
                support[labeled_support == features_sorted[feature]] = True

            # clean up density based on new support
            newrho[~support] = 0

            if DENSS_GPU:
                newrho = cp.array(newrho)
                support = cp.array(support)

        supportV[j] = mysum(support, DENSS_GPU=DENSS_GPU) * dV

        # convert possibly imaginary rg to string for printing
        rg_str = (f"{rg[j].real:3.2f}" if abs(rg[j].imag) < 1e-10 else
                  f"{rg[j].imag:3.2f}j" if abs(rg[j].real) < 1e-10 else
                  f"{rg[j].real:3.2f}{rg[j].imag:+3.2f}j")

        if not quiet:
            if gui:
                my_logger.info("% 5i % 4.2e %s       % 5i          ", j, chi[j], rg_str, supportV[j])
            else:
                sys.stdout.write("\r% 5i  % 4.2e %s       % 5i          " % (j, chi[j], rg_str, supportV[j]))
                sys.stdout.flush()

        # occasionally report progress in logger
        if j % 500 == 0 and not gui:
            my_logger.info('Step % 5i: % 4.2e %s       % 5i          ', j, chi[j], rg_str, supportV[j])

        if j > 101 + shrinkwrap_minstep:
            if DENSS_GPU:
                lesser = mystd(chi[j - 100:j], DENSS_GPU=DENSS_GPU).get() < chi_end_fraction * mymean(chi[j - 100:j],
                                                                                                      DENSS_GPU=DENSS_GPU).get()
            else:
                lesser = mystd(chi[j - 100:j], DENSS_GPU=DENSS_GPU) < chi_end_fraction * mymean(chi[j - 100:j],
                                                                                                DENSS_GPU=DENSS_GPU)
            if lesser:
                break

        rho = newrho

    # convert back to numpy outside of for loop
    if DENSS_GPU:
        rho = cp.asnumpy(rho)
        qbin_labels = cp.asnumpy(qbin_labels)
        qbin_args = cp.asnumpy(qbin_args)
        sigqdata = cp.asnumpy(sigqdata)
        Imean = cp.asnumpy(Imean)
        chi = cp.asnumpy(chi)
        qbins = cp.asnumpy(qbins)
        qbinsc = cp.asnumpy(qbinsc)
        Idata = cp.asnumpy(Idata)
        support = cp.asnumpy(support)
        supportV = cp.asnumpy(supportV)
        Idata = cp.asnumpy(Idata)
        newrho = cp.asnumpy(newrho)
        qblravel = cp.asnumpy(qblravel)
        xcount = cp.asnumpy(xcount)

    # F = myfftn(rho)
    F = myrfftn(rho)
    # calculate spherical average intensity from 3D Fs
    I3D = abs2(F)
    # I3D = myabs(F)**2
    Imean = mybinmean(I3D.ravel(), qblravel, xcount=xcount)

    # scale Fs to match data
    factors = np.sqrt(Idata / Imean)
    factors[~qba] = 1.0
    F *= factors[qbin_labels]
    # rho = myifftn(F)
    rho = myirfftn(F)
    rho = rho.real

    # negative images yield the same scattering, so flip the image
    # to have more positive than negative values if necessary
    # to make sure averaging is done properly
    # whether theres actually more positive than negative values
    # is ambiguous, but this ensures all maps are at least likely
    # the same designation when averaging
    if np.sum(np.abs(rho[rho < 0])) > np.sum(rho[rho > 0]):
        rho *= -1

    # scale total number of electrons
    if ne is not None:
        rho *= ne / np.sum(rho)

    rg[j + 1] = calc_rg_by_guinier_first_2_points(qbinsc, Imean)
    supportV[j + 1] = supportV[j]

    # change rho to be the electron density in e-/angstroms^3, rather than number of electrons,
    # which is what the FFT assumes
    rho /= dV
    my_logger.info('FINISHED DENSITY REFINEMENT')

    if cutout:
        # here were going to cut rho out of the large real space box
        # to the voxels that contain the particle
        # use D to estimate particle size
        # assume the particle is in the center of the box
        # calculate how many voxels needed to contain particle of size D
        # use bigger than D to make sure we don't crop actual particle in case its larger than expected
        # lets clip it to a maximum of 2*D to be safe
        nD = int(2 * D / dx) + 1
        # make sure final box will still have even samples
        if nD % 2 == 1:
            nD += 1

        nmin = nbox // 2 - nD // 2
        nmax = nbox // 2 + nD // 2 + 2
        # create new rho array containing only the particle
        newrho = rho[nmin:nmax, nmin:nmax, nmin:nmax]
        rho = newrho
        # do the same for the support
        newsupport = support[nmin:nmax, nmin:nmax, nmin:nmax]
        support = newsupport
        # update side to new size of box
        side = dx * (nmax - nmin)

    if write_xplor_format:
        write_xplor(rho, side, fprefix + ".xplor")
        write_xplor(np.ones_like(rho) * support, side, fprefix + "_support.xplor")

    write_mrc(rho, side, fprefix + ".mrc")
    write_mrc(np.ones_like(rho) * support, side, fprefix + "_support.mrc")

    # return original unscaled values of Idata (and therefore Imean) for comparison with real data
    Idata /= scale_factor
    sigqdata /= scale_factor
    Imean /= scale_factor
    I /= scale_factor
    sigq /= scale_factor

    # Write some more output files
    Iq_exp = np.vstack((qraw, Iraw, sigqraw)).T
    Iq_calc = np.vstack((qbinsc, Imean, Imean * 0.01)).T
    idx = np.where(Iraw > 0)
    Iq_exp = Iq_exp[idx]
    qmax = np.min([Iq_exp[:, 0].max(), Iq_calc[:, 0].max()])
    Iq_exp = Iq_exp[Iq_exp[:, 0] <= qmax]
    Iq_calc = Iq_calc[Iq_calc[:, 0] <= qmax]
    final_chi2, exp_scale_factor, offset, fit = calc_chi2(Iq_exp, Iq_calc, scale=True, offset=False, interpolation=True,
                                                          return_sf=True, return_fit=True)

    final_step = j+1

    chi[final_step] = final_chi2

    np.savetxt(fprefix + '_map.fit', fit, delimiter=' ', fmt='%.5e',
               header='q(data),I(data),error(data),I(density); chi2=%.3f' % final_chi2)

    # Create formatted strings for each column (enabling printing of complex rg values)
    chi_str = [f"{chi[i].real:.5e}" for i in range(final_step)]
    rg_str = [f"{rg[i].real:.5e}" if abs(rg[i].imag) < 1e-10 else f"{rg[i].imag:.5e}j"
              for i in range(final_step)]
    support_str = [f"{supportV[i].real:.5e}" for i in range(final_step)]

    np.savetxt(fprefix + '_stats_by_step.dat',
               np.column_stack((chi_str, rg_str, support_str)),
               delimiter=" ", fmt="%s", header='Chi2 Rg SupportVolume')

    my_logger.info('Number of steps: %i', j)
    my_logger.info('Final Chi2: %.3e', chi[-1])
    my_logger.info('Final Rg: %s', np.round(rg[-1],3))
    my_logger.info('Final Support Volume: %3.3f', supportV[-1])
    my_logger.info('Mean Density (all voxels): %3.5f', np.mean(rho))
    my_logger.info('Std. Dev. of Density (all voxels): %3.5f', np.std(rho))
    my_logger.info('RMSD of Density (all voxels): %3.5f', np.sqrt(np.mean(np.square(rho))))



    return qdata, Idata, sigqdata, qbinsc, Imean, chi, rg, supportV, rho, side, fit, final_chi2



