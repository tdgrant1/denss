#!/usr/bin/env python
#
#    SAXStats
#    A collection of python functions useful for solution scattering
#
#    Tested using Anaconda / Python 2.7
#
#    Author: Thomas D. Grant
#    Email:  <tgrant@hwi.buffalo.edu>
#    Copyright 2017 The Research Foundation for SUNY
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

import sys, logging, datetime
import numpy as np
from scipy import spatial, ndimage

def chi2(exp, calc, sig):
    """Return the chi2 discrepancy between experimental and calculated data"""
    return np.sum(np.square(exp - calc) / np.square(sig))

def center_rho(rho):
    """Move electron density map so its center of mass aligns with the center of the grid by interpolation"""
    rhocom = np.array(ndimage.measurements.center_of_mass(rho))
    gridcenter = np.array(rho.shape)/2.
    shift = gridcenter-rhocom
    rho = ndimage.interpolation.shift(rho,shift,order=3,mode='wrap')
    return rho

def center_rho_roll(rho):
    """Move electron density map so its center of mass aligns near the center of the grid by rolling array"""
    rhocom = np.array(ndimage.measurements.center_of_mass(rho))
    gridcenter = np.array(rho.shape)/2.
    shift = gridcenter-rhocom
    shift = shift.astype(int)
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    return rho

def rho2rg(rho,side=None,r=None,support=None,dx=None):
    """Calculate radius of gyration from an electron density map."""
    if side is None and r is None:
        print "Error: To calculate Rg, must provide either side or r parameters."
        sys.exit()
    if side is not None and r is None:
        n = rho.shape[0]
        x_ = np.linspace(-side/2.,side/2.,n)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        r = np.sqrt(x**2 + y**2 + z**2)
    if support is None:
        support = np.ones_like(rho,dtype=bool)
    if dx is None:
        print "Error: To calculate Rg, must provide dx"
        sys.exit()
    rhocom = (np.array(ndimage.measurements.center_of_mass(rho))-np.array(rho.shape)/2.)*dx
    rg2 = np.sum(r[support]**2*rho[support])/np.sum(rho[support])
    rg2 = rg2 - np.linalg.norm(rhocom)**2
    rg = np.sign(rg2)*np.abs(rg2)**0.5
    return rg

def write_xplor(rho,side,filename="map.xplor"):
    """Write an XPLOR formatted electron density map."""
    xs, ys, zs = rho.shape
    title_lines = ['REMARK FILENAME="'+filename+'"','REMARK DATE= '+str(datetime.datetime.today())]
    f = open(filename,'wb')
    f.write("\n")
    f.write("%8d !NTITLE\n" % len(title_lines))
    for line in title_lines:
        f.write("%-264s\n" % line)
    f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (xs,-xs/2+1,xs/2,ys,-ys/2+1,ys/2,zs,-zs/2+1,zs/2))
    f.write("% -.5E% -.5E% -.5E% -.5E% -.5E% -.5E\n" % (side,side,side,90,90,90))
    f.write("ZYX\n")
    for k in range(zs):
        f.write("%8s\n" % k)
        for j in range(ys):
            for i in range(xs):
                if (i+j*ys) % 6 == 5:
                    f.write("% -.5E\n" % rho[i,j,k])
                else:
                    f.write("% -.5E" % rho[i,j,k])
        f.write("\n")
    f.write("    -9999\n")
    f.write("  %.4E  %.4E" % (np.average(rho), np.std(rho)))
    f.close()

def read_xplor(filename="map.xplor"):
    """Read an XPLOR formatted electron density map."""
    with open(filename) as f:
        n = 0
        for line in f:
            n+=1
            if n==5:
                na, amin, amax, nb, bmin, bmax, nc, cmin, cmax = map(int, line.split())
                print line
                print na, amin, amax, nb, bmin, bmax, nc, cmin, cmax
                break
    rho = np.zeros((na, nb, nc))
    i = 0
    j = 0
    k = 0
    n = 0
    x = []
    y = []
    with open(filename) as f:
        for line in f:
            n+=1
            #ignore header, section headers, and last two lines
            if n>7 and n%(int((na*nb)/6) + (lambda x: 0 if(x%6==0) else 1)(na*nb) + 1) - 8 != 0 and (n-8)/(int((na*nb)/6) + (lambda x: 0 if(x%6==0) else 1)(na*nb) + 1)<nc:
                x.append(zip(*(iter(line),) * 12))
    for i in range(len(x)):
        for j in range(len(x[i])):
            y.append(float(''.join(x[i][j])))
    print len(y)
    rho = np.array(y).reshape((na,nb,nc))
    return rho

class Xplor(object):
    """Xplor electron density map."""
    def __init__(self, argument):
        if isinstance(argument, str):
            self.read(argument)
        elif isinstance(argument, np.ndarray):
            self.rho = argument
        else:
            pass

    def attributes(self):
        """Print required attributes, and current settings"""
        print "self.header %10s" % self.header
        print "self.na %10s" % self.na
        print "self.amin %10s" % self.amin
        print "self.amax %10s" % self.amax
        print "self.nb %10s" % self.nb
        print "self.bmin %10s" % self.bmin
        print "self.bmax %10s" % self.bmax
        print "self.nc %10s" % self.nc
        print "self.cmin %10s" % self.cmin
        print "self.cmax %10s" % self.cmax
        print "self.a %10s" % self.a
        print "self.b %10s" % self.b
        print "self.c %10s" % self.c
        print "self.rho %10s" % np.array(self.rho.shape)

    def read(self,filename="map.xplor"):
        """Read an XPLOR formatted electron density map."""
        self.header = []
        with open(filename) as f:
            n = 0
            for line in f:
                n+=1
                if n==3:
                    self.header.append(line)
                if n==4:
                    self.header.append(line)
                if n==5:
                    self.na, self.amin, self.amax, self.nb, self.bmin, self.bmax, self.nc, self.cmin, self.cmax = map(int, line.split())
                if n==6:
                    self.a, self.b, self.c, self.alpha, self.beta, self.gamma = map(float,line.split())
                    break
        i = 0
        j = 0
        k = 0
        n = 0
        x = []
        y = []
        with open(filename) as f:
            for line in f:
                n+=1
                #ignore header, section headers, and last two lines
                if (
                    n>7
                    and
                    n % (int((self.na*self.nb)/6) + (lambda x: 0 if(x%6==0) else 1)(self.na*self.nb) + 1) - 8 != 0
                    and
                    (n-8)/(int((self.na*self.nb)/6) + (lambda x: 0 if(x%6==0) else 1)(self.na*self.nb) + 1) < self.nc
                    ):
                    x.append(zip(*(iter(line),) * 12))
        for i in range(len(x)):
            for j in range(len(x[i])):
                y.append(float(''.join(x[i][j])))
        self.rho = np.array(y).reshape((self.na,self.nb,self.nc))
        self.rho = np.swapaxes(self.rho,0,2)

    def write(self,filename="map.xplor"):
        """Write an XPLOR formatted electron density map."""
        self.header = ['REMARK FILENAME="'+filename+'"','REMARK DATE= '+str(datetime.datetime.today())]
        f = open(filename,'wb')
        f.write("\n")
        f.write("%8d !NTITLE\n" % len(self.header))
        for line in self.header:
            f.write("%-264s\n" % line)
        f.write("%8d%8d%8d%8d%8d%8d%8d%8d%8d\n" % (self.na,-self.na/2+1,self.na/2,self.nb,-self.nb/2+1,self.nb/2,self.nc,-self.nc/2+1,self.nc/2))
        f.write("% -.5E% -.5E% -.5E% -.5E% -.5E% -.5E\n" % (self.a,self.b,self.c,self.alpha,self.beta,self.gamma))
        f.write("ZYX\n")
        for k in range(self.nc):
            f.write("%8s\n" % k)
            for j in range(self.nb):
                for i in range(self.na):
                    if (i+j*self.nb) % 6 == 5:
                        f.write("% -.5E\n" % self.rho[i,j,k])
                    else:
                        f.write("% -.5E" % self.rho[i,j,k])
            f.write("\n")
        f.write("    -9999\n")
        f.write("  %.4E  %.4E" % (np.average(self.rho), np.std(self.rho)))
        f.close()

def pdb2support(pdb,D,voxel=10.,oversampling=1.0,filename="support_dam.pdb",radii=None):
    """Calculate simple support from pdb coordinates."""
    side = oversampling*D
    halfside = side/2
    n = int(side/voxel)
    if n%2==0: n += 1
    dx = side/n
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    xyz = np.column_stack([x.flat,y.flat,z.flat])
    support = np.zeros(x.shape,dtype=bool)
    xyz_nearby = []
    xyz_nearby_i = []
    #assumes atom radius of 1.7 angstroms plus 1.5 for solvent molecule.
    #ideally should replace this with something like Connolly surface.
    if radii is None:
        radii = np.ones((pdb.natoms)) * (1.7 + 1.5)
    else:
        radii = np.atleast_1d(radii)
        if radii.size == 1:
            radii = np.ones((pdb.natoms)) * radii
    for i in range(pdb.natoms):
        xyz_nearby_i.append(np.where(spatial.distance.cdist(pdb.coords[i,None],xyz) < 3.2)[1])
    xyz_nearby_i = np.unique(np.concatenate(xyz_nearby_i))
    writedam(filename,xyz[xyz_nearby_i])
    support[np.unravel_index(xyz_nearby_i,support.shape)] = True
    return support

def denss(q,I,sigq,D,supportpdb=None,rhostart=None,ne=None,rhobounds=None,voxel=10.,oversampling=2.,usedmax=False,recenter=True,positivity=True,extrapolate=True,write=True,filename="map",steps=1000,seed=None,shrinkwrap=True,shrinkwrap_sigma_start=3,shrinkwrap_sigma_end=1.5,shrinkwrap_sigma_decay=0.99,shrinkwrap_threshold_fraction=0.2,shrinkwrap_iter=20,shrinkwrap_minstep=100,chi_end_fraction=0.001):
    """Calculate electron density from scattering data."""
    side = oversampling*D
    halfside = side/2
    n = int(side/voxel)
    #want odd n so that there exists an F[0,0,0] in the center that equals the number of electrons for easy scaling
    if n%2==0: n += 1
    dx = side/n
    dV = dx**3
    V = side**3
    x_ = np.linspace(-halfside,halfside,n)
    x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    df = 1/side
    qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi
    qz_ = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
    qx, qy, qz = np.meshgrid(qx_,qx_,qz_,indexing='ij')
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    #create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    #only move the non-zero terms, since the zeroth term should be at q=0.
    qbinsc[1:] += qstep/2.
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.digitize(qr, qbins)
    qbin_labels -= 1
    #allow for any range of q data
    qdata = qbinsc[np.where( (qbinsc>=q.min()) & (qbinsc<=q.max()) )]
    Idata = np.interp(qdata,q,I)
    qbin_args_data = range(np.where(qbinsc==qdata.min())[0][0], np.where(qbinsc==qdata.max())[0][0]+1)
    if extrapolate:
        qextend = qbinsc[np.where(qbinsc>=qdata.max())]
        Iextend = qextend**-4
        Iextend = Iextend/Iextend[0] * Idata[-1]
        qdata = np.concatenate((qdata,qextend[1:]))
        Idata = np.concatenate((Idata,Iextend[1:]))
    #how to scale Idata if I(0) is not known, i.e. if only WAXS data?
    realne = np.copy(ne)
    factor = ne**2/Idata[0]
    Idata *= factor
    sigqdata = np.interp(qdata,q,sigq)
    sigqdata *= factor
    #create list of qbin indices just in region of data for later F scaling
    qbin_args = range(np.where(qbinsc==qdata.min())[0][0], np.where(qbinsc==qdata.max())[0][0]+1)
    Imean = np.zeros((steps,len(qbins)))
    indices = np.zeros(len(qbins),dtype='O')
    for i in range(len(qbins)):
        indices[i] = np.where(qbin_labels==i)
    chi = np.zeros((steps+1))
    rg = np.zeros((steps+1))
    supportV = np.zeros((steps+1))
    chibest = np.inf
    beta = 0.9 #HIO feedback parameter. Not currently in use.
    if rhobounds is not None:
        rho_bounds = np.array(rhobounds)
        print rho_bounds
    usesupport = True
    if supportpdb is not None and usesupport:
        support = pdb2support(supportpdb, D=D, voxel=voxel, oversampling=oversampling)
        print "Support Volume = %d" % (dV*np.sum(support))
        writepdb(supportpdb,filename+"-support-aligned.pdb")
        write_xplor(np.ones_like(x)*support,side,filename+"_support.xplor")
    elif usesupport and usedmax:
        support = np.zeros(x.shape,dtype=bool)
        support[r<=D*0.6] = True
    else:
        support = np.ones(x.shape,dtype=bool)
    if seed is None:
        seed = np.random.randint(2**32-1)
    else:
        seed = int(seed)
    print "Seed = %i " % seed
    prng = np.random.RandomState(seed)
    rho = prng.random_sample(size=x.shape)
    rho *= ne / np.sum(rho)
    update_support = True
    sigma = shrinkwrap_sigma_start
    if rhostart is not None:
        rhostart = Xplor(rhostart)
        rho = np.fft.irfftn(np.fft.rfftn(rhostart.rho),r.shape)
        rho *= ne / np.sum(rho)

    logging.info('Grid size (voxels): %i x %i x %i', n, n, n)
    logging.info('Real space box width (angstroms): %3.3f', side)
    logging.info('Real space box range (angstroms): %3.3f < x < %3.3f', x_.min(), x_.max())
    logging.info('Real space box volume (angstroms^3): %3.3f', V)
    logging.info('Real space voxel size (angstroms): %3.3f', dx)
    logging.info('Real space voxel volume (angstroms^3): %3.3f', dV)
    logging.info('Reciprocal space box width (angstroms^(-1)): %3.3f', qx_.max()-qx_.min())
    logging.info('Reciprocal space box range (angstroms^(-1)): %3.3f < qx < %3.2f', qx_.min(), qx_.max())
    logging.info('Maximum q vector (diagonal) (angstroms^(-1)): %3.3f', qr.max())
    logging.info('Number of q shells: %i', nbins)
    logging.info('Width of q shells (angstroms^(-1)): %3.3f', qstep)
    if rhostart is not None:
        logging.info('Random seed: N/A. Starting map: %s', rhostart)
    else:
        logging.info('Random seed: %i', seed)

    print "Step  Chi2      Rg      Support Volume"
    print "----- --------- ------- --------------"
    for j in range(steps):
        F = np.fft.rfftn(rho)
        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        Imean[j] = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
        #scale Fs to match data
        qbin_args2 = qbin_args #[:5+j/300]
        for i in qbin_args2:
            I_data = Idata[i]
            I_calc = Imean[j,i]
            if not np.allclose(I_calc,0.0):
                factor = np.sqrt(I_data/I_calc)
                F[indices[i]] = F[indices[i]] * factor
        chi[j] = np.sum(((Imean[j,qbin_args_data]-Idata[qbin_args_data])/sigqdata[qbin_args_data])**2)/len(qbin_args_data)
        #APPLY REAL SPACE RESTRAINTS
        rhoprime = np.fft.irfftn(F,rho.shape)
        rg[j] = rho2rg(rhoprime,r=r,support=support,dx=dx)
        newrho = np.zeros_like(rho)
        #Solvent flattening
        newrho[support] = rhoprime[support]
        newrho[~support] = 0.0
        #Hybrid Input-Output
        #newrho[support] = rhoprime[support]
        #newrho[~support] = rho[~support] - beta*rhoprime[~support]
        #enforce positivity by making all negative density points zero.
        if positivity:
            newrho[newrho<0] = 0.0
            newrho *= ne / np.sum(newrho)
        if rhobounds is not None:
            outofbounds = np.zeros_like(support)
            outofbounds[(newrho<rho_bounds[0])&(newrho>rho_bounds[1])] = True
            neoutofbounds = np.sum(newrho[outofbounds])
            if neoutofbounds != 0:
                newrho.clip(rho_bounds[0],rho_bounds[1])
                newrho[~outofbounds] *= (ne - np.sum(newrho[~outofbounds])) / neoutofbounds
        #update support using shrinkwrap method
        if shrinkwrap and j >= shrinkwrap_minstep and j%shrinkwrap_iter==0:
            if recenter: newrho = center_rho(newrho)
            rho_blurred = ndimage.filters.gaussian_filter(newrho,sigma=sigma)
            support = np.zeros(rho.shape,dtype=bool)
            support[rho_blurred >= shrinkwrap_threshold_fraction*rho_blurred.max()] = True
            if usedmax: support[r>0.6*D] = False
            if sigma > shrinkwrap_sigma_end:
                sigma = shrinkwrap_sigma_decay*sigma
        supportV[j] = np.sum(support)*dV
        sys.stdout.write("\r% 5i % 4.2e % 3.2f       % 5i          " % (j, chi[j], rg[j], supportV[j]))
        sys.stdout.flush()
        
        if j > 101 + shrinkwrap_minstep and np.std(chi[j-100:j]) < chi_end_fraction * np.median(chi[j-100:j]) and 1 == 1:
            break
        else:
            rho = newrho
            rho *= ne / np.sum(rho)
            F = np.fft.rfftn(rho)
            Fbest = F
            rhobest = rho
            Imeanbest = Imean[j]
            chis = chi
            update_support = True
    print
    F = Fbest
    
    #calculate spherical average intensity from 3D Fs
    Imean = ndimage.mean(np.abs(F)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    #scale Fs to match data one last time
    for i in qbin_args2:
        I_data = Idata[i]
        I_calc = Imean[i]
        if not np.allclose(I_calc,0.0):
            factor = np.sqrt(I_data/I_calc)
            F[indices[i]] = F[indices[i]] * factor
    chi[j+1] = np.sum(((Imean[qbin_args]-Idata)/sigqdata)**2)/len(qbin_args)
    rho = np.fft.irfftn(F,rho.shape)
    rho *= realne / np.sum(rho)

    rg[j+1] = rho2rg(rho=rho,r=r,support=support,dx=dx)
    supportV[j+1] = supportV[j]

    if write:
        write_xplor(rho,side,filename+".xplor")
        write_xplor(np.ones_like(rho)*support,side,filename+"_support.xplor")

    logging.info('Number of steps: %i', j)
    logging.info('Final Chi2: %3.3f', chis[j+1])
    logging.info('Final Rg: %3.3f', rg[j+1])
    logging.info('Final Support Volume: %3.3f', supportV[j+1])

    return qdata, Idata, sigqdata, qbinsc, Imean, chi, rg, supportV


