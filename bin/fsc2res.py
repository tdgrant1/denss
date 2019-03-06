#!/usr/bin/env python

import numpy as np
import os, sys, argparse
import imp
try:
    imp.find_module('matplotlib')
    matplotlib_found = True
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_found = False

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, nargs='+', help="FSC (Fourier Shell Correlation) filename(s) (multiple FSCs will be averaged)")
parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
parser.add_argument("-o", "--output", default=None, help="Output filename prefix")
if matplotlib_found:
    parser.set_defaults(plot=True)
else:
    parser.set_defaults(plot=False)
args = parser.parse_args()

if args.output is None:
    basename, ext = os.path.splitext(args.file[0])
    output = basename+'_fsc'
else:
    output = args.output

def find_nearest_i(array,value):
    """Return the index of the array item nearest to specified value"""
    return (np.abs(array-value)).argmin()

nf = len(args.file)
fscs = []
for i in range(nf):
    fscs.append(np.loadtxt(args.file[i]))
fscs = np.array(fscs)

if nf==1:
    fsc = fscs[0]
else:
    fsc = np.mean(fscs,axis=0)

np.savetxt(output+'_avg.txt',fsc,delimiter=' ',fmt='%.5e')

x = np.linspace(fsc[0,0],fsc[-1,0],1000)
y = np.interp(x, fsc[:,0], fsc[:,1])

resi = np.argmin(y>=0.5)
resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])

resn = round(float(1./resx),1)

print "Resolution: %.1f" % resn, u'\u212B'.encode('utf-8')

if args.plot:
    import matplotlib.pyplot as plt
    plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')
    for i in range(nf):
        plt.plot(fscs[i,:,0],fscs[i,:,1],'k--',alpha=0.1)
    plt.plot(fsc[:,0],fsc[:,1],'bo-')
    #plt.plot(x,y,'k-')
    plt.plot([resx],[0.5],'ro',label='Resolution = '+str(resn)+r'$\mathrm{\AA}$')
    plt.legend()
    plt.xlabel('Resolution (1/$\mathrm{\AA}$)')
    plt.ylabel('Fourier Shell Correlation')
    plt.savefig(output,ext='png',dpi=150)
    plt.close()

