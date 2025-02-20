#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os, argparse
import denss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=str, nargs='+', help="FSC (Fourier Shell Correlation) filename(s) (multiple FSCs will be averaged)")
    parser.add_argument("--plot_on", dest="plot", action="store_true", help="Create simple plots of results (requires Matplotlib, default if module exists).")
    parser.add_argument("--plot_off", dest="plot", action="store_false", help="Do not create simple plots of results. (Default if Matplotlib does not exist)")
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
        fname_nopath = os.path.basename(args.files[0])
        basename, ext = os.path.splitext(fname_nopath)
        output = basename
    else:
        output = args.output

    nf = len(args.files)
    fscs = []
    for i in range(nf):
        fscs.append(np.loadtxt(args.files[i]))
    fscs = np.array(fscs)

    if nf==1:
        fsc = fscs[0]
    else:
        fsc = np.mean(fscs,axis=0)

    resn, x, y, resx = denss.fsc2res(fsc, return_plot=True)
    if np.min(fsc[:,1]) > 0.5:
        print("Resolution: < %.1f A (maximum possible)" % resn)
    else:
        print("Resolution: %.1f A" % resn)

    np.savetxt(output+'.dat',fsc,delimiter=' ',fmt='%.5e',header="1/resolution, FSC")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')
        for i in range(nf):
            plt.plot(fscs[i,:,0],fscs[i,:,1],'k--',alpha=0.1)
        plt.plot(fsc[:,0],fsc[:,1],'bo-')
        #plt.plot(x,y,'k-')
        plt.plot([resx],[0.5],'ro',label=f'Resolution = {resn:.2f} $\AA$')
        plt.legend()
        plt.xlabel('Resolution (1/$\AA$)')
        plt.ylabel('Fourier Shell Correlation')
        print(output)
        plt.savefig(output+'.png',dpi=150)
        plt.close()


if __name__ == "__main__":
    main()