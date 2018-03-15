#!/bin/bash

while getopts hf:j:n: opt; do
  case $opt in
    h)
      echo ""
      echo " ------------------------------------------------------------------------------ "
      echo " best_enantiomers.sh is a script that uses EMAN2 to generate and select the most "
      echo " similar enantiomers among a set of 3D volumes. "
      echo ""
      echo " -f: filenames of 3D volumes (space separated list enclosed in quotes, e.g. \"*[0-9].mrc\" )"
      echo " -j: the number of cores to use for parallel processing (default one less than system)"
      echo " -n: the number of failed attempts allowed to calculate enantiomer until skipping (default 5)"
      echo " ----------------------------------------------------------------------------- "
      echo ""
      exit 0
      ;;
    f)
      maps=$OPTARG
      ;;
    j)
      j=$OPTARG
      ;;
    n)
      attempts=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." #>&2
      exit 1
      ;;
  esac
done

if [ -z $j ];
then
    cores=`getconf _NPROCESSORS_ONLN 2>/dev/null || getconf NPROCESSORS_ONLN 2>/dev/null || echo 1` ;
    if [ $cores -eq 1 ];
    then
        j=`echo $cores | awk '{print $1}'`
    else
        j=`echo $cores | awk '{print $1-1}'`
    fi
fi
echo "Using $j processors in the best_enantiomers.sh script"

if [ -z $attempts ]; then attempts=5 ; fi

#Determine the best fitting enantiomer
#first create a reference from the given maps
#create eman2 stack of original volumes

e2buildstacks.py --stackname stack.hdf $maps
#create reference from original volumes
e2spt_binarytree.py --path=spt_en --input=stack.hdf --parallel=thread:${j}
cp spt_en_01/final_avg.hdf reference.hdf

#for each of the 20 reconstructions, determine best enantiomer
#first generate the enantiomers, then compare them against
#the reference to identify the best
mkdir spt_ali

#create python script for alignment on the fly
#first identify the needed EMAN2 python location
e2shebang=`head -n 1 \`which e2proc3d.py\``
#then write the script, placing the eman2 shebang at the top
echo "${e2shebang}

import numpy as np
import os, sys
from EMAN2 import *

basename, ext = os.path.splitext(sys.argv[1])
e = EMData()
e.read_image(sys.argv[1])
an=Analyzers.get(\"inertiamatrix\",{\"verbose\":0})
an.insert_image(e)
mxi=an.analyze()
mx=EMNumPy.em2numpy(mxi[0])
eigvv=np.linalg.eig(mx)
eig=[(1.0/eigvv[0][i],eigvv[1][:,i]) for i in xrange(3)]
eig=sorted(eig)
T=np.array([eig[0][1],eig[1][1],eig[2][1]])
T=Transform((float(i) for i in (eig[0][1][0],eig[0][1][1],eig[0][1][2],0,
                                eig[1][1][0],eig[1][1][1],eig[1][1][2],0,
                                eig[2][1][0],eig[2][1][1],eig[2][1][2],0)))
e.transform(T)
e.write_image(basename+'_ali2xyz.hdf')
" > ali2xyz.py
chmod +x ./ali2xyz.py

for map in $maps;
do
    #align individual map to principal axes
    ./ali2xyz.py $map
    ali2xyz=${map%.*}_ali2xyz.hdf
    enants=(${ali2xyz} \
            ${ali2xyz%.*}_x.hdf \
            ${ali2xyz%.*}_y.hdf \
            ${ali2xyz%.*}_z.hdf \
            ${ali2xyz%.*}_xy.hdf \
            ${ali2xyz%.*}_xz.hdf \
            ${ali2xyz%.*}_yz.hdf \
            ${ali2xyz%.*}_xyz.hdf)

    #create 7 enantiomers (for total of 8) by flipping over each axis
    e2proc3d.py ${enants[0]} ${enants[1]} --process xform.flip:axis=x
    e2proc3d.py ${enants[0]} ${enants[2]} --process xform.flip:axis=y
    e2proc3d.py ${enants[0]} ${enants[3]} --process xform.flip:axis=z
    e2proc3d.py ${enants[1]} ${enants[4]} --process xform.flip:axis=y
    e2proc3d.py ${enants[1]} ${enants[5]} --process xform.flip:axis=z
    e2proc3d.py ${enants[2]} ${enants[6]} --process xform.flip:axis=z
    e2proc3d.py ${enants[4]} ${enants[7]} --process xform.flip:axis=z

    #create stack of enantiomers for alignment
    e2buildstacks.py --stackname ${map%.*}_allali2xyz.hdf ${enants[@]}

    #for some reason e2spt_align.py fails with a malloc error randomly
    #simply repeating this same command until it works appears to do the trick
    #until a bug fix is released
    repeat="True"
    counter=0
    until [ $repeat == "False" ];
    do
        sleep 0.5 #included just to give some time in case of a keyboard interrupt
        rm -f spt_ali/particle_parms_*.json
        #align each of the enantiomers to the reference
        echo "Aligning enantiomers..."
        e2spt_align.py --path=spt_ali --threads=${j} ${map%.*}_allali2xyz.hdf reference.hdf
        #read .json file to extract which enantiomer has the best fit (the most negative score, -1 is perfect)
        best_i=`grep "score" spt_ali/particle_parms_01.json | awk '{print NR, $2*1}' | sort -k 2 -n | head -n 1 | awk '{print $1-1}'`
        if [ "${best_i}" == "" ];
        then
            if [ $counter -le $attempts ];
            then
                echo "Calculation failed. Trying again (Attempt # ${counter})."
                repeat="True"
                let counter++
            else
                echo "Calculation failed. Exceeded number of tries (5). Skipping to next file."
                repeat="False"
            fi
        else
            repeat="False"
        fi
    done
    best_enant=${enants[${best_i}]}
    echo "Best enantiomer: " $best_enant
    for i in ${enants[*]} ;
    do
        if [ "$i" == "$best_enant" ];
        then
            mv $i ${map%.*}_enant.hdf
            echo $i " renamed to " ${map%.*}_enant.hdf
        else
            rm -f $i
        fi
    done
    rm -f ${map%.*}_allali2xyz.hdf

done

