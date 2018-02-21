#!/bin/bash

maps=$@

#Determine the best fitting enantiomer
#first create a reference from the given maps
#create eman2 stack of original volumes

e2buildstacks.py --stackname stack.hdf $maps
#create reference from original volumes
e2spt_binarytree.py --input=stack.hdf
cp spt_01/final_avg.hdf reference.hdf

#for each of the 20 reconstructions, determine best enantiomer
#first generate the enantiomers, then compare them against
#the reference to identify the best

for map in $maps;
do
    #align individual map to principal axes
    ali2xyz.py $map
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
    e2buildstacks.py --stackname ${map%.*}_allali2xyz.hdf ${ali2xyz%.*}*.hdf

    #for some reason e2spt_align.py fails with a malloc error randomly
    #simply repeating this same command until it works appears to do the trick
    #until a bug fix is released
    repeat="True"
    until [ $repeat == "False" ];
    do
        #align each of the enantiomers to the reference
        echo "Aligning enantiomers..."
        e2spt_align.py --path=spt_01 ${map%.*}_allali2xyz.hdf reference.hdf
        #read .json file to extract which enantiomer has the best fit (the most negative score, -1 is perfect)
        if [ -f spt_01/particle_parms_02.json ];
        then
            mv spt_01/particle_parms_02.json spt_01/particle_parms_01.json
        fi
        best_i=`grep "score" spt_01/particle_parms_01.json | awk '{print NR, $2*1}' | sort -k 2 -n | head -n 1 | awk '{print $1}'`
        if [ "${best_i}" == "" ];
        then
            echo "Calculation failed. Trying again."
            repeat="True"
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

