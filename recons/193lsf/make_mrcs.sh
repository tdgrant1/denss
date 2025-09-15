#!/bin/bash




for i in $(seq 1 100);do
    echo denss-pdb2mrc -f ./193l.pdb -o 193lsf_${i} --PAscalefactor $(echo "scale=2; ${i}/100"|bc);
done
