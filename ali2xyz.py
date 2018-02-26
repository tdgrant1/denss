#!/usr/bin/env python

import numpy as np
import os, sys
from EMAN2 import *

basename, ext = os.path.splitext(sys.argv[1])
e = EMData()
e.read_image(sys.argv[1])
an=Analyzers.get("inertiamatrix",{"verbose":0})
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
