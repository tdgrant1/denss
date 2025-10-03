
import numpy as np




a_params = [1]
c_params = [1]
b_params = [1]
# d_params = [0.8, 0.9, 1, 1.1, 1.2]
d_params = np.arange(0.8, 1.2, 0.02)



for a in a_params:
    for b in b_params:
        for c in c_params:
            for d in d_params:


                astr = f'{a:.2f}'.replace('.', 'p')
                bstr = f'{b:.2f}'.replace('.', 'p')
                cstr = f'{c:.2f}'.replace('.', 'p')
                dstr = f'{d:.2f}'.replace('.', 'p')

                pdb2mrc_cmd = f'denss-pdb2mrc -f ../../inputs/193l.pdb -o 193l_a{astr}_b{bstr}_c{cstr}_d{dstr} --PArhoinvacuosf {a} --PAsf_ex {b} --PAsf_sh {c} --PApowersf {d}'
                print(pdb2mrc_cmd)

                mrc2sas_cmd = f'denss-mrc2sas -f 193l_a{astr}_b{bstr}_c{cstr}_d{dstr}_insolvent.mrc'
                print(mrc2sas_cmd)


