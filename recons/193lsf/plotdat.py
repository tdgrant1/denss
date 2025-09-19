
import numpy as np
import matplotlib.pyplot as plt


sas1 = np.loadtxt('./193linvac_10_insolvent.mrc2sas.dat')
sas2 = np.loadtxt('./193linvac_05_insolvent.mrc2sas.dat')
plt.figure()

stds = []
print(f'mean ratio\t', f'std')
for i in range(1,101):
    sas = np.loadtxt(f'./193lsf_{i}_insolvent.mrc2sas.dat')
    sas[sas1 != 0] /=sas1[ sas1 != 0]
    plt.plot(sas1[:,0], sas[:,1], label=f'sf={i/10}')
    # print(round(sas[100][1],3), '\t\t', round((i/10)**2,3))
    print(round(sas[:,1].mean(), 3), f'\t\t{sas[:,1].std():.2g}')
    stds.append(sas[:,1].std())
plt.legend()
# plt.yscale('log')
 
plt.figure()
plt.plot( stds)
plt.ylabel('Standard dev.')
plt.xlabel('% sf')

plt.show()




