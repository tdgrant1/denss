
import numpy as np
import matplotlib.pyplot as plt


sas1 = np.loadtxt('./193linvac_10_insolvent.mrc2sas.dat')
sas2 = np.loadtxt('./193linvac_075_insolvent.mrc2sas.dat')
sas3 = np.loadtxt('./193linvac_05_insolvent.mrc2sas.dat')
plt.figure()

plt.title('rho invacuo sf')
plt.plot(sas1[:,0], sas1[:,1], label='100%')
plt.plot(sas2[:,0], sas2[:,1], label='75%')
plt.plot(sas3[:,0], sas3[:,1], label='50%')
plt.legend()
plt.yscale('log')
plt.show()




