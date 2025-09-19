
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl






x_params = np.arange(0.5, 2.5, 0.1)


cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0.5, vmax=1)




fig, ax = plt.subplots()

for x  in x_params:

    cstr = f'{x:.2f}'.replace('.', 'p')
    astr = '1p00'
    dstr = '1p00'
    bstr = '1p00'



    sas = np.loadtxt(f'./193l_a{astr}_b{bstr}_c{cstr}_d{dstr}_insolvent.mrc2sas.dat')
    print(sas)
    ax.plot(sas[:,0], sas[:,1], label=f'{x:.2f}', color = cmap(norm(x)))

plt.yscale('log')
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.colorbar(sm, ax=ax)

plt.title('a param')

plt.show()




