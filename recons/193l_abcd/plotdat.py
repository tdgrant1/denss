
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl






d_params = np.arange(0.8, 1.2, 0.02)


sass =  np.zeros(  (len(d_params), 501))
for i_d, d in enumerate(d_params):

    dstr = f'{d:.2f}'.replace('.', 'p')

    astr = '1p00'
    bstr = '1p00'
    cstr = '1p00'

    sas = np.loadtxt(f'./193l_a{astr}_b{bstr}_c{cstr}_d{dstr}_insolvent.mrc2sas.dat')
    sass[i_d, : ] = sas[:,1]
    q = sas[:,0]




invsass = np.zeros( ( len(d_params), 501))
for i_d, d in enumerate(d_params):
    invsass[i_d,:] = sass[i_d, :]**(1/d)


ave_invsass = np.mean(invsass, axis=0)





fig, ax = plt.subplots()
ax.plot(q, ave_invsass, label='ave invesass')
i_d = 10
d = d_params[i_d]
ax.plot(q, sass[i_d], label=f'd={d_params[i_d]}')
plt.yscale('log')
plt.legend()



fig, ax = plt.subplots()

cmap = mpl.cm.winter
norm = mpl.colors.Normalize(vmin=np.min(d_params), vmax=np.max(d_params))

for d, sas in  zip(d_params, sass):
    ax.plot(q, sas, label=f'{d:.2f}', color = cmap(norm(d)))

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)

plt.yscale('log')
plt.title('d param')



fig, ax = plt.subplots()

cmap = mpl.cm.winter
norm = mpl.colors.Normalize(vmin=np.min(d_params), vmax=np.max(d_params))

for d, sas in  zip(d_params, invsass):
    ax.plot(q, sas, label=f'{d:.2f}', color = cmap(norm(d)))

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)

plt.yscale('log')
plt.title('d param invsass')



plt.show()




