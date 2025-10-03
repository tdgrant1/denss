

import matplotlib.pyplot as plt
import numpy as np




x = np.linspace(-2, 2, 100)

xx,yy,zz = np.meshgrid(x,x,x)


rho = np.exp(- (xx**2) - (yy**2) - (zz**2))

plt.figure()
plt.imshow(rho[50,:,:])
plt.colorbar()


d_params = [0.8, 0.9, 1, 1.1, 1.2]


plt.show()
