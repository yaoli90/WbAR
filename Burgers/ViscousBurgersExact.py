import numpy as np
from numpy import cos
from numpy import exp
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'cm',
    "font.serif": ['Times New Roman'],
    "axes.unicode_minus": False
}
rcParams.update(config)

gx, gw = np.polynomial.hermite.hermgauss(160)
# print(gx)
# print(gw)


# def vis_burg_point(xx, tt, mmu, ghx, ghw):
#     f = exp(0.5 * (cos(2 * sqrt(mmu * tt) * ghx + xx) - 1) / mmu)
#     ff = -2 * sqrt(mmu / tt) * ghx * f
#     return np.sum(ghw * ff) / np.sum(ghw * f)


def vis_burg_point(xx, tt, mmu, ghx, ghw):
    cc = 2 * sqrt(mmu * tt)
    pp = 0.5 * (cos(cc * ghx + xx) - 1) / mmu
    f = exp(pp)
    ff = -2 * sqrt(mmu / tt) * ghx * f
    numerator = np.sum(ghw * ff)
    denominator = np.sum(ghw * f)
    return numerator / denominator


def vis_burg(xx, tt, mmu, ghx, ghw):
    nn = np.size(xx)
    uu = np.empty(xx.shape)
    for i in range(nn):
        uu[i] = vis_burg_point(xx[i], tt, mmu, ghx, ghw)
    return uu


def ue_vis_burg(xx, tt, mmu, ghx, ghw):
    n1, n2 = xx.shape[0], xx.shape[1]
    uu = np.empty(xx.shape)
    for i in range(n1):
        for j in range(n2):
            uu[i, j] = vis_burg_point(xx[i, j], tt, mmu, ghx, ghw)
    return uu


def burgers_gt(ts):
    mu = 0.01/np.pi
    n = 256
    x = np.linspace(0, 2*np.pi, n)
    us = np.zeros((256,len(ts)))
    for i in range(len(ts)):
        t = ts[i]
        if t >0:
            u = vis_burg(x, t, mu, gx, gw)
        else:
            u = np.sin(x)
        us[:,i] = u
    return us

def burgers_gt_x_t(X_T):
    mu = 0.01/np.pi
    U = np.zeros((len(X_T),1))
    for i in range(len(X_T)):
        x, t = X_T[i]
        t = t*np.pi
        x = (x+1)*np.pi
        if t >0:
            u = vis_burg_point(x, t, mu, gx, gw)
        else:
            u = np.sin(x)
        U[i] = u
    return U




'''
print('x[5] =', x[5])
print('u[5] =', u[5])
print('max u =', np.max(u))

fig, ax = plt.subplots()
ax.scatter(x, u, c='r', marker='+', zorder=20, label='exact solution')
# ax.plot(x1, ue_burgers(x1, T), c='mediumaquamarine', zorder=10, label='exact')
ax.set_title('Viscous Burgers equation - exact solution')
ax.legend(loc='best', fontsize=13.5)
ax.grid(c='lightgray', linestyle='--')
fig.tight_layout()
plt.minorticks_on()
plt.show()
'''
