import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import matplotlib.colors as cls
import numpy as np

def plot_u(u, x, t, log=False, title='$u(x;theta)$'):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    if not log:
        h = ax.imshow(u, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(u, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', norm=cls.LogNorm(vmin=1e-5,vmax=1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(title, fontsize = 10)
    return ax

def plot_x(u, U_gt, x, pos):
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,U_gt.T[pos[0],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[0],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,U_gt.T[pos[1],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[1],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,U_gt.T[pos[2],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[2],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.75s$', fontsize = 10)

def plot_u_x(u, U_gt, x, t, pos=[50,100,150], title='$u(x;theta)$'):
    ax = plot_u(u, x, t, title=title)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[pos[0]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[pos[1]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[pos[2]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    plot_x(u, U_gt, x, pos)
    plt.show()

def plot_samples(x, f):
    sample_mse = np.mean(np.square(f))
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.scatter(x[:,1],x[:,0], c=f, cmap="rainbow", norm=cls.LogNorm(vmin=1e-5), s=15)
    ax.set_xlabel('adversarial samples, %.5f' % sample_mse)
    ax.set_xlim([0,1])
    ax.set_ylim([-1,1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    fig.colorbar(h, cax=cax)
    plt.show()
