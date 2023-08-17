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
    gs0.update(top=1-0.06, bottom=1-0.46, left=0.15, right=0.55, wspace=0)
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
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title, fontsize = 10)
    plt.show()
    return ax

def plot_x(u, U_gt, x, pos):
    fig, ax = plt.subplots()
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(top=1-0.06, bottom=1-0.46, left=0.15, right=0.95, wspace=0)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,U_gt[pos,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u[pos,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,y)$')
    ax.set_title('$x = 0.5$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

def plot_u_x(u, U_gt, x, y, pos=192, title='$u(x;theta)$'):
    ax = plot_u(u, x, y, title=title)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(y[pos]*np.ones((2,1)), line, 'w-', linewidth = 1)
    plot_x(u, U_gt, x, pos)
    plt.show()

def plot_samples(s, f):



    sample_mse = np.mean(np.square(f))
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-0.46, left=0.15, right=0.55, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.scatter(s[:,1],s[:,0], c=f, cmap="rainbow", norm=cls.LogNorm(vmin=1e-5), s=15)
    ax.set_xlabel('adversarial samples, %.5f' % sample_mse)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('square')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.colorbar(h, cax=cax)
    plt.show()
