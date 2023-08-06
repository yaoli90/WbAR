import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import matplotlib.colors as cls
import numpy as np

def plot_u(x, u, title='$u(x;theta)$', ylim=[0,0.45]):
    plt.figure(figsize=(3,3))
    plt.plot(x, u)
    plt.ylim(ylim)
    plt.xlim([0,np.pi])
    plt.title(title)
    plt.show()


def plot_samples(x, f):
    plt.figure(figsize=(3,3))
    plt.plot(x ,f, '.')
    plt.ylim([-1,3])
    plt.xlim([0,np.pi])
    plt.show()
