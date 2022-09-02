import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
import numpy as np
from numpy import ma

CMAP = cm.viridis


def surface_2d(X, Y, Z, info):

    # select linear or log scale
    norm = colors.LogNorm() if (info['scale'] in 'log') else colors.Normalize()

    fig, ax = plt.subplots(1, 1)

    cp = ax.scatter(X, Y, c=Z, cmap=CMAP, norm=norm)
    fig.colorbar(cp)

    # cp = plt.contour(X, Y, Z,
    #                  locator=ticker.LogLocator(),
    #                  colors='white',
    #                  linestyles='solid',
    #                  linewidths=0.5)

    ax.set_title(f"{info['name']} Test Function")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # plt.show()


def filled_contour(X, Y, Z):
    fig, ax = plt.subplots(1, 1)

    cp = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap=CMAP)
    fig.colorbar(cp)

    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
