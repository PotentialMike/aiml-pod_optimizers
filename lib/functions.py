import numpy as np
import random

import lib.plots as plot

NUM_POINTS = 200


def picker():
    # func_list = [Ackley, Beale, Easom, Himmelblau,
    #              HolderTable, Rastrigin]
    # func_list = [Ackley, Beale, Easom, Himmelblau]
    func_list = [Beale]
    return random.choice(func_list)()


class TestFunction:
    '''
        Parent class to the specific test functions
    '''

    def plot_surface(self):
        x = np.linspace(self.range_min, self.range_max, num=NUM_POINTS)
        y = np.linspace(self.range_min, self.range_max, num=NUM_POINTS)
        x, y = np.meshgrid(x, y)
        z = self.eval(x, y)
        plot.surface_2d(x, y, z, self.info)


class Beale(TestFunction):
    '''
        Global Minima:
        f(3, 0.5) = 0
    '''

    def __init__(self, x=None, y=None):
        self.name = 'Beale'
        self.scale = 'log'
        self.range_min = -4.5
        self.range_max = 4.5
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        z1 = np.square(1.5 - x + x * y)
        z2 = np.square(2.25 - x + x * y**2)
        z3 = np.square(2.625 - x + x * y**3)
        return z1 + z2 + z3


class Ackley(TestFunction):
    '''
        Global Minima:
        f(0, 0) = 0
    '''

    def __init__(self, x=None, y=None):
        self.name = 'Ackley'
        self.scale = 'linear'
        self.range_min = -5
        self.range_max = 5
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        z1 = np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        z2 = np.exp(0.5 * np.cos(2 * np.pi * x) + 0.5 * np.cos(2 * np.pi * y))
        return -20 * z1 - z2 + np.e + 20


class Rastrigin(TestFunction):
    '''
        Global Minima:
        f(0, 0) = 0
    '''

    def __init__(self, x=None, y=None):
        self.name = 'Rastrigin'
        self.scale = 'linear'
        self.range_min = -5.12
        self.range_max = 5.12
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        A = 10  # constant
        n = 2   # dimensions
        zn = 0  # init
        for i in range(1, n + 1):
            zn = zn + np.square(x) - A * np.cos(2 * np.pi * x) \
                + np.square(y) - A * np.cos(2 * np.pi * y)
        return A * n + zn


class Himmelblau(TestFunction):
    '''
        Global Minima:
        f(3.0,2.0)             = 0.0
        f(-2.805118,3.131312)  = 0.0
        f(-3.779310,-3.283186) = 0.0
        f(3.584428,-1.848126)  = 0.0
    '''

    def __init__(self, x=None, y=None):
        self.name = 'Himmelblau'
        self.scale = 'log'
        self.range_min = -5
        self.range_max = 5
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        return np.square(x**2 + y - 11) + np.square(x + y**2 - 7)


class Easom(TestFunction):
    '''
        Global Minima:
        f(pi, pi) = -1
    '''

    def __init__(self, x=None, y=None):
        self.name = 'Easom'
        self.scale = 'linear'
        self.range_min = -100
        self.range_max = 100
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        return -np.cos(x) * np.cos(y) * \
            np.exp(-(np.square(x - np.pi) + np.square(y - np.pi)))


class HolderTable(TestFunction):
    '''
        Global Minima:
        f(+/- 8.05502, +/- 9.66459) = -19.2085
    '''

    def __init__(self, x=None, y=None):
        self.name = 'HolderTable'
        self.scale = 'linear'
        self.range_min = -10
        self.range_max = 10
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        z1 = np.sin(x) * np.cos(y)
        z2 = np.exp(np.abs(1 - (np.sqrt(x**2 + y**2)/np.pi)))
        return -np.abs(z1 * z2)


class ShafferNo4(TestFunction):
    '''
        Global Minima:
        f(0, +/- 1.25313) = -0.292579
        f(+/- 1.25313, 0) = -0.292579
    '''

    def __init__(self, x=None, y=None):
        self.name = 'ShafferNo4'
        self.scale = 'linear'
        self.range_min = -100
        self.range_max = 100
        self.info = {'name': self.name,
                     'scale': self.scale}

    def eval(self, x, y):
        z1 = np.square(np.cos(np.sin(np.abs(x**2 - y**2)))) - 0.5
        z2 = np.square(1 + 0.001 * (np.square(x) + np.square(y)))
        return 0.5 + z1/z2
