# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-17 23:07:11

import numpy as np
import matplotlib.pyplot as plt

from ..utils import logger, si_format
from .xymap import XYMap


class DivergenceMap(XYMap):
    ''' Interface to a 2D map showing divergence of the SONIC output from a
        cycle-averaged NICE output, for various combinations of parameters.
    '''

    def __init__(self, benchmark, xvec, yvec, sim_args, eval_mode, eval_args, *args, **kwargs):
        self.benchmark = benchmark
        self.sim_args = sim_args
        self.eval_mode = eval_mode
        self.eval_args = eval_args
        super().__init__(self.benchmark.outdir, xvec, yvec, *args, **kwargs)

    @property
    def eval_mode(self):
        return self._eval_mode

    @eval_mode.setter
    def eval_mode(self, value):
        if value not in self.benchmark.eval_funcs().keys():
            raise ValueError(f'unknown evalation mode: {value}')
        self._eval_mode = value

    @property
    def zkey(self):
        return self.eval_mode

    @property
    def zunit(self):
        return self.benchmark.eval_funcs()[self.eval_mode][1]

    @property
    def zfactor(self):
        if self.eval_mode == 'ss':
            return 1e5
        else:
            return 1e0

    @property
    def suffix(self):
        s = self.eval_mode
        if len(self.eval_args) > 0:
            s = f'{s}_{"_".join([f"{x:.2e}" for x in self.eval_args])}'
        return s

    def descPair(self, x1, x2):
        raise NotImplementedError

    def logDiv(self, x, div):
        ''' Log divergence for a particular inputs combination. '''
        logger.info(f'{self.descPair(*x)}: {self.eval_mode} = {div:.2e} {self.zunit}')

    def compute(self, x):
        data, _ = self.benchmark.getModelAndRunSims(*self.sim_args, *x)
        div = self.benchmark.computeDivergence(data, self.eval_mode, *self.eval_args)
        self.logDiv(x, div)
        return div

    def callbackPltFunc(self):
        raise NotImplementedError

    def onClick(self, event):
        x = self.getOnClickXY(event)
        data, _ = self.benchmark.getModelAndRunSims(*self.sim_args, *x)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlabel('time (ms)')
        ylabel = 'Qm (nC/cm2)'
        if self.eval_mode == 'transient':
            ylabel = 'Qm-norm'
        ax.set_ylabel(ylabel)
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
        ax.set_title(self.descPair(*x))
        self.callbackPltFunc()(ax, data)
        plt.show()

    def render(self, zscale='log', zbounds=(1e-1, 1e1), extend_under=True, extend_over=True,
               cmap='Spectral_r', figsize=(6, 4), fs=12, ax=None, **kwargs):
        ''' Render with default log scale, zbounds, cmap and cbar properties. '''
        fig = super().render(
            zscale=zscale, zbounds=zbounds, extend_under=extend_under, extend_over=extend_over,
            cmap=cmap, figsize=figsize, fs=fs, ax=ax, **kwargs)
        if ax is None:
            fig.canvas.manager.set_window_title(f'{self.corecode()} - {self.eval_mode}')
        return fig


class PassiveDivergenceMap(DivergenceMap):
    ''' Divergence map of a passive model for various combinations of
        membrane time constants (taum) and axial time constant (tauax)
    '''

    xkey = 'taum'
    xfactor = 1e0
    xunit = 's'
    ykey = 'tauax'
    yfactor = 1e0
    yunit = 's'

    @property
    def title(self):
        return f'passive divmap - {self.eval_mode}'

    def corecode(self):
        return f'divmap_{self.benchmark.code()}'

    def descPair(self, taum, tauax):
        return f'taum = {si_format(taum, 2)}s, tauax = {si_format(tauax, 2)}s'

    @staticmethod
    def addPeriodicityLines(ax, T, dims='xy', color='k', pattern='cross'):
        xmin, ymin = 0, 0
        xmax, ymax = 1, 1
        if pattern in ['upper-square', 'lower-square']:
            data_to_axis = ax.transData + ax.transAxes.inverted()
            xc, yc = data_to_axis.transform((T, T))
            if pattern == 'upper-square':
                xmin, ymin = xc, yc
            else:
                xmax, ymax = xc, yc
        if 'x' in dims:
            ax.axvline(T, ymin=ymin, ymax=ymax, color=color, linestyle='--', linewidth=1.5)
        if 'y' in dims:
            ax.axhline(T, xmin=xmin, xmax=xmax, color=color, linestyle='--', linewidth=1.5)

    def render(self, xscale='log', yscale='log', T=None, ax=None, **kwargs):
        ''' Render with drive periodicty indicator. '''
        fig = super().render(xscale=xscale, yscale=yscale, ax=ax, **kwargs)
        if ax is None:
            ax = fig.axes[0]
        if T is not None:
            self.addPeriodicityLines(ax, T)
        return fig

    def callbackPltFunc(self):
        return {
            'ss': self.benchmark.plotQm,
            'transient': self.benchmark.plotQnorm
        }[self.eval_mode]


class FiberDivergenceMap(DivergenceMap):
    ''' Divergence map of a fiber model for various combinations of
        acoustic pressure amplitudes in both compartments
    '''

    xkey = 'A1'
    xfactor = 1e0
    xunit = 'Pa'
    ykey = 'A2'
    yfactor = 1e0
    yunit = 'Pa'

    def __init__(self, benchmark, Avec, *args, **kwargs):
        super().__init__(benchmark, Avec, Avec, *args, **kwargs)

    @property
    def title(self):
        return f'fiber divmap - {self.eval_mode}'

    def corecode(self):
        return f'divmap_{self.benchmark.code()}'

    def descPair(self, *amps):
        return f"A = {', '.join(f'{si_format(A, 2)}Pa' for A in amps)}"

    def compute(self, x):
        if x[0] < x[1]:
            return np.nan
        return super().compute(x)

    def render(self, Ascale='log', **kwargs):
        return super().render(xscale=Ascale, yscale=Ascale, **kwargs)
