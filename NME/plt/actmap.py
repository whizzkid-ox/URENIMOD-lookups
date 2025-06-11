# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:24:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-22 15:47:32

import abc
import numpy as np
import matplotlib.pyplot as plt

from ..core import NeuronalBilayerSonophore, PulsedProtocol, AcousticDrive, Batch
from ..utils import logger, si_format
from .xymap import XYMap
from .timeseries import GroupedTimeSeries
from ..postpro import detectSpikes


class ActivationMap(XYMap):

    xkey = 'Duty cycle'
    xfactor = 1e2
    xunit = '%'
    ykey = 'Amplitude'
    yfactor = 1e-3
    yunit = 'kPa'
    onclick_colors = None

    def __init__(self, root, pneuron, a, fs, f, tstim, PRF, amps, DCs):
        self.nbls = NeuronalBilayerSonophore(a, pneuron)
        self.drive = AcousticDrive(f, None)
        self.pp = PulsedProtocol(tstim, 0., PRF, .5)
        self.fs = fs
        super().__init__(root, DCs * self.xfactor, amps * self.yfactor)

    @property
    def sim_args(self):
        return [self.drive, self.pp, self.fs, 'sonic', None]

    @property
    def title(self):
        s = 'Activation map - {} neuron @ {}Hz, {}Hz PRF ({}m sonophore'.format(
            self.nbls.pneuron.name, *si_format([self.drive.f, self.pp.PRF, self.nbls.a]))
        if self.fs < 1:
            s = f'{s}, {self.fs * 1e2:.0f}% coverage'
        return f'{s})'

    def corecode(self):
        corecodes = self.nbls.filecodes(*self.sim_args)
        del corecodes['nature']
        if 'DC' in corecodes:
            del corecodes['DC']
        return '_'.join(filter(lambda x: x is not None, corecodes.values()))

    def compute(self, x):
        ''' Compute firing rate from simulation output '''
        # Adapt drive and pulsed protocol
        self.pp.DC = x[0] / self.xfactor
        self.drive.A = x[1] / self.yfactor

        # Get model output, running simulation if needed
        data, _ = self.nbls.getOutput(*self.sim_args, outputdir=self.root)
        return self.xfunc(data)

    @abc.abstractmethod
    def xfunc(self, data):
        raise NotImplementedError

    def addThresholdCurve(self, ax, fs, mpi=False):
        queue = [[
            self.drive,
            PulsedProtocol(self.pp.tstim, self.pp.toffset, self.pp.PRF, DC / self.xfactor),
            self.fs, 'sonic', None] for DC in self.xvec]
        batch = Batch(self.nbls.titrate, queue)
        Athrs = np.array(batch.run(mpi=mpi, loglevel=logger.level))
        ax.plot(self.xvec, Athrs * self.yfactor, '-', color='#F26522', linewidth=3,
                label='threshold amplitudes')
        ax.legend(loc='lower center', frameon=False, fontsize=fs)

    @property
    @abc.abstractmethod
    def onclick_pltscheme(self):
        raise NotImplementedError

    def onClick(self, event):
        ''' Execute action when the user clicks on a cell in the 2D map. '''
        DC, A = self.getOnClickXY(event)
        self.plotTimeseries(DC, A)
        plt.show()

    def plotTimeseries(self, DC, A, **kwargs):
        ''' Plot related timeseries for a given duty cycle and amplitude. '''
        self.drive.A = A / self.yfactor
        self.pp.DC = DC / self.xfactor

        # Get model output, running simulation if needed
        data, meta = self.nbls.getOutput(*self.sim_args, outputdir=self.root)

        # Plot timeseries of appropriate variables
        timeseries = GroupedTimeSeries([(data, meta)], pltscheme=self.onclick_pltscheme)
        return timeseries.render(colors=self.onclick_colors, **kwargs)[0]

    def render(self, yscale='log', thresholds=False, mpi=False, **kwargs):
        fig = super().render(yscale=yscale, **kwargs)
        if thresholds:
            self.addThresholdCurve(fig.axes[0], fs=12, mpi=mpi)
        return fig


class FiringRateMap(ActivationMap):

    zkey = 'Firing rate'
    zunit = 'Hz'
    zfactor = 1e0
    suffix = 'FRmap'
    onclick_pltscheme = {'V_m\ |\ Q_/C_{m0}': ['Vm', 'Qm/Cm0']}
    onclick_colors = ['darkgrey', 'k']

    def xfunc(self, data):
        ''' Detect spikes in data and compute firing rate. '''
        ispikes, _ = detectSpikes(data)
        if ispikes.size > 1:
            t = data['t'].values
            sr = 1 / np.diff(t[ispikes])
            return np.mean(sr)
        else:
            return np.nan

    def render(self, zscale='log', **kwargs):
        return super().render(zscale=zscale, **kwargs)


class CalciumMap(ActivationMap):

    zkey = '[Ca2+]i'
    zunit = 'uM'
    zfactor = 1e6
    suffix = 'Camap'
    onclick_pltscheme = {'Cai': ['Cai']}

    def xfunc(self, data):
        ''' Detect spikes in data and compute firing rate. '''
        Cai = data['Cai'].values * self.zfactor  # uM
        return np.mean(Cai)

    def render(self, zscale='log', **kwargs):
        return super().render(zscale=zscale, **kwargs)


map_classes = {
    'FR': FiringRateMap,
    'Cai': CalciumMap
}


def getActivationMap(key, *args, **kwargs):
    if key not in map_classes:
        raise ValueError(f'{key} is not a valid map type')
    return map_classes[key](*args, **kwargs)
