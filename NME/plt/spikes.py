# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-10-01 20:40:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-03 20:09:06

import numpy as np
import matplotlib.pyplot as plt

from ..core import getModel
from ..utils import *
from ..constants import *
from .pltutils import *
from ..postpro import detectSpikes, convertPeaksProperties


class SpikesDiagram(ComparativePlot):

    phaseplotvars = {
        'Vm': {
            'label': 'V_m\ (mV)',
            'dlabel': 'dV/dt\ (V/s)',
            'factor': 1e0,
            'lim': (-80.0, 50.0),
            'dfactor': 1e-3,
            'dlim': (-300, 700),
            'thr_amp': SPIKE_MIN_VAMP,
            'thr_prom': SPIKE_MIN_VPROM
        },
        'Qm': {
            'label': 'Q_m\ (nC/cm^2)',
            'dlabel': 'I\ (A/m^2)',
            'factor': 1e5,
            'lim': (-80.0, 50.0),
            'dfactor': 1e0,
            'dlim': (-2, 5),
            'thr_amp': SPIKE_MIN_QAMP,
            'thr_prom': SPIKE_MIN_QPROM
        }
    }

    @classmethod
    def createBackBone(cls, pltvar, tbounds, fs, prettify):
                # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # 1st axis: variable as function of time
        ax = axes[0]
        ax.set_xlabel('$\\rm time\ (ms)$', fontsize=fs)
        ax.set_ylabel('$\\rm {}$'.format(pltvar["label"]), fontsize=fs)
        ax.set_xlim(tbounds)
        ax.set_ylim(pltvar['lim'])

        # 2nd axis: phase plot (derivative of variable vs variable)
        ax = axes[1]
        ax.set_xlabel('$\\rm {}$'.format(pltvar["label"]), fontsize=fs)
        ax.set_ylabel('$\\rm {}$'.format(pltvar["dlabel"]), fontsize=fs)
        ax.set_xlim(pltvar['lim'])
        ax.set_ylim(pltvar['dlim'])
        ax.plot([0, 0], [pltvar['dlim'][0], pltvar['dlim'][1]], '--', color='k', linewidth=1)
        ax.plot([pltvar['lim'][0], pltvar['lim'][1]], [0, 0], '--', color='k', linewidth=1)

        if prettify:
            cls.prettify(axes[0], xticks=tbounds, yticks=pltvar['lim'])
            cls.prettify(axes[1], xticks=pltvar['lim'], yticks=pltvar['dlim'])
        for ax in axes:
            cls.removeSpines(ax)
            cls.setTickLabelsFontSize(ax, fs)

        return fig, axes

    def checkInputs(self, labels):
        self.checkLabels(labels)

    @staticmethod
    def extractSpikesData(t, y, tbounds, rel_tbounds, tspikes):
        spikes_tvec, spikes_yvec, spikes_dydtvec = [], [], []
        for j, (tspike, tbound) in enumerate(zip(tspikes, tbounds)):
            left_bound = max(tbound[0], rel_tbounds[0] + tspike)
            right_bound = min(tbound[1], rel_tbounds[1] + tspike)
            inds = np.where((t > left_bound) & (t < right_bound))[0]
            spikes_tvec.append(t[inds] - tspike)
            spikes_yvec.append(y[inds])
            dinds = np.hstack(([inds[0] - 1], inds, [inds[-1] + 1]))
            dydt = np.diff(y[dinds]) / np.diff(t[dinds])
            spikes_dydtvec.append((dydt[:-1] + dydt[1:]) / 2)  # average of the two
        return spikes_tvec, spikes_yvec, spikes_dydtvec

    def addLegend(self, fig, axes, handles, labels, fs):
        fig.subplots_adjust(top=0.8)
        if len(self.filepaths) > 1:
            axes[0].legend(handles, labels, fontsize=fs, frameon=False,
                           loc='upper center', bbox_to_anchor=(1.0, 1.35))
        else:
            fig.suptitle(labels[0], fontsize=fs)

    def render(self, no_offset=False, no_first=False, labels=None, colors=None,
               fs=10, lw=2, trange=None, rel_tbounds=None, prettify=False,
               cmap=None, cscale='lin'):

        self.checkInputs(labels)

        if rel_tbounds is None:
            rel_tbounds = np.array((-1.5e-3, 1.5e-3))

        # Check pltvar
        if self.varname not in self.phaseplotvars:
            pltvars_str = ', '.join([f'"{p}"' for p in self.phaseplotvars.keys()])
            raise KeyError(
                f'Unknown plot variable: "{self.varname}". Possible plot variables are: {pltvars_str}')
        pltvar = self.phaseplotvars[self.varname]

        fig, axes = self.createBackBone(pltvar, rel_tbounds * 1e3, fs, prettify)

        # Loop through data files
        comp_values, full_labels = [], []
        handles0, handles1 = [], []
        for i, filepath in enumerate(self.filepaths):

            # Load data
            data, meta = self.getData(filepath, trange=trange)
            meta.pop('tcomp')

            # Extract model
            model = getModel(meta)
            full_labels.append(self.figtitle(model, meta))

            # Check consistency of sim types and check differing inputs
            comp_values = self.checkConsistency(meta, comp_values)

            # Extract time and y-variable
            t = data['t'].values
            y = data[self.varname].values

            # Detect spikes in signal
            ispikes, properties = detectSpikes(
                data, key=self.varname, mph=pltvar['thr_amp'], mpp=pltvar['thr_prom'])
            nspikes = ispikes.size
            tspikes = t[ispikes]
            yspikes = y[ispikes]
            properties = convertPeaksProperties(t, properties)
            tbounds = np.array(list(zip(properties['left_bases'], properties['right_bases'])))

            if nspikes == 0:
                logger.warning('No spikes detected')
            else:
                # Store spikes in dedicated lists
                spikes_tvec, spikes_yvec, spikes_dydtvec = self.extractSpikesData(
                    t, y, tbounds, rel_tbounds, tspikes)

                # Plot spikes temporal profiles and phase-plane diagrams
                lh0, lh1 = [], []
                for j in range(nspikes):
                    if colors is None:
                        icolor = i if len(self.filepaths) > 1 else j % 10
                        color = f'C{icolor}'
                    else:
                        color = colors[i]
                    lh0.append(axes[0].plot(
                        spikes_tvec[j] * 1e3, spikes_yvec[j] * pltvar['factor'],
                        linewidth=lw, c=color)[0])
                    lh1.append(axes[1].plot(
                        spikes_yvec[j] * pltvar['factor'], spikes_dydtvec[j] * pltvar['dfactor'],
                        linewidth=lw, c=color)[0])

                handles0.append(lh0)
                handles1.append(lh1)

        # Determine labels
        if self.comp_ref_key is not None:
            self.comp_info = model.inputs().get(self.comp_ref_key, None)
        comp_values, comp_labels = self.getCompLabels(comp_values)
        labels = self.chooseLabels(labels, comp_labels, full_labels)

        # Post-process figure
        fig.tight_layout()

        # Add labels or colorbar legend
        if cmap is not None:
            if not self.is_unique_comp:
                raise ValueError('Colormap mode unavailable for multiple differing parameters')
            if self.comp_info is None:
                raise ValueError('Colormap mode unavailable for qualitative comparisons')
            cmap_handles = [h0 + h1 for h0, h1 in zip(handles0, handles1)]
            self.addCmap(
                fig, cmap, cmap_handles, comp_values, self.comp_info, fs, prettify, zscale=cscale)
        else:
            leg_handles = [x[0] for x in handles0]
            self.addLegend(fig, axes, leg_handles, labels, fs)

        return fig
