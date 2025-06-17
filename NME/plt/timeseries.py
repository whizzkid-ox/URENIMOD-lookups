# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-25 16:18:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-24 17:02:32

import numpy as np
import matplotlib.pyplot as plt

from ..postpro import detectSpikes, convertPeaksProperties
from ..utils import *
from .pltutils import *


class TimeSeriesPlot(GenericPlot):
    ''' Generic interface to build a plot displaying temporal profiles of model simulations. '''

    @classmethod
    def setTimeLabel(cls, ax, tplt, fs):
        return super().setXLabel(ax, tplt, fs)

    @classmethod
    def setYLabel(cls, ax, yplt, fs, grouplabel=None):
        if grouplabel is not None:
            yplt['label'] = grouplabel
        return super().setYLabel(ax, yplt, fs)

    def checkInputs(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def getStimStates(df):
        return df['stimstate'].values

    @classmethod
    def getStimPulses(cls, t, states):
        ''' Determine the onset and offset times of pulses from a stimulation vector.

            :param t: time vector (s).
            :param states: a vector of stimulation state (ON/OFF) at each instant in time.
            :return: list of 3-tuples start time, end time and value of each pulse.
        '''
        # Compute states derivatives and identify transition indexes
        dstates = np.diff(states)
        itransitions = np.where(np.abs(dstates) > 0)[0] + 1
        if states[0] != 0.:
            itransitions = np.hstack(([0], itransitions))
        if states[-1] != 0:
            itransitions = np.hstack((itransitions, [t.size - 1]))
        pulses = list(zip(t[itransitions[:-1]], t[itransitions[1:]], states[itransitions[:-1]]))
        return list(filter(lambda x: x[2] != 0, pulses))

    def addLegend(self, fig, ax, handles, labels, fs, color=None, ls=None):
        lh = ax.legend(handles, labels, loc=1, fontsize=fs, frameon=False)
        if color is not None:
            for l in lh.get_lines():
                l.set_color(color)
        if ls:
            for l in lh.get_lines():
                l.set_linestyle(ls)

    @classmethod
    def materializeSpikes(cls, ax, data, tplt, yplt, color, mode, add_to_legend=False):
        ispikes, properties = detectSpikes(data)
        t = data['t'].values
        Qm = data['Qm'].values
        yfactor = yplt.get('factor', 1.0)
        if ispikes is not None:
            yoffset = 5
            ax.plot(t[ispikes] * tplt['factor'], Qm[ispikes] * yfactor + yoffset,
                    'v', color=color, label='spikes' if add_to_legend else None)
            if mode == 'details':
                ileft = properties['left_bases']
                iright = properties['right_bases']
                properties = convertPeaksProperties(t, properties)
                ax.plot(t[ileft] * tplt['factor'], Qm[ileft] * yfactor - 5,
                        '<', color=color, label='left-bases' if add_to_legend else None)
                ax.plot(t[iright] * tplt['factor'], Qm[iright] * yfactor - 10,
                        '>', color=color, label='right-bases' if add_to_legend else None)
                ax.vlines(
                    x=t[ispikes] * tplt['factor'],
                    ymin=(Qm[ispikes] - properties['prominences']) * yfactor,
                    ymax=Qm[ispikes] * yfactor,
                    color=color, linestyles='dashed',
                    label='prominences' if add_to_legend else '')
                ax.hlines(
                    y=properties['width_heights'] * yfactor,
                    xmin=properties['left_ips'] * tplt['factor'],
                    xmax=properties['right_ips'] * tplt['factor'],
                    color=color, linestyles='dotted', label='half-widths' if add_to_legend else '')
        return add_to_legend

    @staticmethod
    def prepareTime(t, tplt):
        if tplt['onset'] > 0.0:
            tonset = t.min() - 0.05 * np.ptp(t)
            t = np.insert(t, 0, tonset)
        return t * tplt['factor']

    @staticmethod
    def getPatchesColors(x):
        if np.all([xx == x[0] for xx in x]):
            return ['#8A8A8A'] * len(x)
        else:
            xabsmax = np.abs(x).max()
            _, sm = setNormalizer(plt.get_cmap('RdGy'), (-xabsmax, xabsmax), 'lin')
            return [sm.to_rgba(xx) for xx in x]

    @classmethod
    def addPatches(cls, ax, pulses, tplt, color=None):
        tstart, tend, x = zip(*pulses)
        if color is None:
            colors = cls.getPatchesColors(x)
        else:
            colors = [color] * len(x)
        for i in range(len(pulses)):
            ax.axvspan(tstart[i] * tplt['factor'], tend[i] * tplt['factor'],
                       edgecolor='none', facecolor=colors[i], alpha=0.2)

    @staticmethod
    def plotInset(inset_ax, inset, t, y, tplt, yplt, line, color, lw):
        inset_ax.plot(t, y, linewidth=lw, linestyle=line, color=color)
        return inset_ax

    @classmethod
    def addInsetPatches(cls, ax, inset_ax, inset, pulses, tplt, color):
        tstart, tend, x = [np.array([z]) for z in zip(*pulses)]
        tfactor = tplt['factor']
        ybottom, ytop = ax.get_ylim()
        cond_start = np.logical_and(tstart > (inset['xlims'][0] / tfactor),
                                    tstart < (inset['xlims'][1] / tfactor))
        cond_end = np.logical_and(tend > (inset['xlims'][0] / tfactor),
                                  tend < (inset['xlims'][1] / tfactor))
        cond_glob = np.logical_and(tstart < (inset['xlims'][0] / tfactor),
                                   tend > (inset['xlims'][1] / tfactor))
        cond_onoff = np.logical_or(cond_start, cond_end)
        cond = np.logical_or(cond_onoff, cond_glob)
        tstart, tend, x = [z[cond] for z in [tstart, tend, x]]
        colors = cls.getPatchesColors(x)
        npatches_inset = tstart.size
        for i in range(npatches_inset):
            inset_ax.add_patch(Rectangle(
                (tstart[i] * tfactor, ybottom),
                (tend[i] - tstart[i]) * tfactor, ytop - ybottom,
                color=colors[i], alpha=0.1))


class CompTimeSeries(ComparativePlot, TimeSeriesPlot):
    ''' Interface to build a comparative plot displaying profiles of a specific output variable
        across different model simulations. '''

    def __init__(self, outputs, varname):
        ''' Constructor.

            :param outputs: list / generator of simulator outputs to be compared.
            :param varname: name of variable to extract and compare
        '''
        ComparativePlot.__init__(self, outputs, varname)

    def checkPatches(self, patches):
        self.greypatch = False
        if patches == 'none':
            self.patchfunc = lambda _: False
        elif patches == 'all':
            self.patchfunc = lambda _: True
        elif patches == 'one':
            self.patchfunc = lambda j: True if j == 0 else False
            self.greypatch = True
        elif isinstance(patches, list):
            if not all(isinstance(p, bool) for p in patches):
                raise TypeError('Invalid patch sequence: all list items must be boolean typed')
            self.patchfunc = lambda j: patches[j] if len(patches) > j else False
        else:
            raise ValueError(
                'Invalid patches: must be either "none", all", "one", or a boolean list')

    def checkInputs(self, labels, patches):
        self.checkLabels(labels)
        self.checkPatches(patches)

    @staticmethod
    def createBackBone(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_zorder(0)
        return fig, ax

    @classmethod
    def postProcess(cls, ax, tplt, yplt, fs, meta, prettify):
        cls.removeSpines(ax)
        if 'bounds' in yplt:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, yplt['bounds'][0]), max(ymax, yplt['bounds'][1]))
        elif 'strictbounds' in yplt:
            ax.set_ylim(*yplt['strictbounds'])
        cls.setTimeLabel(ax, tplt, fs)
        cls.setYLabel(ax, yplt, fs)
        if prettify:
            cls.prettify(ax, xticks=(0, meta['tstim'] * tplt['factor']))
        cls.setTickLabelsFontSize(ax, fs)

    def render(self, figsize=(11, 4), fs=10, lw=2, labels=None, colors=None, lines=None,
               patches='one', inset=None, frequency=1, spikes='none', cmap=None,
               cscale='lin', trange=None, prettify=False):
        ''' Render plot.

            :param figsize: figure size (x, y)
            :param fs: labels fontsize
            :param lw: linewidth
            :param labels: list of labels to use in the legend
            :param colors: list of colors to use for each curve
            :param lines: list of linestyles
            :param patches: string indicating whether/how to mark stimulation periods
                with rectangular patches
            :param inset: string indicating whether/how to mark an inset zooming on
                a particular region of the graph
            :param frequency: frequency at which to plot samples
            :param spikes: string indicating how to show spikes ("none", "marks" or "details")
            :param cmap: color map to use for colobar-based comparison (if not None)
            :param cscale: color scale to use for colobar-based comparison
            :param trange: optional lower and upper bounds to time axis
            :return: figure handle
        '''
        self.checkInputs(labels, patches)
        fcodes = []

        fig, ax = self.createBackBone(figsize)
        if inset is not None:
            inset_ax = self.addInset(fig, ax, inset)

        # Loop through data files
        handles, comp_values, full_labels = [], [], []
        tmin, tmax = np.inf, -np.inf
        for j, output in enumerate(self.outputs):
            color = f'C{j}' if colors is None else colors[j]
            line = '-' if lines is None else lines[j]
            patch = self.patchfunc(j)

            # Load data
            try:
                data, meta = self.getData(output, frequency, trange)
            except ValueError:
                continue
            if 'tcomp' in meta:
                meta.pop('tcomp')

            # Extract model
            model = self.getModel(meta)
            fcodes.append(model.filecode(meta))

            # Add label to list
            full_labels.append(self.figtitle(model, meta))

            # Check consistency of sim types and check differing inputs
            comp_values = self.checkConsistency(meta, comp_values)

            # Extract time and stim pulses
            t = data['t'].values
            stimstate = self.getStimStates(data)
            pulses = self.getStimPulses(t, stimstate)
            tplt = self.getTimePltVar(model.tscale)
            t = self.prepareTime(t, tplt)

            # Extract y-variable
            pltvars = model.getPltVars()
            if self.varname not in pltvars:
                pltvars_str = ', '.join([f'"{p}"' for p in pltvars.keys()])
                raise KeyError(
                    f'Unknown plot variable: "{self.varname}". Candidates are: {pltvars_str}')
            yplt = pltvars[self.varname]
            y = extractPltVar(model, yplt, data, meta, t.size, self.varname)

            #  Plot time series
            handles.append(ax.plot(t, y, linewidth=lw, linestyle=line, color=color)[0])

            # Optional: add spikes
            if self.varname == 'Qm' and spikes != 'none':
                self.materializeSpikes(ax, data, tplt, yplt, color, spikes)

            # Plot optional inset
            if inset is not None:
                inset_ax = self.plotInset(
                    inset_ax, inset, t, y, tplt, yplt, lines[j], color, lw)

            # Add optional STIM-ON patches
            if patch:
                ybottom, ytop = ax.get_ylim()
                patchcolor = None if self.greypatch else handles[j].get_color()
                self.addPatches(ax, pulses, tplt, color=patchcolor)
                if inset is not None:
                    self.addInsetPatches(ax, inset_ax, inset, pulses, tplt, patchcolor)

            tmin, tmax = min(tmin, t.min()), max(tmax, t.max())

        # Get common label and add it as title
        common_label = self.getCommonLabel(full_labels.copy(), seps=':@,()')
        self.wraptitle(ax, common_label, fs=fs)

        # Get comp info if any
        if self.comp_ref_key is not None:
            self.comp_info = model.inputs().get(self.comp_ref_key, None)

        # Post-process figure
        self.postProcess(ax, tplt, yplt, fs, meta, prettify)
        ax.set_xlim(tmin, tmax)
        fig.tight_layout()

        # Materialize inset if any
        if inset is not None:
            self.materializeInset(ax, inset_ax, inset)

        # Add labels or colorbar legend
        if cmap is not None:
            if not self.is_unique_comp:
                raise ValueError('Colormap mode unavailable for multiple differing parameters')
            if self.comp_info is None:
                raise ValueError('Colormap mode unavailable for qualitative comparisons')
            self.addCmap(
                fig, cmap, handles, comp_values, self.comp_info, fs, prettify, zscale=cscale)
        else:
            comp_values, comp_labels = self.getCompLabels(comp_values)
            labels = self.chooseLabels(labels, comp_labels, full_labels)
            self.addLegend(fig, ax, handles, labels, fs)

        # Add window title based on common pattern
        common_fcode = self.getCommonLabel(fcodes.copy())
        fig.canvas.manager.set_window_title(common_fcode)

        return fig


class GroupedTimeSeries(TimeSeriesPlot):
    ''' Interface to build a plot displaying profiles of several output variables
        arranged into specific schemes. '''

    def __init__(self, outputs, pltscheme=None):
        ''' Constructor.

            :param outputs: list / generator of simulation outputs.
            :param varname: name of variable to extract and compare
        '''
        super().__init__(outputs)
        self.pltscheme = pltscheme

    @staticmethod
    def createBackBone(pltscheme):
        naxes = len(pltscheme)
        if naxes == 1:
            fig, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))
        return fig, axes

    @staticmethod
    def shareX(axes):
        for ax in axes[:-1]:
            ax.get_shared_x_axes().join(ax, axes[-1])
            ax.set_xticklabels([])

    @classmethod
    def postProcess(cls, axes, tplt, fs, meta, prettify):
        for ax in axes:
            cls.removeSpines(ax)
            if prettify:
                cls.prettify(ax, xticks=(0, meta['pp'].tstim * tplt['factor']), yfmt=None)
            cls.setTickLabelsFontSize(ax, fs)
        cls.shareX(axes)
        cls.setTimeLabel(axes[-1], tplt, fs)

    def render(self, fs=10, lw=2, labels=None, colors=None, lines=None, patches='one', save=False,
               outputdir=None, fig_ext='png', frequency=1, spikes='none', trange=None,
               prettify=False):
        ''' Render plot.

            :param fs: labels fontsize
            :param lw: linewidth
            :param labels: list of labels to use in the legend
            :param colors: list of colors to use for each curve
            :param lines: list of linestyles
            :param patches: boolean indicating whether to mark stimulation periods
                with rectangular patches
            :param save: boolean indicating whether or not to save the figure(s)
            :param outputdir: path to output directory in which to save figure(s)
            :param fig_ext: string indcating figure extension ("png", "pdf", ...)
            :param frequency: frequency at which to plot samples
            :param spikes: string indicating how to show spikes ("none", "marks" or "details")
            :param trange: optional lower and upper bounds to time axis
            :return: figure handle(s)
        '''
        if colors is None:
            colors = plt.get_cmap('tab10').colors

        figs = []
        for output in self.outputs:

            # Load data and extract model
            try:
                data, meta = self.getData(output, frequency, trange)
            except ValueError:
                continue
            model = self.getModel(meta)

            # Extract time and stim pulses
            t = data['t'].values
            stimstate = self.getStimStates(data)

            pulses = self.getStimPulses(t, stimstate)
            tplt = self.getTimePltVar(model.tscale)
            t = self.prepareTime(t, tplt)

            # Check plot scheme if provided, otherwise generate it
            pltvars = model.getPltVars()
            if self.pltscheme is not None:
                for key in list(sum(list(self.pltscheme.values()), [])):
                    if key not in pltvars:
                        raise KeyError(f'Unknown plot variable: "{key}"')
                pltscheme = self.pltscheme
            else:
                pltscheme = model.pltScheme

            # Create figure
            fig, axes = self.createBackBone(pltscheme)

            # Loop through each subgraph
            for ax, (grouplabel, keys) in zip(axes, pltscheme.items()):
                ax_legend_spikes = False

                # Extract variables to plot
                nvars = len(keys)
                ax_pltvars = [pltvars[k] for k in keys]
                if nvars == 1:
                    ax_pltvars[0]['color'] = 'k'
                    ax_pltvars[0]['ls'] = '-'

                # Plot time series
                icolor = 0
                for yplt, name in zip(ax_pltvars, pltscheme[grouplabel]):
                    color = yplt.get('color', colors[icolor])
                    y = extractPltVar(model, yplt, data, meta, t.size, name)
                    ax.plot(t, y, yplt.get('ls', '-'), c=color, lw=lw,
                            label='$\\rm {}$'.format(yplt["label"]))
                    if 'color' not in yplt:
                        icolor += 1

                    # Optional: add spikes
                    if name == 'Qm' and spikes != 'none':
                        ax_legend_spikes = self.materializeSpikes(
                            ax, data, tplt, yplt, color, spikes, add_to_legend=True)

                # Set y-axis unit and bounds
                self.setYLabel(ax, ax_pltvars[0].copy(), fs, grouplabel=grouplabel)
                if 'bounds' in ax_pltvars[0]:
                    ymin, ymax = ax.get_ylim()
                    ax_min = min(ymin, *[ap['bounds'][0] for ap in ax_pltvars])
                    ax_max = max(ymax, *[ap['bounds'][1] for ap in ax_pltvars])
                    ax.set_ylim(ax_min, ax_max)

                # Add legend
                if nvars > 1 or 'gate' in ax_pltvars[0]['desc'] or ax_legend_spikes:
                    ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1, frameon=False)

            # Set x-limits and add optional patches
            for ax in axes:
                ax.set_xlim(t.min(), t.max())
                if patches != 'none':
                    self.addPatches(ax, pulses, tplt)

            # Post-process figure
            self.postProcess(axes, tplt, fs, meta, prettify)
            self.wraptitle(axes[0], self.figtitle(model, meta), fs=fs)
            fig.tight_layout()

            fig.canvas.manager.set_window_title(model.filecode(meta))

            # Save figure if needed (automatic or checked)
            if save:
                filecode = model.filecode(meta)
                if outputdir is None:
                    raise ValueError('output directory not specified')
                plt_filename = f'{outputdir}/{filecode}.{fig_ext}'
                plt.savefig(plt_filename)
                logger.info(f'Saving figure as "{plt_filename}"')
                plt.close()

            figs.append(fig)
        return figs


if __name__ == '__main__':
    # example of use
    filepaths = OpenFilesDialog('pkl')[0]
    comp_plot = CompTimeSeries(filepaths, 'Qm')
    fig = comp_plot.render(
        lines=['-', '--'],
        labels=['60 kPa', '80 kPa'],
        patches='one',
        colors=['r', 'g'],
        xticks=[0, 100],
        yticks=[-80, +50],
        inset={'xcoords': [5, 40], 'ycoords': [-35, 45], 'xlims': [57.5, 60.5], 'ylims': [10, 35]}
    )

    scheme_plot = GroupedTimeSeries(filepaths)
    figs = scheme_plot.render()

    plt.show()
