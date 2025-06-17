# Author: Ryo Segawa
# Inspired by work of: https://github.com/tjjlemaire/PySONIC

''' Useful functions to generate plots. '''

import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Polygon, Rectangle
from matplotlib import cm, colors
import matplotlib.pyplot as plt

from ..core import getModel
from ..utils import *

# Matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def getSymmetricCmap(cmap_key):
    cmap = plt.get_cmap(cmap_key)
    cl = np.vstack((cmap.colors, cmap.reversed().colors))
    return colors.LinearSegmentedColormap.from_list(f'sym_{cmap_key}', cl)


for k in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
    for cmap_key in [k, f'{k}_r']:
        sym_cmap = getSymmetricCmap(cmap_key)
        #* Attempt to register colormap, compatible with different Matplotlib versions
        try:
            # Modern Matplotlib (2.0+)
            plt.register_cmap(name=sym_cmap.name, cmap=sym_cmap)
        except AttributeError:
            try:
                # Matplotlib 1.5.x
                cm.register_cmap(name=sym_cmap.name, cmap=sym_cmap)
            except AttributeError:
                # Matplotlib < 1.5 (direct dict manipulation)
                if hasattr(cm, 'cmap_d') and hasattr(cm, '_cmapnames'):
                    cm.cmap_d[sym_cmap.name] = sym_cmap
                    if sym_cmap.name not in cm._cmapnames:
                        cm._cmapnames.append(sym_cmap.name)
                else:
                    # If all known methods fail, use the logger (imported from ..utils)
                    logger.warning(
                        f"Could not register colormap '{sym_cmap.name}'. "
                        "Matplotlib version might be too old or have an unexpected API."
                    )


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def extractPltVar(model, pltvar, df, meta=None, nsamples=0, name=''):
    if 'func' in pltvar:
        s = pltvar['func']
        if not s.startswith('meta'):
            s = f'model.{s}'
        try:
            var = eval(s)
        except AttributeError as err:
            if hasattr(model, 'pneuron'):
                var = eval(s.replace('model', 'model.pneuron'))
            else:
                raise err
    elif 'key' in pltvar:
        var = df[pltvar['key']]
    elif 'constant' in pltvar:
        var = eval(pltvar['constant']) * np.ones(nsamples)
    else:
        var = df[name]
    if isinstance(var, pd.Series):
        var = var.values
    var = var.copy()

    if var.size == nsamples - 1:
        var = np.insert(var, 0, var[0])
    var *= pltvar.get('factor', 1)

    return var


def setGrid(n, ncolmax=3):
    ''' Determine number of rows and columns in figure grid, based on number of
        variables to plot. '''
    if n <= ncolmax:
        return (1, n)
    else:
        return ((n - 1) // ncolmax + 1, ncolmax)


def setNormalizer(cmap, bounds, scale='lin'):
    norm = {
        'lin': colors.Normalize,
        'log': colors.LogNorm,
        'symlog': colors.SymLogNorm
    }[scale](*bounds)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    return norm, sm


class GenericPlot:
    def __init__(self, outputs):
        ''' Constructor.

            :param outputs: list / generator of simulation outputs
        '''
        try:
            iter(outputs)
        except TypeError:
            outputs = [outputs]
        self.outputs = outputs

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def figtitle(self, model, meta):
        return model.desc(meta)

    @staticmethod
    def wraptitle(ax, title, maxwidth=120, sep=':', fs=10, y=1.0):
        if len(title) > maxwidth:
            title = '\n'.join(title.split(sep))
            y = 0.94
        h = ax.set_title(title, fontsize=fs)
        h.set_y(y)

    @staticmethod
    def getData(entry, frequency=1, trange=None):
        if entry is None:
            raise ValueError('non-existing data')
        if isinstance(entry, str):
            data, meta = loadData(entry, frequency)
        else:
            data, meta = entry
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def render(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def getSimType(fname):
        ''' Get sim type from filename. '''
        mo = re.search('(^[A-Z]*)_(.*).pkl', fname)
        if not mo:
            raise ValueError(f'Could not find sim-key in filename: "{fname}"')
        return mo.group(1)

    @staticmethod
    def getModel(*args, **kwargs):
        return getModel(*args, **kwargs)

    @staticmethod
    def getTimePltVar(tscale):
        ''' Return time plot variable for a given temporal scale. '''
        return {
            'desc': 'time',
            'label': 'time',
            'unit': tscale,
            'factor': {'ms': 1e3, 'us': 1e6}[tscale],
            'onset': {'ms': 1e-3, 'us': 1e-6}[tscale]
        }

    @staticmethod
    def createBackBone(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def prettify(ax, xticks=None, yticks=None, xfmt='{:.0f}', yfmt='{:+.0f}'):
        try:
            ticks = ax.get_ticks()
            ticks = (min(ticks), max(ticks))
            ax.set_ticks(ticks)
            ax.set_ticklabels([xfmt.format(x) for x in ticks])
        except AttributeError:
            if xticks is None:
                xticks = ax.get_xticks()
                xticks = (min(xticks), max(xticks))
            if yticks is None:
                yticks = ax.get_yticks()
                yticks = (min(yticks), max(yticks))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if xfmt is not None:
                ax.set_xticklabels([xfmt.format(x) for x in xticks])
            if yfmt is not None:
                ax.set_yticklabels([yfmt.format(y) for y in yticks])

    @staticmethod
    def addInset(fig, ax, inset):
        ''' Create inset axis. '''
        inset_ax = fig.add_axes(ax.get_position())
        inset_ax.set_zorder(1)
        inset_ax.set_xlim(inset['xlims'][0], inset['xlims'][1])
        inset_ax.set_ylim(inset['ylims'][0], inset['ylims'][1])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.add_patch(Rectangle((inset['xlims'][0], inset['ylims'][0]),
                                     inset['xlims'][1] - inset['xlims'][0],
                                     inset['ylims'][1] - inset['ylims'][0],
                                     color='w'))
        return inset_ax

    @staticmethod
    def materializeInset(ax, inset_ax, inset):
        ''' Materialize inset with zoom boox. '''
        # Re-position inset axis
        axpos = ax.get_position()
        left, right, = rescale(inset['xcoords'], ax.get_xlim()[0], ax.get_xlim()[1],
                               axpos.x0, axpos.x0 + axpos.width)
        bottom, top, = rescale(inset['ycoords'], ax.get_ylim()[0], ax.get_ylim()[1],
                               axpos.y0, axpos.y0 + axpos.height)
        inset_ax.set_position([left, bottom, right - left, top - bottom])
        for i in inset_ax.spines.values():
            i.set_linewidth(2)

        # Materialize inset target region with contour frame
        ax.plot(inset['xlims'], [inset['ylims'][0]] * 2, linestyle='-', color='k')
        ax.plot(inset['xlims'], [inset['ylims'][1]] * 2, linestyle='-', color='k')
        ax.plot([inset['xlims'][0]] * 2, inset['ylims'], linestyle='-', color='k')
        ax.plot([inset['xlims'][1]] * 2, inset['ylims'], linestyle='-', color='k')

        # Link target and inset with dashed lines if possible
        if inset['xcoords'][1] < inset['xlims'][0]:
            ax.plot([inset['xcoords'][1], inset['xlims'][0]],
                    [inset['ycoords'][0], inset['ylims'][0]],
                    linestyle='--', color='k')
            ax.plot([inset['xcoords'][1], inset['xlims'][0]],
                    [inset['ycoords'][1], inset['ylims'][1]],
                    linestyle='--', color='k')
        elif inset['xcoords'][0] > inset['xlims'][1]:
            ax.plot([inset['xcoords'][0], inset['xlims'][1]],
                    [inset['ycoords'][0], inset['ylims'][0]],
                    linestyle='--', color='k')
            ax.plot([inset['xcoords'][0], inset['xlims'][1]],
                    [inset['ycoords'][1], inset['ylims'][1]],
                    linestyle='--', color='k')
        else:
            logger.warning('Inset x-coordinates intersect with those of target region')

    def postProcess(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def removeSpines(ax):
        for item in ['top', 'right']:
            ax.spines[item].set_visible(False)

    @staticmethod
    def setXTicks(ax, xticks=None):
        if xticks is not None:
            ax.set_xticks(xticks)

    @staticmethod
    def setYTicks(ax, yticks=None):
        if yticks is not None:
            ax.set_yticks(yticks)

    @staticmethod
    def setTickLabelsFontSize(ax, fs):
        ax.tick_params(axis='both', which='major', labelsize=fs)

    @staticmethod
    def setXLabel(ax, xplt, fs):
        ax.set_xlabel('$\\rm {}\ ({})$'.format(xplt["label"], xplt["unit"]), fontsize=fs)

    @staticmethod
    def setYLabel(ax, yplt, fs):
        ax.set_ylabel('$\\rm {}\ ({})$'.format(yplt["label"], yplt.get("unit", "")), fontsize=fs)

    @classmethod
    def addCmap(cls, fig, cmap, handles, comp_values, comp_info, fs, prettify, zscale='lin'):

        if all(isinstance(x, str) for x in comp_values):
            # If list of strings, assume that index suffixes can be extracted
            prefix, suffixes = extractCommonPrefix(comp_values)
            comp_values = [int(s) for s in suffixes]
            desc_str = f'{prefix}\ index'
        else:
            # Rescale comparison values and adjust unit
            comp_values = np.asarray(comp_values) * comp_info.get('factor', 1.)
            comp_factor, comp_prefix = getSIpair(comp_values, scale=zscale)
            comp_values /= comp_factor
            comp_info['unit'] = comp_prefix + comp_info['unit']
            desc_str = comp_info["desc"].replace(" ", "\ ")
            if len(comp_info['unit']) > 0:
                desc_str = f"{desc_str}\ ({comp_info['unit']})"
        nvalues = len(comp_values)

        # Create colormap and normalizer
        try:
            mymap = plt.get_cmap(cmap)
        except ValueError:
            mymap = plt.get_cmap(swapFirstLetterCase(cmap))
        norm, sm = setNormalizer(mymap, (min(comp_values), max(comp_values)), zscale)

        # Extract and adjust line colors
        zcolors = sm.to_rgba(comp_values)
        for lh, c in zip(handles, zcolors):
            if isIterable(lh):
                for item in lh:
                    item.set_color(c)
            else:
                lh.set_color(c)

        # Add colorbar
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
        cbar_kwargs = {}
        if all(isinstance(x, int) for x in comp_values):
            dx = np.diff(comp_values)
            if all(x == dx[0] for x in dx):
                dx = dx[0]
                ticks = comp_values
                bounds = np.hstack((ticks, [max(ticks) + dx])) - dx / 2
                if nvalues > 10:
                    ticks = [ticks[0], ticks[-1]]
                cbar_kwargs.update({'ticks': ticks, 'boundaries': bounds, 'format': '%1i'})
                cbarax.tick_params(axis='both', which='both', length=0)
        cbar = fig.colorbar(sm, cax=cbarax, **cbar_kwargs)
        fig.sm = sm  # add scalar mappable as figure attribute in case of future need
        cbarax.set_ylabel(f'$\\rm {desc_str}$', fontsize=fs)
        if prettify:
            cls.prettify(cbar)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)


class ComparativePlot(GenericPlot):

    def __init__(self, outputs, varname):
        ''' Constructor.

            :param outputs: list /generator of simulation outputs to be compared.
            :param varname: name of variable to extract and compare.
        '''
        super().__init__(outputs)
        self.varname = varname
        self.comp_ref_key = None
        self.meta_ref = None
        self.comp_info = None
        self.is_unique_comp = False

    def checkLabels(self, labels):
        if labels is not None:
            if not isIterable(labels):
                raise TypeError('Invalid labels: must be an iterable')
            if not all(isinstance(x, str) for x in labels):
                raise TypeError('Invalid labels: must be string typed')

    def checkSimType(self, meta):
        ''' Check consistency of sim types across files. '''
        if meta['simkey'] != self.meta_ref['simkey']:
            raise ValueError('Invalid comparison: different simulation types')

    def checkCompValues(self, meta, comp_values):
        ''' Check consistency of differing values across files. '''
        # Get differing values across meta dictionaries
        diffs = differing(self.meta_ref, meta, subdkey='meta')

        # Check that only one value differs
        if len(diffs) > 1:
            logger.warning('More than one differing inputs')
            self.comp_ref_key = None
            return []

        # Get the key and differing values
        zkey, refval, val = diffs[0]

        # If no comparison key yet, fill it up
        if self.comp_ref_key is None:
            self.comp_ref_key = zkey
            self.is_unique_comp = True
            comp_values += [refval, val]
        # Otherwise, check that comparison matches the existing one
        else:
            if zkey != self.comp_ref_key:
                logger.warning('inconsistent differing inputs')
                self.comp_ref_key = None
                return []
            else:
                comp_values.append(val)
        return comp_values

    def checkConsistency(self, meta, comp_values):
        ''' Check consistency of sim types and check differing inputs. '''
        if self.meta_ref is None:
            self.meta_ref = meta
        else:
            self.checkSimType(meta)
            comp_values = self.checkCompValues(meta, comp_values)
            if self.comp_ref_key is None:
                self.is_unique_comp = False
        return comp_values

    def getCompLabels(self, comp_values):
        if self.comp_info is not None:
            comp_values = np.array(comp_values) * self.comp_info.get('factor', 1)
            if 'unit' in self.comp_info:
                p = self.comp_info.get('precision', 0)
                comp_values = [f"{si_format(v, p)}{self.comp_info['unit']}".replace(' ', '\ ')
                               for v in comp_values]
            comp_labels = ['$\\rm{} = {}$'.format(self.comp_info['label'], x) for x in comp_values]
        else:
            comp_labels = comp_values
        return comp_values, comp_labels

    def chooseLabels(self, labels, comp_labels, full_labels):
        if labels is not None:
            return labels
        else:
            if self.is_unique_comp:
                return comp_labels
            else:
                return full_labels

    @staticmethod
    def getCommonLabel(lbls, seps='_'):
        ''' Get a common label from a list of labels, by removing parts that differ across them. '''

        # Split every label according to list of separator characters, and save splitters as well
        splt_lbls = [re.split(f'([{seps}])', x) for x in lbls]
        pieces = [x[::2] for x in splt_lbls]
        splitters = [x[1::2] for x in splt_lbls]
        ncomps = len(pieces[0])

        # Assert that splitters are equivalent across all labels, and reduce them to a single array
        assert (x == x[0] for x in splitters), 'Inconsistent splitters'
        splitters = np.array(splitters[0])

        # Transform pieces into 2D matrix, and evaluate equality of every piece across labels
        pieces = np.array(pieces).T
        all_identical = [np.all(x == x[0]) for x in pieces]
        if np.sum(all_identical) < ncomps - 1:
            logger.warning('More than one differing inputs')
            return ''

        # Discard differing pieces and remove associated splitters
        pieces = pieces[all_identical, 0]
        splitters = splitters[all_identical[:-1]]

        # Remove last splitter if the last pieces were discarded
        if splitters.size == pieces.size:
            splitters = splitters[:-1]

        # Join common pieces and associated splitters into a single label
        common_lbl = ''
        for p, s in zip(pieces, splitters):
            common_lbl += f'{p}{s}'
        common_lbl += pieces[-1]

        return common_lbl.replace('( ', '(')


def addExcitationInset(ax, is_excited):
    ''' Add text inset on axis stating excitation status. '''
    ax.text(
        0.7, 0.7, f'{"" if is_excited else "not "}excited',
        transform=ax.transAxes,
        ha='center', va='center', size=30, bbox=dict(
            boxstyle='round',
            fc=(0.8, 1.0, 0.8) if is_excited else (1., 0.8, 0.8)
        ))


def mirrorProp(org, new, prop):
    ''' Mirror an instance property onto another instance of the same class. '''
    getattr(new, f'set_{prop}')(getattr(org, f'get_{prop}')())


def mirrorAxis(org_ax, new_ax, p=False):
    ''' Mirror content of original axis to a new axis. That includes:
        - position on the figure
        - spines properties
        - ticks, ticklabels, and labels
        - vertical spans
    '''
    mirrorProp(org_ax, new_ax, 'position')
    for sk in ['bottom', 'left', 'right', 'top']:
        mirrorProp(org_ax.spines[sk], new_ax.spines[sk], 'visible')
    for prop in ['label', 'ticks', 'ticklabels']:
        for k in ['x', 'y']:
            mirrorProp(org_ax, new_ax, f'{k}{prop}')
    ax_children = org_ax.get_children()
    vspans = filter(lambda x: isinstance(x, Polygon), ax_children)
    for vs in vspans:
        props = vs.properties()
        xmin, xmax = [props['xy'][i][0] for i in [0, 2]]
        kwargs = {k: props[k] for k in ['alpha', 'edgecolor', 'facecolor']}
        if kwargs['edgecolor'] == (0.0, 0.0, 0.0, 0.0):
            kwargs['edgecolor'] = 'none'
        new_ax.axvspan(xmin, xmax, **kwargs)


def harmonizeAxesLimits(axes, dim='xy'):
    ''' Harmonize x and/or y limits across an array of axes. '''
    axes = axes.flatten()
    xlims, ylims = [np.inf, -np.inf], [np.inf, -np.inf]
    for ax in axes:
        xlims = [min(xlims[0], ax.get_xlim()[0]), max(xlims[1], ax.get_xlim()[1])]
        ylims = [min(ylims[0], ax.get_ylim()[0]), max(ylims[1], ax.get_ylim()[1])]
    for ax in axes:
        if dim in ['xy', 'x']:
            ax.set_xlim(*xlims)
        if dim in ['xy', 'y']:
            ax.set_ylim(*ylims)


def hideSpines(ax, spines='all'):
    if isIterable(ax):
        for item in ax:
            hideSpines(item, spines=spines)
    else:
        if spines == 'all':
            spines = ['top', 'bottom', 'right', 'left']
        for sk in spines:
            ax.spines[sk].set_visible(False)


def hideTicks(ax, key='xy'):
    if isIterable(ax):
        for item in ax:
            hideTicks(item, key=key)
    if key in ['xy', 'x']:
        ax.set_xticks([])
    if key in ['xy', 'y']:
        ax.set_yticks([])


def addXscale(ax, xoffset, yoffset, unit='', lw=2, fmt='.0f', fs=10, side='bottom'):
    ybase = {'bottom': 0, 'top': 1}[side]
    text_extra_yoffset = 0.07
    if side == 'bottom':
        yoffset = -yoffset
        text_extra_yoffset = -text_extra_yoffset
    ax.plot([xoffset, 1 + xoffset], [ybase + yoffset] * 2, c='k',
            transform=ax.transAxes, linewidth=lw, clip_on=False)
    xytext = (0.5 + xoffset, ybase + yoffset + text_extra_yoffset)
    va = {'top': 'bottom', 'bottom': 'top'}[side]
    xscale = np.ptp(ax.get_xlim())
    ax.text(*xytext, f'{xscale:{fmt}} {unit}', transform=ax.transAxes,
            ha='center', va=va, fontsize=fs)


def addYscale(ax, xoffset, yoffset, unit='', lw=2, fmt='.0f', fs=10, side='right'):
    xbase = {'left': 0, 'right': 1}[side]
    text_extra_xoffset = 0.07
    if side == 'left':
        xoffset = -xoffset
        text_extra_xoffset = -text_extra_xoffset
    ax.plot([xbase + xoffset] * 2, [yoffset, 1 + yoffset], c='k',
            transform=ax.transAxes, linewidth=lw, clip_on=False)
    xytext = (xbase + xoffset + text_extra_xoffset, .5 + yoffset)
    ha = {'left': 'right', 'right': 'left'}[side]
    yscale = np.ptp(ax.get_ylim())
    ax.text(*xytext, f'{yscale:{fmt}} {unit}', transform=ax.transAxes,
            ha=ha, va='center', rotation=90, fontsize=fs)
