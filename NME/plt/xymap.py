# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:24:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-18 18:35:13

import abc
import csv
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RectBivariateSpline

from ..core import LogBatch
from ..utils import logger, isIterable, rangecode, bounds
from .pltutils import cm2inch, setNormalizer


class XYMap(LogBatch):
    ''' Generic 2D map object interface. '''

    offset_options = {
        'lr': (1, -1),
        'ur': (1, 1),
        'll': (-1, -1),
        'ul': (-1, 1)
    }

    def __init__(self, root, xvec, yvec):
        self.root = root
        self.xvec = xvec
        self.yvec = yvec
        super().__init__([list(pair) for pair in product(self.xvec, self.yvec)], root=root)

    def checkVector(self, name, value):
        if not isIterable(value):
            raise ValueError(f'{name} vector must be an iterable')
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if len(value.shape) > 1:
            raise ValueError(f'{name} vector must be one-dimensional')
        return value

    @property
    def in_key(self):
        return self.xkey

    @property
    def unit(self):
        return self.xunit

    @property
    def xvec(self):
        return self._xvec

    @xvec.setter
    def xvec(self, value):
        self._xvec = self.checkVector('x', value)

    @property
    def yvec(self):
        return self._yvec

    @yvec.setter
    def yvec(self, value):
        self._yvec = self.checkVector('x', value)

    @property
    @abc.abstractmethod
    def xkey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xfactor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ykey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def yfactor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def yunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zkey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zfactor(self):
        raise NotImplementedError

    @property
    def out_keys(self):
        return [f'{self.zkey} ({self.zunit})']

    @property
    def in_labels(self):
        return [f'{self.xkey} ({self.xunit})', f'{self.ykey} ({self.yunit})']

    def getLogData(self):
        ''' Retrieve the batch log file data (inputs and outputs) as a dataframe. '''
        return pd.read_csv(self.fpath, sep=self.delimiter).sort_values(self.in_labels)

    def getInput(self):
        ''' Retrieve the logged batch inputs as an array. '''
        return self.getLogData()[self.in_labels].values

    def getOutput(self):
        ''' Return map output, shaped as an nx-by-ny matrix. '''
        return np.reshape(super().getOutput(), (self.xvec.size, self.yvec.size))

    def writeLabels(self):
        with open(self.fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow([*self.in_labels, *self.out_keys])

    def isEntry(self, comb):
        ''' Check if a given input is logged in the batch log file. '''
        inputs = self.getInput()
        if len(inputs) == 0:
            return False
        imatches_x = np.where(np.isclose(inputs[:, 0], comb[0], rtol=self.rtol, atol=self.atol))[0]
        imatches_y = np.where(np.isclose(inputs[:, 1], comb[1], rtol=self.rtol, atol=self.atol))[0]
        imatches = list(set(imatches_x).intersection(imatches_y))
        if len(imatches) == 0:
            return False
        return True

    @property
    def inputscode(self):
        ''' String describing the batch inputs. '''
        xcode = rangecode(self.xvec, self.xkey, self.xunit)
        ycode = rangecode(self.yvec, self.ykey, self.yunit)
        return '_'.join([xcode, ycode])

    @staticmethod
    def getScaleType(x):
        xmin, xmax, nx = x.min(), x.max(), x.size
        if np.all(np.isclose(x, np.logspace(np.log10(xmin), np.log10(xmax), nx))):
            return 'log'
        else:
            return 'lin'
        # elif np.all(np.isclose(x, np.linspace(xmin, xmax, nx))):
        #     return 'lin'
        # else:
        #     raise ValueError('Unknown distribution type')

    @property
    def xscale(self):
        return self.getScaleType(self.xvec)

    @property
    def yscale(self):
        return self.getScaleType(self.yvec)

    @staticmethod
    def computeMeshEdges(x, scale):
        ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

            :param x: the input vector
            :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
            :return: the edges vector
        '''
        if scale == 'log':
            x = np.log10(x)
            range_func = np.logspace
        else:
            range_func = np.linspace
        dx = x[1] - x[0]
        n = x.size + 1
        return range_func(x[0] - dx / 2, x[-1] + dx / 2, n)

    @abc.abstractmethod
    def compute(self, x):
        ''' Compute the necessary output(s) for a given inputs combination. '''
        raise NotImplementedError

    def run(self, **kwargs):
        super().run(**kwargs)
        self.getLogData().to_csv(self.filepath(), sep=self.delimiter, index=False)

    def getOnClickXY(self, event):
        ''' Get x and y values from from x and y click event coordinates. '''
        x = self.xvec[np.searchsorted(self.xedges, event.xdata) - 1]
        y = self.yvec[np.searchsorted(self.yedges, event.ydata) - 1]
        return x, y

    def onClickWrapper(self, event):
        if event.inaxes == self.ax:
            return self.onClick(event)

    def onClick(self, event):
        ''' Exexecute specific action when the user clicks on a cell in the 2D map. '''
        pass

    @property
    @abc.abstractmethod
    def title(self):
        raise NotImplementedError

    def getZBounds(self):
        matrix = self.getOutput() * self.zfactor
        zmin, zmax = np.nanmin(matrix), np.nanmax(matrix)
        logger.info(
            f'{self.zkey} range: {zmin:.2f} - {zmax:.2f} {self.zunit}')
        return zmin, zmax

    def checkZbounds(self, zbounds):
        zmin, zmax = self.getZBounds()
        if zmin < zbounds[0]:
            logger.warning(
                f'Minimal {self.zkey} ({zmin:.2f} {self.zunit}) is below defined lower bound ({zbounds[0]:.2f} {self.zunit})')
        if zmax > zbounds[1]:
            logger.warning(
                f'Maximal {self.zkey} ({zmax:.2f} {self.zunit}) is above defined upper bound ({zbounds[1]:.2f} {self.zunit})')

    @staticmethod
    def addInsets(ax, insets, fs, minimal=False):
        ax.update_datalim(list(insets.values()))
        xyoffset = np.array([0, 0.05])
        data_to_axis = ax.transData + ax.transAxes.inverted()
        for k, xydata in insets.items():
            ax.scatter(*xydata, s=20, facecolor='k', edgecolor='none')
            if not minimal:
                xyaxis = np.array(data_to_axis.transform(xydata))
                ax.annotate(
                    k, xy=xyaxis, xytext=xyaxis + xyoffset, xycoords=ax.transAxes,
                    fontsize=fs, arrowprops={'facecolor': 'black', 'arrowstyle': '-'},
                    ha='right')

    @staticmethod
    def extrapolate(xref, yref, data, xscale, yscale, xextra=None, yextra=None):
        nextra = sum([x is not None for x in [xextra, yextra]])
        if nextra == 0:
            raise ValueError('no extrapolation vector provided')
        if nextra == 2:
            x, y, data = XYMap.extrapolate(xref, yref, data, xscale, yscale, xextra=xextra)
            return XYMap.extrapolate(x, y, data, xscale, yscale, yextra=yextra)
        if xscale == 'log':
            xref = np.log10(xref)
        if yscale == 'log':
            yref = np.log10(yref)
        valid_data = ~np.isnan(data)
        validrows, validcols = [np.all(valid_data, axis=i) for i in [1, 0]]
        ref_xyz = [xref, yref, data]
        if xextra is not None:
            k, stackaxis, vref, vextra, vscale, other = 'x', 0, xref, xextra, xscale, yref
            reverse = False
            ref_xyz = ref_xyz[0][validrows], ref_xyz[1], ref_xyz[-1][validrows, :]
        if yextra is not None:
            k, stackaxis, vref, vextra, vscale, other = 'y', 1, yref, yextra, yscale, xref
            reverse = True
            ref_xyz = ref_xyz[0], ref_xyz[1][validcols], ref_xyz[-1][:, validcols]
        f = RectBivariateSpline(*ref_xyz)
        vmin, vmax = bounds(vref)
        if any(vmin < vv < vmax for vv in vextra):
            raise ValueError(f'new {k} vector must sit entirely outside of reference {k}-range')
        if vscale == 'log':
            vextra = np.log10(vextra)
        interp_xy = [vextra, other]
        if reverse:
            interp_xy = interp_xy[::-1]
        interp_data = f(*interp_xy)
        if vextra[0] > vref.max():
            v = (vref, vextra)
            data = (data, interp_data)
        else:
            v = (vextra, vref)
            data = (interp_data, data)
        v = np.hstack(v)
        data = np.concatenate(data, axis=stackaxis)

        if xextra is not None:
            x, y = v, yref
        if yextra is not None:
            x, y = xref, v

        if xscale == 'log':
            x = np.power(10., x)
        if yscale == 'log':
            y = np.power(10., y)
        return x, y, data

    def render(self, xscale='lin', yscale='lin', zscale='lin', zbounds=None, fs=8, cmap='viridis',
               interactive=False, figsize=None, insets=None, inset_offset=0.05,
               extend_under=False, extend_over=False, ax=None, cbarax=None, cbarlabel='vertical',
               title=None, minimal=False, levels=None, flip=False, plt_cbar=True,
               xextra=None, yextra=None, render_mode='map', ccolor='k'):
        # Compute z-bounds
        if zbounds is None:
            extend_under = False
            extend_over = False
            zbounds = self.getZBounds()
        else:
            self.checkZbounds(zbounds)
        # Compute Z normalizer
        mymap = copy.copy(plt.get_cmap(cmap))
        mymap.set_bad('silver')
        if not extend_under:
            mymap.set_under('silver')
        if not extend_over:
            mymap.set_over('silver')
        norm, sm = setNormalizer(mymap, zbounds, zscale)

        # Create figure if required
        if ax is None:
            if figsize is None:
                figsize = cm2inch(12, 7)
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
        else:
            fig = ax.get_figure()

        # Set axis properties
        if title is None:
            title = self.title
        if len(title) > 0:
            ax.set_title(title, fontsize=fs)
        if minimal:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis='both', which='both', bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        else:
            ax.set_xlabel(f'{self.xkey} ({self.xunit})', fontsize=fs, labelpad=-0.5)
            ax.set_ylabel(f'{self.ykey} ({self.yunit})', fontsize=fs)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')

        # Retrieve data and extrapolate if needed
        x, y, data = self.xvec, self.yvec, self.getOutput() * self.zfactor
        if xextra is not None:
            x, y, data = self.extrapolate(x, y, data, xscale, yscale, xextra=xextra)
        elif yextra is not None:
            x, y, data = self.extrapolate(x, y, data, xscale, yscale, yextra=yextra)

        # Flip data if required
        if flip:
            data = data.T

        # If map render mode
        if render_mode == 'map':
            # Compute mesh edges and plot map with specific color code
            self.xedges = self.computeMeshEdges(x, xscale)
            self.yedges = self.computeMeshEdges(y, yscale)
            ax.pcolormesh(self.xedges, self.yedges, data.T, cmap=mymap, norm=norm)
            # Add contour levels if needed
            if levels is not None:
                CS = ax.contour(x, y, data.T, levels, colors=[ccolor])
                ax.clabel(CS, fontsize=fs, fmt=lambda x: f'{x:g}', inline_spacing=2)
        else:
            if levels is None or len(levels) != 1:
                raise ValueError('conv/div rendering requires exactly 1 threshold level')
            plt_cbar = False
            ax.contour(x, y, data.T, levels, colors=[ccolor])
            zthr = levels[0]
            if render_mode == 'divarea':
                ax.contourf(x, y, data.T, [zthr, np.inf], colors=[ccolor], alpha=0.2)
            elif render_mode == 'convarea':
                ax.contourf(x, y, data.T, [-np.inf, zthr], colors=[ccolor], alpha=0.2)

        # Add potential insets
        if insets is not None:
            self.addInsets(ax, insets, fs, minimal=minimal)

        # Plot z-scale colorbar if required
        if plt_cbar:
            if cbarax is None:
                pos1 = ax.get_position()  # get the map axis position
                cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
            if not extend_under and not extend_over:
                extend = 'neither'
            elif extend_under and extend_over:
                extend = 'both'
            else:
                extend = 'max' if extend_over else 'min'
            self.cbar = plt.colorbar(sm, cax=cbarax, extend=extend)
            if cbarlabel == 'vertical':
                cbarax.set_ylabel(f'{self.zkey} ({self.zunit})', fontsize=fs)
            else:
                cbarax.set_title(f'{self.zkey} ({self.zunit})', fontsize=fs)
            for item in cbarax.get_yticklabels():
                item.set_fontsize(fs)

        if interactive:
            self.ax = ax
            fig.canvas.mpl_connect('button_press_event', self.onClickWrapper)

        return fig
