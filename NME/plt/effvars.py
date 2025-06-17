# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-10-02 01:44:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-23 13:45:31

from inspect import signature
import numpy as np
import matplotlib.pyplot as plt

from ..utils import logger, isWithin, getSIpair
from ..core import NeuronalBilayerSonophore
from .pltutils import setGrid, setNormalizer


def isVoltageDependent(func):
    return 'Vm' in list(signature(func).parameters.keys())


def plotGatingKinetics(pneuron, fs=15, tau_scale='lin', tau_pas=True):
    ''' Plot the voltage-dependent steady-states and time constants of activation and
        inactivation gates of the different ionic currents involved in a specific
        neuron's membrane.

        :param pneuron: point-neuron model
        :param fs: labels and title font size
        :param tau_scale: y-scale type for time constants ("lin" or "log")
    '''

    # Input membrane potential vector
    Vm = np.linspace(-100, 50, 300)

    xinf_dict = {}
    taux_dict = {}

    logger.info('Computing %s neuron gating kinetics', pneuron.name)
    names = pneuron.states
    for xname in names:
        Vm_state = True

        # Names of functions of interest
        xinf_func_str = xname.lower() + 'inf'
        taux_func_str = 'tau' + xname.lower()
        alphax_func_str = 'alpha' + xname.lower()
        betax_func_str = 'beta' + xname.lower()
        # derx_func_str = 'der' + xname.upper()

        # 1st choice: use xinf and taux function
        if hasattr(pneuron, xinf_func_str) and hasattr(pneuron, taux_func_str):
            xinf_func = getattr(pneuron, xinf_func_str)
            taux_func = getattr(pneuron, taux_func_str)
            xinf = np.array([xinf_func(v) for v in Vm])
            if isinstance(taux_func, float):
                taux = taux_func * np.ones(len(Vm))
            else:
                taux = np.array([taux_func(v) for v in Vm])

        # 2nd choice: use alphax and betax functions
        elif hasattr(pneuron, alphax_func_str) and hasattr(pneuron, betax_func_str):
            alphax_func = getattr(pneuron, alphax_func_str)
            betax_func = getattr(pneuron, betax_func_str)
            if not isVoltageDependent(alphax_func):
                Vm_state = False
            else:
                alphax = np.array([alphax_func(v) for v in Vm])
                if isinstance(betax_func, float):
                    betax = betax_func * np.ones(len(Vm))
                else:
                    betax = np.array([betax_func(v) for v in Vm])
                taux = 1.0 / (alphax + betax)
                xinf = taux * alphax

        # # 3rd choice: use derX choice
        # elif hasattr(pneuron, derx_func_str):
        #     derx_func = getattr(pneuron, derx_func_str)
        #     xinf = brentq(lambda x: derx_func(pneuron.Vm, x), 0, 1)
        else:
            Vm_state = False

        if not Vm_state:
            logger.error('%s-state gating kinetics is not Vm-dependent', xname)
        else:
            xinf_dict[xname] = xinf
            taux_dict[xname] = taux

    fig, axes = plt.subplots(2)
    fig.suptitle(f'{pneuron.name} neuron: gating dynamics')

    ax = axes[0]
    ax.get_xaxis().set_ticklabels([])
    ax.set_ylabel('$X_{\infty}$', fontsize=fs)
    for xname in names:
        if xname in xinf_dict:
            ax.plot(Vm, xinf_dict[xname], lw=2, label='$' + xname + '_{\infty}$')
    ax.legend(fontsize=fs, loc=7)

    ax = axes[1]
    ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
    ax.set_ylabel('$\\tau_X\ (ms)$', fontsize=fs)
    if tau_scale == 'log':
        ax.set_yscale('log')
    for xname in names:
        if xname in taux_dict:
            ax.plot(Vm, taux_dict[xname] * 1e3, lw=2, label='$\\tau_{' + xname + '}$')
    if tau_pas:
        ax.axhline(pneuron.tau_pas * 1e3, lw=2, linestyle='--', c='k', label='$\\tau_{pas}$')
    ax.legend(fontsize=fs, loc=7)

    return fig


def plotEffectiveVariables(pneuron, a=None, f=None, A=None, nlevels=10,
                           zscale='lin', cmap=None, fs=12, ncolmax=1):
    ''' Plot the profiles of effective variables of a specific neuron as a function of charge density
        and another reference variable (z-variable). For each effective variable, one charge-profile
        per z-value is plotted, with a color code based on the z-variable value.

        :param pneuron: point-neuron model
        :param a: sonophore radius (m)
        :param f: acoustic drive frequency (Hz)
        :param A: acoustic pressure amplitude (Pa)
        :param nlevels: number of levels for the z-variable
        :param zscale: scale type for the z-variable ('lin' or 'log')
        :param cmap: colormap name
        :param fs: figure fontsize
        :param ncolmax: max number of columns on the figure
        :return: handle to the created figure
    '''
    if sum(isinstance(x, float) for x in [a, f, A]) < 2:
        raise ValueError('at least 2 parameters in (a, f, A) must be fixed')

    if cmap is None:
        cmap = 'viridis'

    nbls = NeuronalBilayerSonophore(32e-9, pneuron)
    pltvars = nbls.getPltVars()

    # Get lookups and re-organize them
    lkp = nbls.getLookup().squeeze()
    Qref = lkp.refs['Q']
    lkp.rename('V', 'Vm')
    lkp['Cm'] = Qref / lkp['Vm'] * 1e3  # uF/cm2

    # Sort keys for display
    keys = lkp.outputs
    del keys[keys.index('Cm')]
    del keys[keys.index('Vm')]
    keys = ['Cm', 'Vm'] + keys

    # Get reference US-OFF lookups (1D)
    lookupsoff = lkp.projectOff()

    # Get 2D lookups at specific combination
    inputs = {}
    for k, v in {'a': a, 'f': f, 'A': A}.items():
        if v is not None:
            inputs[k] = v
    lookups2D = lkp.projectN(inputs)

    # Get z-variable from remaining inputs
    for key in lookups2D.inputs:
        if key != 'Q':
            zkey = key
            zref = lookups2D.refs[key]
    zvar = nbls.inputs()[zkey]
    zref *= zvar.get('factor', 1.)

    # Optional: interpolate along z dimension if nlevels specified
    if zscale == 'log':
        zref_pos = zref[zref > 0]
        znew = np.logspace(np.log10(zref_pos.min()), np.log10(zref_pos.max()), nlevels)
    elif zscale == 'lin':
        znew = np.linspace(zref.min(), zref.max(), nlevels)
    else:
        raise ValueError('unknown scale type (should be "lin" or "log")')
    znew = np.array([isWithin(zvar['label'], z, (zref.min(), zref.max())) for z in znew])
    lookups2D = lookups2D.project(zkey, znew)
    zref = znew

    #  Define color code
    mymap = plt.get_cmap(cmap)
    norm, sm = setNormalizer(mymap, (zref.min(), zref.max()), zscale)

    # Plot
    logger.info('plotting')
    nrows, ncols = setGrid(len(keys), ncolmax=ncolmax)
    xvar = pltvars['Qm']
    Qbounds = np.array([Qref.min(), Qref.max()]) * xvar['factor']

    fig, _ = plt.subplots(figsize=(3.5 * ncols, 1 * nrows), squeeze=False)
    for j, key in enumerate(keys):
        ax = plt.subplot2grid((nrows, ncols), (j // ncols, j % ncols))
        for s in ['right', 'top']:
            ax.spines[s].set_visible(False)
        yvar = pltvars[key]
        if j // ncols == nrows - 1:
            ax.set_xlabel('$\\rm {}\ ({})$'.format(xvar["label"], xvar["unit"]), fontsize=fs)
            ax.set_xticks(Qbounds)
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.02, 0.5)

        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

        ymin = np.inf
        ymax = -np.inf

        # Plot effective variable for each selected z-value
        y0 = lookupsoff[key]
        for i, z in enumerate(zref):
            y = lookups2D[key][i]
            if 'alpha' in key or 'beta' in key:
                y[y > y0.max() * 2] = np.nan
            ax.plot(Qref * xvar.get('factor', 1), y * yvar.get('factor', 1), c=sm.to_rgba(z))
            ymin = min(ymin, y.min())
            ymax = max(ymax, y.max())

        # Plot reference variable
        ax.plot(Qref * xvar.get('factor', 1), y0 * yvar.get('factor', 1), '--', c='k')
        ymax = max(ymax, y0.max())
        ymin = min(ymin, y0.min())

        # Set axis y-limits
        if 'alpha' in key or 'beta' in key:
            ymax = y0.max() * 2
        ylim = [ymin * yvar.get('factor', 1), ymax * yvar.get('factor', 1)]
        if key == 'Cm':
            factor = 1e1
            ylim = [np.floor(ylim[0] * factor) / factor, np.ceil(ylim[1] * factor) / factor]
        else:
            factor = 1 / np.power(10, np.floor(np.log10(ylim[1])))
            ylim = [np.floor(ylim[0] * factor) / factor, np.ceil(ylim[1] * factor) / factor]
        ax.set_yticks(ylim)
        ax.set_ylim(ylim)
        ax.set_ylabel('$\\rm {}\ ({})$'.format(yvar["label"], yvar["unit"]),
                      fontsize=fs, rotation=0, ha='right', va='center')

    fig.suptitle(f'{pneuron.name} neuron: {zvar["label"]} \n modulated effective variables')

    # Adjust colorbar factor and unit prefix if zvar is US frequency of amplitude
    zfactor, zprefix = getSIpair(zref, scale=zscale)
    zvar['unit'] = zprefix + zvar['unit']
    _, sm = setNormalizer(mymap, (zref.min() / zfactor, zref.max() / zfactor), zscale)

    # Plot colorbar
    fig.subplots_adjust(left=0.20, bottom=0.05, top=0.8, right=0.80, hspace=0.5)
    cbarax = fig.add_axes([0.10, 0.90, 0.80, 0.02])
    fig.colorbar(sm, cax=cbarax, orientation='horizontal')
    cbarax.set_xlabel(f'{zvar["label"]} ({zvar["unit"]})', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return fig
