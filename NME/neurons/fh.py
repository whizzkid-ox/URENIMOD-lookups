# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-01-07 18:41:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 16:19:53

import numpy as np
from ..core import PointNeuron, addSonicFeatures
from ..constants import Z_Na, Z_K


@addSonicFeatures
class FrankenhaeuserHuxleyNode(PointNeuron):
    ''' Amphibien (xenopus) myelinated fiber node.

        Reference:
        *Frankenhaeuser, B., and Huxley, A.F. (1964). The action potential in the myelinated nerve
        fibre of Xenopus laevis as computed on the basis of voltage clamp data.
        J Physiol 171, 302–315.*
    '''

    # Mechanism name
    name = 'FHnode'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 2e-2  # Membrane capacitance (F/m2)
    Vm0 = -70.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -69.974  # Leakage resting potential (mV)

    # Maximal channel conductances (S/m2)
    gLeak = 300.3  # Leakage conductance (S/m2)

    # Channel permeability constant (m/s)
    pNabar = 8e-5   # Sodium
    pKbar = 1.2e-5  # Potassium
    pPbar = .54e-5  # Non-specific

    # Ionic concentrations (M)
    Nai = 13.74e-3  # Intracellular Sodium
    Nao = 114.5e-3  # Extracellular Sodium
    Ki = 120e-3     # Intracellular Potassium
    Ko = 2.5e-3     # Extracellular Potassium

    # Additional parameters
    celsius_FH = 20.0  # Temperature in Frankenhaeuser-Huxley 1964 (Celsius)
    # celsius = 20.0     # Temperature (Celsius)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iP gate'
    }

    def __new__(cls):
        cls.q10 = 3**((cls.celsius - cls.celsius_FH) / 10)
        return super(FrankenhaeuserHuxleyNode, cls).__new__(cls)

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars['Qm']['bounds'] = (-150, 100)
        return pltvars

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return cls.q10 * 0.36 * cls.vtrap(22. - (Vm - cls.Vm0), 3.) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return cls.q10 * 0.4 * cls.vtrap(Vm - cls.Vm0 - 13., 20.) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return cls.q10 * 0.1 * cls.vtrap(Vm - cls.Vm0 + 10.0, 6.) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return cls.q10 * 4.5 / (np.exp((45. - (Vm - cls.Vm0)) / 10.) + 1) * 1e3  # s-1

    @classmethod
    def alphan(cls, Vm):
        return cls.q10 * 0.02 * cls.vtrap(35. - (Vm - cls.Vm0), 10.0) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return cls.q10 * 0.05 * cls.vtrap(Vm - cls.Vm0 - 10., 10.) * 1e3  # s-1

    @classmethod
    def alphap(cls, Vm):
        return cls.q10 * 0.006 * cls.vtrap(40. - (Vm - cls.Vm0), 10.0) * 1e3  # s-1

    @classmethod
    def betap(cls, Vm):
        return cls.q10 * 0.09 * cls.vtrap(Vm - cls.Vm0 + 25., 20.) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'p': lambda Vm, x: cls.alphap(Vm) * (1 - x['p']) - cls.betap(Vm) * x['p']
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            'p': lambda Vm: cls.alphap(Vm) / (cls.alphap(Vm) + cls.betap(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.pNabar * m**2 * h * cls.ghkDrive(Vm, Z_Na, cls.Nai, cls.Nao, cls.T)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.pKbar * n**2 * cls.ghkDrive(Vm, Z_K, cls.Ki, cls.Ko, cls.T)  # mA/m2

    @classmethod
    def iP(cls, p, Vm):
        ''' non-specific delayed current '''
        return cls.pPbar * p**2 * cls.ghkDrive(Vm, Z_Na, cls.Nai, cls.Nao, cls.T)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iP': lambda Vm, x: cls.iP(x['p'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        return super().chooseTimeStep() * 1e-1
