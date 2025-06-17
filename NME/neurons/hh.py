# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-07-21 14:53:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 16:14:22

import numpy as np

from ..core import PointNeuron, addSonicFeatures


@addSonicFeatures
class HodgkinHuxleySegment(PointNeuron):
    ''' Unmyelinated giant squid axon segment.

        Reference:
        *A quantitative description of membrane current and its application to conduction
        and excitation in nerve. J.Physiol. 117:500-544 (1952).*
    '''

    # Neuron name
    name = 'HHseg'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -65.0  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 50.      # Sodium
    EK = -77.      # Potassium
    ELeak = -54.3  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 1200.0  # Sodium
    gKdbar = 360.0   # Delayed-rectifier Potassium
    gLeak = 3.0      # Non-specific leakage

    # Additional parameters
    celsius_HH = 6.3  # Temperature in Hodgkin Huxley 1952 (Celsius)
    # celsius = 6.3  # Temperature (Celsius)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def __new__(cls):
        cls.q10 = 3**((cls.celsius - cls.celsius_HH) / 10.)  # from Hodgkin Huxley 1952
        return super(HodgkinHuxleySegment, cls).__new__(cls)

    @classmethod
    def alpham(cls, Vm):
        return cls.q10 * 0.1 * cls.vtrap(-(Vm + 40), 10) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return cls.q10 * 4 * np.exp(-(Vm + 65) / 18) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return cls.q10 * 0.07 * np.exp(-(Vm + 65) / 20) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return cls.q10 * 1.0 / (np.exp(-(Vm + 35) / 10) + 1) * 1e3  # s-1

    @classmethod
    def alphan(cls, Vm):
        return cls.q10 * 0.01 * cls.vtrap(-(Vm + 55), 10) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return cls.q10 * 0.125 * np.exp(-(Vm + 65) / 80) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n']
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**4 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        ''' neuron-specific time step for fast dynamics. '''
        return super().chooseTimeStep() * 1e-1
