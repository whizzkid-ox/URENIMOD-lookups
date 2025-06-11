# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-03 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-05 17:36:34

import numpy as np
from ..core import PointNeuron, addSonicFeatures
from ..utils import logger
from ..postpro import detectSpikes


@addSonicFeatures
class SundtSegment(PointNeuron):
    ''' Unmyelinated C-fiber segment.

        Reference:
        *Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)*
    '''

    # Mechanism name
    name = 'SUseg'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)
    Vm0 = -60.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0  # Sodium
    EK = -90.0  # Potassium

    # Maximal channel conductances (S/m2)
    gNabar = 400.0  # Sodium
    gKdbar = 400.0  # Delayed-rectifier Potassium
    gLeak = 1.0     # Non-specific leakage

    # Na+ current parameters
    Vrest_Traub = -65.  # Resting potential in Traub 1991 (mV), used as reference for m & h rates
    mshift = -6.0       # m-gate activation voltage shift, from ModelDB file (mV)
    hshift = 6.0        # h-gate activation voltage shift, from ModelDB file (mV)

    # Additional parameters
    # celsius = 35.0      # Temperature in ModelDB file (Celsius)
    celsius_Traub = 30.0  # Temperature in Traub 1991 (Celsius)
    celsius_BG = 30.0     # Temperature in Borg-Graham 1987 (Celsius)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd activation gate',
        'l': 'iKd inactivation gate'
    }

    def __new__(cls):
        cls.q10_Traub = 3**((cls.celsius - cls.celsius_Traub) / 10)
        cls.q10_BG = 3**((cls.celsius - cls.celsius_BG) / 10)

        # Compute Eleak such that iLeak cancels out the net current at resting potential
        sstates = {k: cls.steadyStates()[k](cls.Vm0) for k in cls.statesNames()}
        i_dict = cls.currents()
        del i_dict['iLeak']
        iNet = sum([cfunc(cls.Vm0, sstates) for cfunc in i_dict.values()])  # mA/m2
        cls.ELeak = cls.Vm0 + iNet / cls.gLeak  # mV
        logger.debug(f'Eleak = {cls.ELeak:.2f} mV')

        return super(SundtSegment, cls).__new__(cls)

    # ------------------------------ Gating states kinetics ------------------------------

    # iNa kinetics: adapted from Traub 1991, with 2 notable changes:
    # - Q10 correction to account for temperature adaptation from 30 to 35 degrees
    # - 65 mV voltage offset to account for Traub 1991 relative voltage definition (Vm = v - Vrest)
    # - voltage offsets in the m-gate (+6mV) and h-gate (-6mV) to shift iNa voltage dependence
    #   approximately midway between values reported for Nav1.7 and Nav1.8 currents.

    @classmethod
    def alpham(cls, Vm):
        Vm -= cls.Vrest_Traub
        Vm += cls.mshift
        return cls.q10_Traub * 0.32 * cls.vtrap((13.1 - Vm), 4) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        Vm -= cls.Vrest_Traub
        Vm += cls.mshift
        return cls.q10_Traub * 0.28 * cls.vtrap((Vm - 40.1), 5) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        Vm -= cls.Vrest_Traub
        Vm += cls.hshift
        return cls.q10_Traub * 0.128 * np.exp((17.0 - Vm) / 18) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        Vm -= cls.Vrest_Traub
        Vm += cls.hshift
        return cls.q10_Traub * 4 / (1 + np.exp((40.0 - Vm) / 5)) * 1e3  # s-1

    # iKd kinetics: using Migliore 1995 values, with Borg-Graham 1991 formalism, with:
    # - Q10 correction to account for temperature adaptation from 30 to 35 degrees

    @classmethod
    def alphan(cls, Vm):
        return cls.q10_BG * cls.alphaBG(0.03, -5, 0.4, -32., Vm) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return cls.q10_BG * cls.betaBG(0.03, -5, 0.4, -32., Vm) * 1e3  # s-1

    @classmethod
    def alphal(cls, Vm):
        return cls.q10_BG * cls.alphaBG(0.001, 2, 1., -61., Vm) * 1e3  # s-1

    @classmethod
    def betal(cls, Vm):
        return cls.q10_BG * cls.betaBG(0.001, 2, 1., -61., Vm) * 1e3  # s-1

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'l': lambda Vm, x: cls.alphal(Vm) * (1 - x['l']) - cls.betal(Vm) * x['l']
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            'l': lambda Vm: cls.alphal(Vm) / (cls.alphal(Vm) + cls.betal(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current.

            Gating formalism from Migliore 1995, using 3rd power for m
            to reproduce 1 ms AP half-width

            ..Note: inconsistency with 1991 ref: m2h vs. m3h
        '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, l, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**3 * l * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], x['l'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        ''' neuron-specific time step for fast dynamics. '''
        return super().chooseTimeStep() * 1e-2

    @staticmethod
    def getNSpikes(data):
        return detectSpikes(data, mph=-8.0e-5)[0].size
