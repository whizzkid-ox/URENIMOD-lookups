# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-27 21:24:05
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-29 19:35:12

import numpy as np
from ..core import PointNeuron, addSonicFeatures


@addSonicFeatures
class MRGNode(PointNeuron):
    ''' Mammalian myelinated fiber node.

        Reference:
        *McIntyre, C.C., Richardson, A.G., and Grill, W.M. (2002). Modeling the excitability
        of mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
        J. Neurophysiol. 87, 995–1006.*
    '''

    # Mechanism name
    name = 'MRGnode'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 2e-2  # Membrane capacitance (F/m2)
    Vm0 = -80.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 50.     # Sodium
    EK = -90.     # Potassium
    ELeak = -90.  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNafbar = 3e4   # Fast Sodium
    gNapbar = 100.  # Persistent Sodium
    gKsbar = 800.   # Slow Potassium
    gLeak = 70.     # Non-specific leakage

    # Additional parameters
    # celsius = 36.0          # Temperature (Celsius)
    celsius_Schwarz = 20.0  # Temperature in Schwarz 1995 (Celsius)
    celsius_Ks = 36.0       # Temperature used for Ks channels (unknown ref.)
    mhshift = 3.            # m and h gates voltage shift (mV)
    vtraub = -80.           # Reference voltage for the definition of the s rate constants (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNaf activation gate',
        'h': 'iNaf inactivation gate',
        'p': 'iNap activation gate',
        's': 'iKs activation gate',
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def __new__(cls):
        cls.q10_mp = 2.2**((cls.celsius - cls.celsius_Schwarz) / 10)  # from Schwarz 1987
        cls.q10_h = 2.9**((cls.celsius - cls.celsius_Schwarz) / 10)   # from Schwarz 1987
        cls.q10_s = 3.0**((cls.celsius - cls.celsius_Ks) / 10)        # from ???
        return super(MRGNode, cls).__new__(cls)

    # iNaf kinetics: adapted from Schwarz 1995, with notable changes:
    # - Q10 correction to account for temperature adaptation from 20 degrees reference
    # - 3 mV offset to m and h gates to shift voltage dependence in the hyperpolarizing direction,
    #   to increase the conduction velocity and strength-duration time constant (from McIntyre 2002)
    # - increase in tau_h to slow down inactivation (achieved through alphah?)

    @classmethod
    def alpham(cls, Vm):
        Vm += cls.mhshift
        return cls.q10_mp * 1.86 * cls.vtrap(-(Vm + 18.4), 10.3) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        Vm += cls.mhshift
        return cls.q10_mp * 0.086 * cls.vtrap(Vm + 22.7, 9.16) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        Vm += cls.mhshift
        return cls.q10_h * 0.062 * cls.vtrap(Vm + 111.0, 11.0) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        Vm += cls.mhshift
        return cls.q10_h * 2.3 / (1 + np.exp(-(Vm + 28.8) / 13.4)) * 1e3  # s-1

    # iNap kinetics: adapted from ???, with notable changes:
    # - Q10 correction to account for temperature adaptation from 20 degrees reference
    # - increase in tau_p in order to extend the duration and amplitude of the DAP.

    @classmethod
    def alphap(cls, Vm):
        return cls.q10_mp * 0.01 * cls.vtrap(-(Vm + 27.), 10.2) * 1e3  # s-1

    @classmethod
    def betap(cls, Vm):
        return cls.q10_mp * 0.00025 * cls.vtrap(Vm + 34., 10.) * 1e3  # s-1

    # iKs kinetics: adapted from ???, with notable changes:
    # - Q10 correction to account for temperature adaptation from 36 degrees reference
    # - increase in tau_s in order to to create an AHP that matches experimental records.

    @classmethod
    def alphas(cls, Vm):
        Vm -= cls.vtraub
        return cls.q10_s * 0.3 / (1 + np.exp(-(Vm - 27.) / 5.)) * 1e3  # s-1

    @classmethod
    def betas(cls, Vm):
        Vm -= cls.vtraub
        return cls.q10_s * 0.03 / (1 + np.exp(-(Vm + 10.) / 1.)) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'p': lambda Vm, x: cls.alphap(Vm) * (1 - x['p']) - cls.betap(Vm) * x['p'],
            's': lambda Vm, x: cls.alphas(Vm) * (1 - x['s']) - cls.betas(Vm) * x['s'],
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'p': lambda Vm: cls.alphap(Vm) / (cls.alphap(Vm) + cls.betap(Vm)),
            's': lambda Vm: cls.alphas(Vm) / (cls.alphas(Vm) + cls.betas(Vm)),
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNaf(cls, m, h, Vm):
        ''' fast Sodium current. '''
        return cls.gNafbar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iNap(cls, p, Vm):
        ''' persistent Sodium current. '''
        return cls.gNapbar * p**3 * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKs(cls, s, Vm):
        ''' slow Potassium current '''
        return cls.gKsbar * s * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNaf': lambda Vm, x: cls.iNaf(x['m'], x['h'], Vm),
            'iNap': lambda Vm, x: cls.iNap(x['p'], Vm),
            'iKs': lambda Vm, x: cls.iKs(x['s'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        ''' neuron-specific time step for fast dynamics. '''
        return super().chooseTimeStep() * 1e-2
