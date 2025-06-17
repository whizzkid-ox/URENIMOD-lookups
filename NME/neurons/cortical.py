# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:19:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 16:16:57

import numpy as np
from ..core import PointNeuron, addSonicFeatures


class Cortical(PointNeuron):
    ''' Generic cortical neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)

    # Reversal potentials (mV)
    ENa = 50.0   # Sodium
    EK = -90.0   # Potassium
    ECa = 120.0  # Calcium

    # Additional parameters
    # celsius = 36.0  # Temperature in Pospischil 2008 (Celsius)

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return 0.32 * cls.vtrap(13 - (Vm - cls.VT), 4) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return 0.28 * cls.vtrap((Vm - cls.VT) - 40, 5) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return 0.128 * np.exp(-((Vm - cls.VT) - 17) / 18) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return 4 / (1 + np.exp(-((Vm - cls.VT) - 40) / 5)) * 1e3  # s-1

    @classmethod
    def alphan(cls, Vm):
        return 0.032 * cls.vtrap(15 - (Vm - cls.VT), 5) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return 0.5 * np.exp(-((Vm - cls.VT) - 10) / 40) * 1e3  # s-1

    @staticmethod
    def pinf(Vm):
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))

    @classmethod
    def taup(cls, Vm):
        return cls.TauMax / (3.3 * np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20))  # s

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'p': lambda Vm, x: (cls.pinf(Vm) - x['p']) / cls.taup(Vm)
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            'p': lambda Vm: cls.pinf(Vm)
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
    def iM(cls, p, Vm):
        ''' slow non-inactivating Potassium current '''
        return cls.gMbar * p * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iM': lambda Vm, x: cls.iM(x['p'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }


@addSonicFeatures
class CorticalRS(Cortical):
    ''' Cortical regular spiking neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Neuron name
    name = 'RS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.9  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70.3  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 560.0  # Sodium
    gKdbar = 60.0   # Delayed-rectifier Potassium
    gMbar = 0.75    # Slow non-inactivating Potassium
    gLeak = 0.205   # Non-specific leakage

    # Additional parameters
    VT = -56.2       # Spike threshold adjustment parameter (mV)
    TauMax = 0.608   # Max. adaptation decay of slow non-inactivating Potassium current (s)
    area = 11.84e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate'
    }


@addSonicFeatures
class CorticalFS(Cortical):
    ''' Cortical fast-spiking neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Neuron name
    name = 'FS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.4  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70.4  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 580.0  # Sodium
    gKdbar = 39.0   # Delayed-rectifier Potassium
    gMbar = 0.787   # Slow non-inactivating Potassium
    gLeak = 0.38    # Non-specific leakage

    # Additional parameters
    VT = -57.9       # Spike threshold adjustment parameter (mV)
    TauMax = 0.502   # Max. adaptation decay of slow non-inactivating Potassium current (s)
    area = 10.17e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate'
    }


@addSonicFeatures
class CorticalLTS(Cortical):
    ''' Cortical low-threshold spiking neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Huguenard, J.R., and McCormick, D.A. (1992). Simulation of the currents involved in
        rhythmic oscillations in thalamic relay neurons. J. Neurophysiol. 68, 1373–1383.*

    '''

    # Neuron name
    name = 'LTS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -54.0  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -50.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 500.0  # Sodium
    gKdbar = 40.0   # Delayed-rectifier Potassium
    gMbar = 0.28    # Slow non-inactivating Potassium
    gCaTbar = 4.0   # Low-threshold Calcium
    gLeak = 0.19    # Non-specific leakage

    # Additional parameters
    VT = -50.0       # Spike threshold adjustment parameter (mV)
    TauMax = 4.0     # Max. adaptation decay of slow non-inactivating Potassium current (s)
    Vx = -7.0        # Voltage-dependence uniform shift factor at 36°C (mV)
    area = 25.00e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def sinf(cls, Vm):
        return 1.0 / (1.0 + np.exp(-(Vm + cls.Vx + 57.0) / 6.2))

    @classmethod
    def taus(cls, Vm):
        x = np.exp(-(Vm + cls.Vx + 132.0) / 16.7) + np.exp((Vm + cls.Vx + 16.8) / 18.2)
        return 1.0 / 3.7 * (0.612 + 1.0 / x) * 1e-3  # s

    @classmethod
    def uinf(cls, Vm):
        return 1.0 / (1.0 + np.exp((Vm + cls.Vx + 81.0) / 4.0))

    @classmethod
    def tauu(cls, Vm):
        if Vm + cls.Vx < -80.0:
            return 1.0 / 3.7 * np.exp((Vm + cls.Vx + 467.0) / 66.6) * 1e-3  # s
        else:
            return 1.0 / 3.7 * (np.exp(-(Vm + cls.Vx + 22) / 10.5) + 28.0) * 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {**super().derStates(), **{
            's': lambda Vm, x: (cls.sinf(Vm) - x['s']) / cls.taus(Vm),
            'u': lambda Vm, x: (cls.uinf(Vm) - x['u']) / cls.tauu(Vm)
        }}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {**super().steadyStates(), **{
            's': lambda Vm: cls.sinf(Vm),
            'u': lambda Vm: cls.uinf(Vm)
        }}

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iCaT(cls, s, u, Vm):
        ''' low-threshold (T-type) Calcium current '''
        return cls.gCaTbar * s**2 * u * (Vm - cls.ECa)  # mA/m2

    @classmethod
    def currents(cls):
        return {**super().currents(), **{
            'iCaT': lambda Vm, x: cls.iCaT(x['s'], x['u'], Vm)
        }}


@addSonicFeatures
class CorticalIB(Cortical):
    ''' Cortical intrinsically bursting neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Reuveni, I., Friedman, A., Amitai, Y., and Gutnick, M.J. (1993). Stepwise
        repolarization from Ca2+ plateaus in neocortical pyramidal cells: evidence
        for nonhomogeneous distribution of HVA Ca2+ channels in dendrites.
        J. Neurosci. 13, 4609–4621.*
    '''

    # Neuron name
    name = 'IB'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.4  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 500   # Sodium
    gKdbar = 50    # Delayed-rectifier Potassium
    gMbar = 0.3    # Slow non-inactivating Potassium
    gCaLbar = 1.0  # High-threshold Calcium
    gLeak = 0.1    # Non-specific leakage

    # Additional parameters
    VT = -56.2       # Spike threshold adjustment parameter (mV)
    TauMax = 0.608   # Max. adaptation decay of slow non-inactivating Potassium current (s)
    area = 28.95e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate',
        'q': 'iCaL activation gate',
        'r': 'iCaL inactivation gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alphaq(cls, Vm):
        return 0.055 * cls.vtrap(-(Vm + 27), 3.8) * 1e3  # s-1

    @staticmethod
    def betaq(Vm):
        return 0.94 * np.exp(-(Vm + 75) / 17) * 1e3  # s-1

    @staticmethod
    def alphar(Vm):
        return 0.000457 * np.exp(-(Vm + 13) / 50) * 1e3  # s-1

    @staticmethod
    def betar(Vm):
        return 0.0065 / (np.exp(-(Vm + 15) / 28) + 1) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {**super().derStates(), **{
            'q': lambda Vm, x: cls.alphaq(Vm) * (1 - x['q']) - cls.betaq(Vm) * x['q'],
            'r': lambda Vm, x: cls.alphar(Vm) * (1 - x['r']) - cls.betar(Vm) * x['r']
        }}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {**super().steadyStates(), **{
            'q': lambda Vm: cls.alphaq(Vm) / (cls.alphaq(Vm) + cls.betaq(Vm)),
            'r': lambda Vm: cls.alphar(Vm) / (cls.alphar(Vm) + cls.betar(Vm))
        }}

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iCaL(cls, q, r, Vm):
        ''' high-threshold (L-type) Calcium current '''
        return cls.gCaLbar * q**2 * r * (Vm - cls.ECa)  # mA/m2

    @classmethod
    def currents(cls):
        return {**super().currents(), **{
            'iCaL': lambda Vm, x: cls.iCaL(x['q'], x['r'], Vm)
        }}
