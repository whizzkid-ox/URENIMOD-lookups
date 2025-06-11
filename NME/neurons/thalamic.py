# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 16:17:08

import numpy as np
from ..core import PointNeuron, addSonicFeatures
from ..constants import Z_Ca


class Thalamic(PointNeuron):
    ''' Generic thalamic neuron

        Reference:
        *Plaksin, M., Kimmel, E., and Shoham, S. (2016). Cell-Type-Selective Effects of
        Intramembrane Cavitation as a Unifying Theoretical Framework for Ultrasonic
        Neuromodulation. eNeuro 3.*
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

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            's': lambda Vm, x: (cls.sinf(Vm) - x['s']) / cls.taus(Vm),
            'u': lambda Vm, x: (cls.uinf(Vm) - x['u']) / cls.tauu(Vm)
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            's': lambda Vm: cls.sinf(Vm),
            'u': lambda Vm: cls.uinf(Vm)
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**4 * (Vm - cls.EK)

    @classmethod
    def iCaT(cls, s, u, Vm):
        ''' low-threshold (Ts-type) Calcium current '''
        return cls.gCaTbar * s**2 * u * (Vm - cls.ECa)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, states: cls.iNa(states['m'], states['h'], Vm),
            'iKd': lambda Vm, states: cls.iKd(states['n'], Vm),
            'iCaT': lambda Vm, states: cls.iCaT(states['s'], states['u'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }


@addSonicFeatures
class ThalamicRE(Thalamic):
    ''' Thalamic reticular neuron

        References:
        *Destexhe, A., Contreras, D., Steriade, M., Sejnowski, T.J., and Huguenard, J.R. (1996).
        In vivo, in vitro, and computational analysis of dendritic calcium currents in thalamic
        reticular neurons. J. Neurosci. 16, 169–185.*

        *Huguenard, J.R., and Prince, D.A. (1992). A novel T-type current underlies prolonged
        Ca(2+)-dependent burst firing in GABAergic neurons of rat thalamic reticular nucleus.
        J. Neurosci. 12, 3804–3817.*

    '''

    # Neuron name
    name = 'RE'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -89.5  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -90.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 2000.0  # Sodium
    gKdbar = 200.0   # Delayed-rectifier Potassium
    gCaTbar = 30.0   # Low-threshold Calcium
    gLeak = 0.5      # Non-specific leakage

    # Additional parameters
    VT = -67.0       # Spike threshold adjustment parameter (mV)
    area = 14.00e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def sinf(Vm):
        return 1.0 / (1.0 + np.exp(-(Vm + 52.0) / 7.4))

    @staticmethod
    def taus(Vm):
        return (1 + 0.33 / (np.exp((Vm + 27.0) / 10.0) + np.exp(-(Vm + 102.0) / 15.0))) * 1e-3  # s

    @staticmethod
    def uinf(Vm):
        return 1.0 / (1.0 + np.exp((Vm + 80.0) / 5.0))

    @staticmethod
    def tauu(Vm):
        return (28.3 + 0.33 / (
            np.exp((Vm + 48.0) / 4.0) + np.exp(-(Vm + 407.0) / 50.0))) * 1e-3  # s


@addSonicFeatures
class ThalamoCortical(Thalamic):
    ''' Thalamo-cortical neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
        Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
        classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
        underlying synchronized oscillations and propagating waves in a model of ferret
        thalamic slices. J. Neurophysiol. 76, 2049–2070.*

        Model of Ca2+ buffering and contribution from iCaT derived from:
        *McCormick, D.A., and Huguenard, J.R. (1992). A model of the electrophysiological
        properties of thalamocortical relay neurons. J. Neurophysiol. 68, 1384–1400.*
    '''

    # Neuron name
    name = 'TC'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    # Vm0 = -63.4  # Membrane potential (mV)
    Vm0 = -61.93  # Membrane potential (mV)

    # Reversal potentials (mV)
    EH = -40.0     # Mixed cationic current
    ELeak = -70.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 900.0  # Sodium
    gKdbar = 100.0  # Delayed-rectifier Potassium
    gCaTbar = 20.0  # Low-threshold Calcium
    gKLeak = 0.138  # Leakage Potassium
    gHbar = 0.175   # Mixed cationic current
    gLeak = 0.1     # Non-specific leakage

    # Additional parameters
    VT = -52.0       # Spike threshold adjustment parameter (mV)
    Vx = 0.0         # Voltage-dependence uniform shift factor at 36°C (mV)
    taur_Cai = 5e-3  # decay time constant for intracellular Ca2+ dissolution (s)
    Cai_min = 50e-9  # minimal intracellular Calcium concentration (M)
    deff = 100e-9    # effective depth beneath membrane for intracellular [Ca2+] calculation
    nCa = 4          # number of Calcium binding sites on regulating factor
    k1 = 2.5e22      # intracellular Ca2+ regulation factor (M-4 s-1)
    k2 = 0.4         # intracellular Ca2+ regulation factor (s-1)
    k3 = 100.0       # intracellular Ca2+ regulation factor (s-1)
    k4 = 1.0         # intracellular Ca2+ regulation factor (s-1)
    area = 29.00e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate',
        'Cai': 'submembrane Ca2+ concentration (M)',
        'P0': 'proportion of unbound iH regulating factor',
        'O': 'iH gate open state',
        'C': 'iH gate closed state',
    }

    def __new__(cls):
        cls.current_to_molar_rate_Ca = cls.currentToConcentrationRate(Z_Ca, cls.deff)
        return super(ThalamoCortical, cls).__new__(cls)

    @staticmethod
    def OL(O, C):
        ''' O-gate locked-open probability '''
        return 1 - O - C

    @property
    def pltScheme(self):
        pltscheme = super().pltScheme
        pltscheme['i_{H}\\ kin.'] = ['O', 'OL', 'P0']
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright), **{
            'Cai': {
                'desc': 'sumbmembrane Ca2+ concentration',
                'label': '[Ca^{2+}]_i',
                'unit': 'uM',
                'factor': 1e6
            },
            'OL': {
                'desc': 'iH O-gate locked-opening',
                'label': 'O_L',
                'bounds': (-0.1, 1.1),
                'func': f'OL({wrapleft}O{wrapright}, {wrapleft}C{wrapright})'
            },
            'P0': {
                'desc': 'iH regulating factor activation',
                'label': 'P_0',
                'bounds': (-0.1, 1.1)
            }
        }}

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
            return 1 / 3.7 * (np.exp(-(Vm + cls.Vx + 22) / 10.5) + 28.0) * 1e-3  # s

    @staticmethod
    def oinf(Vm):
        return 1.0 / (1.0 + np.exp((Vm + 75.0) / 5.5))

    @staticmethod
    def tauo(Vm):
        return 1 / (np.exp(-14.59 - 0.086 * Vm) + np.exp(-1.87 + 0.0701 * Vm)) * 1e-3  # s

    @classmethod
    def alphao(cls, Vm):
        return cls.oinf(Vm) / cls.tauo(Vm)  # s-1

    @classmethod
    def betao(cls, Vm):
        return (1 - cls.oinf(Vm)) / cls.tauo(Vm)  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {**super().derStates(), **{
            'Cai': lambda Vm, x: ((cls.Cai_min - x['Cai']) / cls.taur_Cai -
                                  cls.current_to_molar_rate_Ca * cls.iCaT(x['s'], x['u'], Vm)),  # M/s
            'P0': lambda _, x: cls.k2 * (1 - x['P0']) - cls.k1 * x['P0'] * x['Cai']**cls.nCa,
            'O': lambda Vm, x: (cls.alphao(Vm) * x['C'] - cls.betao(Vm) * x['O'] -
                                cls.k3 * x['O'] * (1 - x['P0']) + cls.k4 * (1 - x['O'] - x['C'])),
            'C': lambda Vm, x: cls.betao(Vm) * x['O'] - cls.alphao(Vm) * x['C'],
        }}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        lambda_dict = super().steadyStates()
        lambda_dict['Cai'] = lambda Vm: (cls.Cai_min - cls.taur_Cai * cls.current_to_molar_rate_Ca *
                                         cls.iCaT(cls.sinf(Vm), cls.uinf(Vm), Vm))  # M
        lambda_dict['P0'] = lambda Vm: cls.k2 / (cls.k2 + cls.k1 * lambda_dict['Cai'](Vm)**cls.nCa)
        lambda_dict['O'] = lambda Vm: (cls.k4 / (cls.k3 * (1 - lambda_dict['P0'](Vm)) +
                                       cls.k4 * (1 + cls.betao(Vm) / cls.alphao(Vm))))
        lambda_dict['C'] = lambda Vm: cls.betao(Vm) / cls.alphao(Vm) * lambda_dict['O'](Vm)
        return lambda_dict

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iKLeak(cls, Vm):
        ''' Potassium leakage current '''
        return cls.gKLeak * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iH(cls, O, C, Vm):
        ''' outward mixed cationic current '''
        return cls.gHbar * (O + 2 * cls.OL(O, C)) * (Vm - cls.EH)  # mA/m2

    @classmethod
    def currents(cls):
        return {**super().currents(), **{
            'iKLeak': lambda Vm, x: cls.iKLeak(Vm),
            'iH': lambda Vm, x: cls.iH(x['O'], x['C'], Vm)
        }}


# class ThalamoCorticalTweak70(ThalamoCortical):

#     name = 'TCtweak70'
#     Vm0 = -70.0
#     gKLeak = 0.377


# class ThalamoCorticalTweak50(ThalamoCortical):

#     name = 'TCtweak50'
#     Vm0 = -50.0
#     gCaTbar = 140.0  # or gHbar = 51.5
