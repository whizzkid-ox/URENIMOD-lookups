# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-11-29 16:56:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 16:16:10

import numpy as np
from ..core import PointNeuron, addSonicFeatures
from ..constants import FARADAY, Z_Ca
from ..utils import findModifiedEq


@addSonicFeatures
class OtsukaSTN(PointNeuron):
    ''' Sub-thalamic nucleus neuron

        References:
        *Otsuka, T., Abe, T., Tsukagawa, T., and Song, W.-J. (2004). Conductance-Based Model
        of the Voltage-Dependent Generation of a Plateau Potential in Subthalamic Neurons.
        Journal of Neurophysiology 92, 255–264.*

        *Tarnaud, T., Joseph, W., Martens, L., and Tanghe, E. (2018). Computational Modeling
        of Ultrasonic Subthalamic Nucleus Stimulation. IEEE Trans Biomed Eng.*
    '''

    # Neuron name
    name = 'STN'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -58.0  # Membrane potential (mV)
    Cai0 = 5e-9  # Intracellular Calcium concentration (M)

    # Reversal potentials (mV)
    ENa = 60.0     # Sodium
    EK = -90.0     # Potassium
    ELeak = -60.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 490.0   # Sodium
    gLeak = 3.5      # Non-specific leakage
    gKdbar = 570.0   # Delayed-rectifier Potassium
    gCaTbar = 50.0   # Low-threshold Calcium
    gCaLbar = 150.0  # High-threshold Calcium
    gAbar = 50.0     # A-type Potassium
    gKCabar = 10.0   # Calcium-dependent Potassium

    # Physical constants
    # celsius = 33.0   # Temperature (Celsius)

    # Calcium dynamics
    Cao = 2e-3         # extracellular Calcium concentration (M)
    taur_Cai = 0.5e-3  # decay time constant for intracellular Ca2+ dissolution (s)

    # Fast Na current m-gate
    thetax_m = -40   # mV
    kx_m = -8        # mV
    tau0_m = 0.2e-3  # s
    tau1_m = 3e-3    # s
    thetaT_m = -53   # mV
    sigmaT_m = -0.7  # mV

    # Fast Na current h-gate
    thetax_h = -45.5  # mV
    kx_h = 6.4        # mV
    tau0_h = 0e-3     # s
    tau1_h = 24.5e-3  # s
    thetaT1_h = -50   # mV
    thetaT2_h = -50   # mV
    sigmaT1_h = -15   # mV
    sigmaT2_h = 16    # mV

    # Delayed rectifier K+ current n-gate
    thetax_n = -41   # mV
    kx_n = -14       # mV
    tau0_n = 0e-3    # s
    tau1_n = 11e-3   # s
    thetaT1_n = -40  # mV
    thetaT2_n = -40  # mV
    sigmaT1_n = -40  # mV
    sigmaT2_n = 50   # mV

    # T-type Ca2+ current p-gate
    thetax_p = -56    # mV
    kx_p = -6.7       # mV
    tau0_p = 5e-3     # s
    tau1_p = 0.33e-3  # s
    thetaT1_p = -27   # mV
    thetaT2_p = -102  # mV
    sigmaT1_p = -10   # mV
    sigmaT2_p = 15    # mV

    # T-type Ca2+ current q-gate
    thetax_q = -85   # mV
    kx_q = 5.8       # mV
    tau0_q = 0e-3    # s
    tau1_q = 400e-3  # s
    thetaT1_q = -50  # mV
    thetaT2_q = -50  # mV
    sigmaT1_q = -15  # mV
    sigmaT2_q = 16   # mV

    # L-type Ca2+ current c-gate
    thetax_c = -30.6  # mV
    kx_c = -5         # mV
    tau0_c = 45e-3    # s
    tau1_c = 10e-3    # s
    thetaT1_c = -27   # mV
    thetaT2_c = -50   # mV
    sigmaT1_c = -20   # mV
    sigmaT2_c = 15    # mV

    # L-type Ca2+ current d1-gate
    thetax_d1 = -60   # mV
    kx_d1 = 7.5       # mV
    tau0_d1 = 400e-3  # s
    tau1_d1 = 500e-3  # s
    thetaT1_d1 = -40  # mV
    thetaT2_d1 = -20  # mV
    sigmaT1_d1 = -15  # mV
    sigmaT2_d1 = 20   # mV

    # L-type Ca2+ current d2-gate
    thetax_d2 = 0.1e-6  # M
    kx_d2 = 0.02e-6     # M
    tau_d2 = 130e-3     # s

    # A-type K+ current a-gate
    thetax_a = -45   # mV
    kx_a = -14.7     # mV
    tau0_a = 1e-3    # s
    tau1_a = 1e-3    # s
    thetaT_a = -40   # mV
    sigmaT_a = -0.5  # mV

    # A-type K+ current b-gate
    thetax_b = -90   # mV
    kx_b = 7.5       # mV
    tau0_b = 0e-3    # s
    tau1_b = 200e-3  # s
    thetaT1_b = -60  # mV
    thetaT2_b = -40  # mV
    sigmaT1_b = -30  # mV
    sigmaT2_b = 10   # mV

    # Ca2+-activated K+ current r-gate
    thetax_r = 0.17e-6  # M
    kx_r = -0.08e-6     # M
    tau_r = 2e-3        # s

    area = 2.86e-9  # Cell membrane area (m2)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'a': 'iA activation gate',
        'b': 'iA inactivation gate',
        'p': 'iCaT activation gate',
        'q': 'iCaT inactivation gate',
        'c': 'iCaL activation gate',
        'd1': 'iCaL inactivation gate 1',
        'd2': 'iCaL inactivation gate 2',
        'r': 'iCaK gate',
        'Cai': 'submembrane Calcium concentration (M)'
    }

    def __new__(cls):
        cls.deff = cls.getEffectiveDepth(cls.Cai0, cls.Vm0)  # m
        cls.current_to_molar_rate_Ca = cls.currentToConcentrationRate(Z_Ca, cls.deff)
        return super(OtsukaSTN, cls).__new__(cls)

    @property
    def pltScheme(self):
        pltscheme = super().pltScheme
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars['Cai'] = {
            'desc': 'submembrane Ca2+ concentration',
            'label': '[Ca^{2+}]_i',
            'unit': 'uM',
            'factor': 1e6
        }
        return pltvars

    @classmethod
    def titrationFunc(cls, *args, **kwargs):
        return cls.isSilenced(*args, **kwargs)

    @classmethod
    def getEffectiveDepth(cls, Cai, Vm):
        ''' Compute effective depth that matches a given membrane potential
            and intracellular Calcium concentration.

            :return: effective depth (m)
        '''
        iCaT = cls.iCaT(cls.pinf(Vm), cls.qinf(Vm), Vm, Cai)  # mA/m2
        iCaL = cls.iCaL(cls.cinf(Vm), cls.d1inf(Vm), cls.d2inf(Cai), Vm, Cai)  # mA/m2
        return -(iCaT + iCaL) / (Z_Ca * FARADAY * Cai / cls.taur_Cai) * 1e-6  # m

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def _xinf(var, theta, k):
        ''' Generic function computing the steady-state opening of a
            particular channel gate at a given voltage or ion concentration.

            :param var: membrane potential (mV) or ion concentration (mM)
            :param theta: half-(in)activation voltage or concentration (mV or mM)
            :param k: slope parameter of (in)activation function (mV or mM)
            :return: steady-state opening (-)
        '''
        return 1 / (1 + np.exp((var - theta) / k))

    @classmethod
    def ainf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_a, cls.kx_a)

    @classmethod
    def binf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_b, cls.kx_b)

    @classmethod
    def cinf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_c, cls.kx_c)

    @classmethod
    def d1inf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_d1, cls.kx_d1)

    @classmethod
    def d2inf(cls, Cai):
        return cls._xinf(Cai, cls.thetax_d2, cls.kx_d2)

    @classmethod
    def minf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_m, cls.kx_m)

    @classmethod
    def hinf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_h, cls.kx_h)

    @classmethod
    def ninf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_n, cls.kx_n)

    @classmethod
    def pinf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_p, cls.kx_p)

    @classmethod
    def qinf(cls, Vm):
        return cls._xinf(Vm, cls.thetax_q, cls.kx_q)

    @classmethod
    def rinf(cls, Cai):
        return cls._xinf(Cai, cls.thetax_r, cls.kx_r)

    @staticmethod
    def _taux1(Vm, theta, sigma, tau0, tau1):
        ''' Generic function computing the voltage-dependent, activation/inactivation time constant
            of a particular ion channel at a given voltage (first variant).

            :param Vm: membrane potential (mV)
            :param theta: voltage at which (in)activation time constant is half-maximal (mV)
            :param sigma: slope parameter of (in)activation time constant function (mV)
            :param tau0: minimal time constant (s)
            :param tau1: modulated time constant (s)
            :return: (in)activation time constant (s)
        '''
        return tau0 + tau1 / (1 + np.exp(-(Vm - theta) / sigma))

    @classmethod
    def taua(cls, Vm):
        return cls._taux1(Vm, cls.thetaT_a, cls.sigmaT_a, cls.tau0_a, cls.tau1_a)

    @classmethod
    def taum(cls, Vm):
        return cls._taux1(Vm, cls.thetaT_m, cls.sigmaT_m, cls.tau0_m, cls.tau1_m)

    @staticmethod
    def _taux2(Vm, theta1, theta2, sigma1, sigma2, tau0, tau1):
        ''' Generic function computing the voltage-dependent, activation/inactivation time constant
            of a particular ion channel at a given voltage (second variant).

            :param Vm: membrane potential (mV)
            :param theta: voltage at which (in)activation time constant is half-maximal (mV)
            :param sigma: slope parameter of (in)activation time constant function (mV)
            :param tau0: minimal time constant (s)
            :param tau1: modulated time constant (s)
            :return: (in)activation time constant (s)
        '''
        return tau0 + tau1 / (np.exp(-(Vm - theta1) / sigma1) + np.exp(-(Vm - theta2) / sigma2))

    @classmethod
    def taub(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_b, cls.thetaT2_b, cls.sigmaT1_b, cls.sigmaT2_b,
                          cls.tau0_b, cls.tau1_b)

    @classmethod
    def tauc(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_c, cls.thetaT2_c, cls.sigmaT1_c, cls.sigmaT2_c,
                          cls.tau0_c, cls.tau1_c)

    @classmethod
    def taud1(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_d1, cls.thetaT2_d1, cls.sigmaT1_d1, cls.sigmaT2_d1,
                          cls.tau0_d1, cls.tau1_d1)

    @classmethod
    def tauh(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_h, cls.thetaT2_h, cls.sigmaT1_h, cls.sigmaT2_h,
                          cls.tau0_h, cls.tau1_h)

    @classmethod
    def taun(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_n, cls.thetaT2_n, cls.sigmaT1_n, cls.sigmaT2_n,
                          cls.tau0_n, cls.tau1_n)

    @classmethod
    def taup(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_p, cls.thetaT2_p, cls.sigmaT1_p, cls.sigmaT2_p,
                          cls.tau0_p, cls.tau1_p)

    @classmethod
    def tauq(cls, Vm):
        return cls._taux2(Vm, cls.thetaT1_q, cls.thetaT2_q, cls.sigmaT1_q, cls.sigmaT2_q,
                          cls.tau0_q, cls.tau1_q)

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derCai(cls, p, q, c, d1, d2, Cai, Vm):
        iCa_tot = cls.iCaT(p, q, Vm, Cai) + cls.iCaL(c, d1, d2, Vm, Cai)
        return - cls.current_to_molar_rate_Ca * iCa_tot - Cai / cls.taur_Cai  # M/s

    @classmethod
    def derStates(cls):
        return {
            'a': lambda Vm, x: (cls.ainf(Vm) - x['a']) / cls.taua(Vm),
            'b': lambda Vm, x: (cls.binf(Vm) - x['b']) / cls.taub(Vm),
            'c': lambda Vm, x: (cls.cinf(Vm) - x['c']) / cls.tauc(Vm),
            'd1': lambda Vm, x: (cls.d1inf(Vm) - x['d1']) / cls.taud1(Vm),
            'd2': lambda Vm, x: (cls.d2inf(x['Cai']) - x['d2']) / cls.tau_d2,
            'm': lambda Vm, x: (cls.minf(Vm) - x['m']) / cls.taum(Vm),
            'h': lambda Vm, x: (cls.hinf(Vm) - x['h']) / cls.tauh(Vm),
            'n': lambda Vm, x: (cls.ninf(Vm) - x['n']) / cls.taun(Vm),
            'p': lambda Vm, x: (cls.pinf(Vm) - x['p']) / cls.taup(Vm),
            'q': lambda Vm, x: (cls.qinf(Vm) - x['q']) / cls.tauq(Vm),
            'r': lambda Vm, x: (cls.rinf(x['Cai']) - x['r']) / cls.tau_r,
            'Cai': lambda Vm, x: cls.derCai(x['p'], x['q'], x['c'], x['d1'], x['d2'], x['Cai'], Vm)
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def Caiinf(cls, p, q, c, d1, Vm):
        ''' Steady-state intracellular Calcium concentration '''
        return findModifiedEq(
            cls.Cai0,
            lambda Cai, p, q, c, d1, Vm: cls.derCai(p, q, c, d1, cls.d2inf(Cai), Cai, Vm),
            p, q, c, d1, Vm)

    @classmethod
    def steadyStates(cls):
        lambda_dict = {
            'a': lambda Vm: cls.ainf(Vm),
            'b': lambda Vm: cls.binf(Vm),
            'c': lambda Vm: cls.cinf(Vm),
            'd1': lambda Vm: cls.d1inf(Vm),
            'm': lambda Vm: cls.minf(Vm),
            'h': lambda Vm: cls.hinf(Vm),
            'n': lambda Vm: cls.ninf(Vm),
            'p': lambda Vm: cls.pinf(Vm),
            'q': lambda Vm: cls.qinf(Vm),
        }
        lambda_dict['Cai'] = lambda Vm: cls.Caiinf(
            lambda_dict['p'](Vm),
            lambda_dict['q'](Vm),
            lambda_dict['c'](Vm),
            lambda_dict['d1'](Vm),
            Vm)
        lambda_dict['d2'] = lambda Vm: cls.d2inf(lambda_dict['Cai'](Vm))
        lambda_dict['r'] = lambda Vm: cls.rinf(lambda_dict['Cai'](Vm))
        return lambda_dict

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
    def iA(cls, a, b, Vm):
        ''' A-type Potassium current '''
        return cls.gAbar * a**2 * b * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iCaT(cls, p, q, Vm, Cai):
        ''' low-threshold (T-type) Calcium current '''
        return cls.gCaTbar * p**2 * q * (Vm - cls.nernst(Z_Ca, Cai, cls.Cao, cls.T))  # mA/m2

    @classmethod
    def iCaL(cls, c, d1, d2, Vm, Cai):
        ''' high-threshold (L-type) Calcium current '''
        return cls.gCaLbar * c**2 * d1 * d2 * (Vm - cls.nernst(Z_Ca, Cai, cls.Cao, cls.T))  # mA/m2

    @classmethod
    def iKCa(cls, r, Vm):
        ''' Calcium-activated Potassium current '''
        return cls.gKCabar * r**2 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iA': lambda Vm, x: cls.iA(x['a'], x['b'], Vm),
            'iCaT': lambda Vm, x: cls.iCaT(x['p'], x['q'], Vm, x['Cai']),
            'iCaL': lambda Vm, x: cls.iCaL(x['c'], x['d1'], x['d2'], Vm, x['Cai']),
            'iKCa': lambda Vm, x: cls.iKCa(x['r'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    # ------------------------------ Other methods ------------------------------

    @staticmethod
    def getLowIntensities():
        ''' Return an array of acoustic intensities (W/m2) used to study the STN neuron in
            Tarnaud, T., Joseph, W., Martens, L., and Tanghe, E. (2018). Computational Modeling
            of Ultrasonic Subthalamic Nucleus Stimulation. IEEE Trans Biomed Eng.
        '''
        return np.hstack((
            np.arange(10, 101, 10),
            np.arange(101, 131, 1),
            np.array([140])
        ))  # W/m2
