# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-31 18:15:43

from functools import partialmethod
import numpy as np

from ..core import PointNeuron, addSonicFeatures
from ..constants import FARADAY, Rg, Z_Na, Z_Ca


@addSonicFeatures
class LeechTouch(PointNeuron):
    ''' Leech touch sensory neuron

        Reference:
        *Cataldo, E., Brunelli, M., Byrne, J.H., Av-Ron, E., Cai, Y., and Baxter, D.A. (2005).
        Computational model of touch sensory cells (T Cells) of the leech: role of the
        afterhyperpolarization (AHP) in activity-dependent conduction failure.
        J Comput Neurosci 18, 5–24.*
    '''

    # Neuron name
    name = 'LeechT'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)
    Vm0 = -53.58  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 45.0        # Sodium
    EK = -62.0        # Potassium
    ECa = 60.0        # Calcium
    ELeak = -48.0     # Non-specific leakage
    EPumpNa = -300.0  # Sodium pump

    # Maximal channel conductances (S/m2)
    gNabar = 3500.0  # Sodium
    gKdbar = 900.0   # Delayed-rectifier Potassium
    gCabar = 20.0    # Calcium
    gKCabar = 236.0  # Calcium-dependent Potassium
    gLeak = 1.0      # Non-specific leakage
    gPumpNa = 20.0   # Sodium pump

    # Activation time constants (s)
    taum = 0.1e-3  # Sodium
    taus = 0.6e-3  # Calcium

    # Original conversion constants from inward ionic current (nA) to build-up of
    # intracellular ion concentration (arb.)
    K_Na_original = 0.016  # iNa to intracellular [Na+]
    K_Ca_original = 0.1    # iCa to intracellular [Ca2+]

    # Constants needed to convert K from original model (soma compartment)
    # to current model (point-neuron)
    surface = 6434.0e-12  # surface of cell assumed as a single soma (m2)
    curr_factor = 1e6     # mA to nA

    # Time constants for the removal of ions from intracellular pools (s)
    taur_Na = 16.0  # Sodium
    taur_Ca = 1.25  # Calcium

    # Time constants for the PumpNa and KCa currents activation
    # from specific intracellular ions (s)
    taua_PumpNa = 0.1  # PumpNa current activation from intracellular Na+
    taua_KCa = 0.01    # KCa current activation from intracellular Ca2+

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'Nai': 'submembrane Na+ concentration (arbitrary unit)',
        'ANa': 'Na+ dependent iPumpNa gate',
        'Cai': 'submembrane Ca2+ concentration (arbitrary unit)',
        'ACa': 'Ca2+ dependent iKCa gate'
    }

    def __new__(cls):
        cls.K_Na = cls.K_Na_original * cls.surface * cls.curr_factor
        cls.K_Ca = cls.K_Ca_original * cls.surface * cls.curr_factor
        return super(LeechTouch, cls).__new__(cls)

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def _xinf(Vm, halfmax, slope, power):
        ''' Generic function computing the steady-state open-probability of a
            particular ion channel gate at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: half-activation voltage (mV)
            :param slope: slope parameter of activation function (mV)
            :param power: power exponent multiplying the exponential expression (integer)
            :return: steady-state open-probability (-)
        '''
        return 1 / (1 + np.exp((Vm - halfmax) / slope))**power

    @staticmethod
    def _taux(Vm, halfmax, slope, tauMax, tauMin):
        ''' Generic function computing the voltage-dependent, adaptation time constant
            of a particular ion channel gate at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: voltage at which adaptation time constant is half-maximal (mV)
            :param slope: slope parameter of adaptation time constant function (mV)
            :return: adptation time constant (s)
        '''
        return (tauMax - tauMin) / (1 + np.exp((Vm - halfmax) / slope)) + tauMin

    @staticmethod
    def _derCion(Cion, Iion, Kion, tau):
        ''' Generic function computing the time derivative of the concentration
            of a specific ion in its intracellular pool.

            :param Cion: ion concentration in the pool (arbitrary unit)
            :param Iion: ionic current (mA/m2)
            :param Kion: scaling factor for current contribution to pool (arb. unit / nA???)
            :param tau: time constant for removal of ions from the pool (s)
            :return: variation of ionic concentration in the pool (arbitrary unit /s)
        '''
        return (Kion * (-Iion) - Cion) / tau

    @staticmethod
    def _derAion(Aion, Cion, tau):
        ''' Generic function computing the time derivative of the concentration and time
            dependent activation function, for a specific pool-dependent ionic current.

            :param Aion: concentration and time dependent activation function (arbitrary unit)
            :param Cion: ion concentration in the pool (arbitrary unit)
            :param tau: time constant for activation function variation (s)
            :return: variation of activation function (arbitrary unit / s)
        '''
        return (Cion - Aion) / tau

    minf = partialmethod(_xinf, halfmax=-35.0, slope=-5.0, power=1)
    hinf = partialmethod(_xinf, halfmax=-50.0, slope=9.0, power=2)
    tauh = partialmethod(_taux, halfmax=-36.0, slope=3.5, tauMax=14.0e-3, tauMin=0.2e-3)
    ninf = partialmethod(_xinf, halfmax=-22.0, slope=-9.0, power=1)
    taun = partialmethod(_taux, halfmax=-10.0, slope=10.0, tauMax=6.0e-3, tauMin=1.0e-3)
    sinf = partialmethod(_xinf, halfmax=-10.0, slope=-2.8, power=1)

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derNai(cls, Nai, m, h, Vm):
        ''' Evolution of submembrane Sodium concentration '''
        return cls._derCion(Nai, cls.iNa(m, h, Vm), cls.K_Na, cls.taur_Na)  # M/s

    @classmethod
    def derCai(cls, Cai, s, Vm):
        ''' Evolution of submembrane Calcium concentration '''
        return cls._derCion(Cai, cls.iCa(s, Vm), cls.K_Ca, cls.taur_Ca)  # M/s

    @classmethod
    def derANa(cls, ANa, Nai):
        ''' Evolution of Na+ dependent iPumpNa gate '''
        return cls._derAion(ANa, Nai, cls.taua_PumpNa)

    @classmethod
    def derACa(cls, ACa, Cai):
        ''' Evolution of Ca2+ dependent iKCa gate '''
        return cls._derAion(ACa, Cai, cls.taua_KCa)

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: (cls.minf(Vm) - x['m']) / cls.taum,
            'h': lambda Vm, x: (cls.hinf(Vm) - x['h']) / cls.tauh(Vm),
            'n': lambda Vm, x: (cls.ninf(Vm) - x['n']) / cls.taun(Vm),
            's': lambda Vm, x: (cls.sinf(Vm) - x['s']) / cls.taus,
            'Nai': lambda Vm, x: cls.derNai(x['Nai'], x['m'], x['h'], Vm),
            'ANa': lambda Vm, x: cls.derANa(x['ANa'], x['Nai']),
            'Cai': lambda Vm, x: cls.derCai(x['Cai'], x['s'], Vm),
            'ACa': lambda Vm, x: cls.derACa(x['ACa'], x['Cai'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        lambda_dict = {
            'm': lambda Vm: cls.minf(Vm),
            'h': lambda Vm: cls.hinf(Vm),
            'n': lambda Vm: cls.ninf(Vm),
            's': lambda Vm: cls.sinf(Vm)
        }
        lambda_dict['Nai'] = lambda Vm: -cls.K_Na * cls.iNa(
            lambda_dict['m'](Vm), lambda_dict['h'](Vm), Vm)
        lambda_dict['Cai'] = lambda Vm: -cls.K_Ca * cls.iCa(lambda_dict['s'](Vm), Vm)
        lambda_dict['ANa'] = lambda Vm: lambda_dict['Nai'](Vm)
        lambda_dict['ACa'] = lambda Vm: lambda_dict['Cai'](Vm)
        return lambda_dict

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' Delayed-rectifier Potassium current '''
        return cls.gKdbar * n**2 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iCa(cls, s, Vm):
        ''' Calcium current '''
        return cls.gCabar * s * (Vm - cls.ECa)  # mA/m2

    @classmethod
    def iKCa(cls, ACa, Vm):
        ''' Calcium-activated Potassium current '''
        return cls.gKCabar * ACa * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iPumpNa(cls, ANa, Vm):
        ''' NaK-ATPase pump current '''
        return cls.gPumpNa * ANa * (Vm - cls.EPumpNa)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' Non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iCa': lambda Vm, x: cls.iCa(x['s'], Vm),
            'iPumpNa': lambda Vm, x: cls.iPumpNa(x['ANa'], Vm),
            'iKCa': lambda Vm, x: cls.iKCa(x['ACa'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }


class LeechMech(PointNeuron):
    ''' Generic leech neuron

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    # ------------------------------ Biophysical parameters ------------------------------

    alphaC_sf = 1e-5  # Calcium activation rate constant scaling factor (M)
    betaC = 0.1e3     # beta rate for the open-probability of iKCa channels (s-1)
    T = 293.15        # Room temperature (K)

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def alpham(Vm):
        return -0.03 * (Vm + 28) / (np.exp(- (Vm + 28) / 15) - 1) * 1e3  # s-1

    @staticmethod
    def betam(Vm):
        return 2.7 * np.exp(-(Vm + 53) / 18) * 1e3  # s-1

    @staticmethod
    def alphah(Vm):
        return 0.045 * np.exp(-(Vm + 58) / 18) * 1e3  # s-1

    @staticmethod
    def betah(Vm):
        ''' .. warning:: the original paper contains an error (multiplication) in the
            expression of this rate constant, corrected in the mod file on ModelDB (division).
        '''
        return 0.72 / (np.exp(-(Vm + 23) / 14) + 1) * 1e3  # s-1

    @staticmethod
    def alphan(Vm):
        return -0.024 * (Vm - 17) / (np.exp(-(Vm - 17) / 8) - 1) * 1e3  # s-1

    @staticmethod
    def betan(Vm):
        return 0.2 * np.exp(-(Vm + 48) / 35) * 1e3  # s-1

    @staticmethod
    def alphas(Vm):
        return -1.5 * (Vm - 20) / (np.exp(-(Vm - 20) / 5) - 1) * 1e3  # s-1

    @staticmethod
    def betas(Vm):
        return 1.5 * np.exp(-(Vm + 25) / 10) * 1e3  # s-1

    @classmethod
    def alphaC(cls, Cai):
        return 0.1 * Cai / cls.alphaC_sf * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derC(cls, c, Cai):
        ''' Evolution of the c-gate open-probability '''
        return cls.alphaC(Cai) * (1 - c) - cls.betaC * c  # s-1

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            's': lambda Vm, x: cls.alphas(Vm) * (1 - x['s']) - cls.betas(Vm) * x['s'],
            'c': lambda Vm, x: cls.derC(x['c'], x['Cai'])
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            's': lambda Vm: cls.alphas(Vm) / (cls.alphas(Vm) + cls.betas(Vm)),
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm, Nai):
        ''' Sodium current '''
        ENa = cls.nernst(Z_Na, Nai, cls.Nao, cls.T)  # mV
        return cls.gNabar * m**4 * h * (Vm - ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' Delayed-rectifier Potassium current '''
        return cls.gKdbar * n**2 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iCa(cls, s, Vm, Cai):
        ''' Calcium current '''
        ECa = cls.nernst(Z_Ca, Cai, cls.Cao, cls.T)  # mV
        return cls.gCabar * s * (Vm - ECa)  # mA/m2

    @classmethod
    def iKCa(cls, c, Vm):
        ''' Calcium-activated Potassium current '''
        return cls.gKCabar * c * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' Non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm, x['Nai']),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iCa': lambda Vm, x: cls.iCa(x['s'], Vm, x['Cai']),
            'iKCa': lambda Vm, x: cls.iKCa(x['c'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }


@addSonicFeatures
class LeechPressure(LeechMech):
    ''' Leech pressure sensory neuron

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    # Neuron name
    name = 'LeechP'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2     # Membrane capacitance (F/m2)
    Vm0 = -48.865  # Membrane potential (mV)
    Nai0 = 0.01    # Intracellular Sodium concentration (M)
    Cai0 = 1e-7    # Intracellular Calcium concentration (M)

    # Reversal potentials (mV)
    # ENa = 60      # Sodium (from MOD file on ModelDB)
    # ECa = 125     # Calcium (from MOD file on ModelDB)
    EK = -68.0     # Potassium
    ELeak = -49.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 3500.0  # Sodium
    gKdbar = 60.0    # Delayed-rectifier Potassium
    gCabar = 0.02    # Calcium
    gKCabar = 8.0    # Calcium-dependent Potassium
    gLeak = 5.0      # Non-specific leakage

    # Ionic concentrations (M)
    Nao = 0.11    # Extracellular Sodium
    Cao = 1.8e-3  # Extracellular Calcium

    # Additional parameters
    INaPmax = 70.0    # Maximum pump rate of the NaK-ATPase (mA/m2)
    khalf_Na = 0.012  # Sodium concentration at which NaK-ATPase is at half its maximum rate (M)
    ksteep_Na = 1e-3  # Sensitivity of NaK-ATPase to varying Sodium concentrations (M)
    iCaS = 0.1        # Calcium pump current parameter (mA/m2)
    diam = 50e-6      # Cell soma diameter (m)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'c': 'iKCa gate',
        'Nai': 'submembrane Na+ concentration (M)',
        'Cai': 'submembrane Ca2+ concentration (M)'
    }

    def __new__(cls):
        # Surface to volume ratio of the (spherical) cell soma (m-1)
        SV_ratio = 6 / cls.diam

        # Conversion constants from membrane ionic currents into
        # change rate of intracellular ionic concentrations (M/s)
        cls.K_Na = SV_ratio / (Z_Na * FARADAY) * 1e-6  # Sodium
        cls.K_Ca = SV_ratio / (Z_Ca * FARADAY) * 1e-6  # Calcium

        return super(LeechPressure, cls).__new__(cls)

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {**super().derStates(), **{
            'Nai': lambda Vm, x: -(cls.iNa(x['m'], x['h'], Vm, x['Nai']) +
                                   cls.iPumpNa(x['Nai'])) * cls.K_Na,
            'Cai': lambda Vm, x: -(cls.iCa(x['s'], Vm, x['Cai']) +
                                   cls.iPumpCa(x['Cai'])) * cls.K_Ca
        }}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def cinf(cls, Cai):
        return cls.alphaC(Cai) / (cls.alphaC(Cai) + cls.betaC)

    @classmethod
    def steadyStates(cls):
        lambda_dict = {**super().steadyStates(), **{
            'Nai': lambda _: cls.Nai0,
            'Cai': lambda _: cls.Cai0,
        }}
        lambda_dict['c'] = lambda _: cls.cinf(lambda_dict['Cai'](_))
        return lambda_dict

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iPumpNa(cls, Nai):
        ''' NaK-ATPase pump current '''
        return cls.INaPmax / (1 + np.exp((cls.khalf_Na - Nai) / cls.ksteep_Na))  # mA/m2

    @classmethod
    def iPumpCa(cls, Cai):
        ''' Calcium pump current '''
        return cls.iCaS * (Cai - cls.Cai0) / 1.5  # mA/m2

    @classmethod
    def currents(cls):
        return {**super().currents(), **{
            'iPumpNa': lambda Vm, x: cls.iPumpNa(x['Nai']) / 3.,
            'iPumpCa': lambda Vm, x: cls.iPumpCa(x['Cai'])
        }}


# @addSonicFeatures
class LeechRetzius(LeechMech):
    ''' Leech Retzius neuron

        References:
        *Vazquez, Y., Mendez, B., Trueta, C., and De-Miguel, F.F. (2009). Summation of excitatory
        postsynaptic potentials in electrically-coupled neurones. Neuroscience 163, 202–212.*

        *ModelDB link: https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=120910*

        iA current reference:
        *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
        potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
        J. Neurophysiol. 68, 2086–2099.*
    '''

    # Neuron name
    name = 'LeechR'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 5e-2    # Membrane capacitance (F/m2)
    Vm0 = -44.45  # Membrane resting potential (mV)

    # Reversal potentials (mV)
    ENa = 50.0     # Sodium (from retztemp.ses file on ModelDB)
    EK = -79.0     # Potassium (from retztemp.ses file on ModelDB)
    ECa = 125.0    # Calcium (from cachdend.mod file on ModelDB)
    ELeak = -30.0  # Non-specific leakage (from leakdend.mod file on ModelDB)

    # Maximal channel conductances (S/m2)
    gNabar = 1250.0  # Sodium current
    gKdbar = 10.0    # Delayed-rectifier Potassium
    GAMax = 100.0    # Transient Potassium
    gCabar = 4.0     # Calcium current
    gKCabar = 130.0  # Calcium-dependent Potassium
    gLeak = 1.25     # Non-specific leakage

    # Ionic concentrations (M)
    Cai = 5e-8  # Intracellular Calcium (from retztemp.ses file)

    # Additional parameters
    Vhalf = -73.1  # half-activation voltage (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'c': 'iKCa gate',
        'a': 'iA activation gate',
        'b': 'iA inactivation gate',
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @staticmethod
    def ainf(Vm):
        Vth = -55.0  # mV
        return 0 if Vm <= Vth else min(1, 2 * (Vm - Vth)**3 / ((11 - Vth)**3 + (Vm - Vth)**3))

    @classmethod
    def taua(cls, Vm):
        x = -1.5 * (Vm - cls.Vhalf) * 1e-3 * FARADAY / (Rg * cls.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.7 * x)  # ms-1
        return max(0.5, beta / (0.3 * (1 + alpha))) * 1e-3  # s

    @classmethod
    def binf(cls, Vm):
        return 1. / (1 + np.exp((cls.Vhalf - Vm) / -6.3))

    @classmethod
    def taub(cls, Vm):
        x = 2 * (Vm - cls.Vhalf) * 1e-3 * FARADAY / (Rg * cls.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.65 * x)  # ms-1
        return max(7.5, beta / (0.02 * (1 + alpha))) * 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls, Vm, states):
        return {**super().derStates(Vm, states), **{
            'a': lambda Vm, x: (cls.ainf(Vm) - x['a']) / cls.taua(Vm),
            'b': lambda Vm, x: (cls.binf(Vm) - x['b']) / cls.taub(Vm)
        }}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {**super().steadyStates(), **{
            'a': lambda Vm: cls.ainf(Vm),
            'b': lambda Vm: cls.binf(Vm)
        }}

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iA(cls, a, b, Vm):
        ''' Transient Potassium current '''
        return cls.GAMax * a * b * (Vm - cls.EK)  # mA/m2

    @classmethod
    def currents(cls):
        return {**super().currents(), **{
            'iA': lambda Vm, x: cls.iA(x['a'], x['b'], Vm)
        }}
