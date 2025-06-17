# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-07-07 16:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-28 13:05:44

import re
from ..core import PointNeuron, addSonicFeatures

float_pattern = r'([+-]?\d+\.?\d*)'
pattern = re.compile(
    r'pas_Cm0_{0}uF_cm2_gLeak_{0}S_m2_ELeak_{0}mV'.format(float_pattern))


def passiveNeuron(*args):
    if len(args) == 1:
        Cm0, gLeak, ELeak = [float(x) for x in re.findall(pattern, args[0])[0]]
        Cm0 *= 1e-2
    else:
        Cm0, gLeak, ELeak = args

    @addSonicFeatures
    class PassiveNeuron(PointNeuron):
        ''' Generic point-neuron model with only a passive current. '''

        states = {}

        def __new__(cls, Cm0, gLeak, ELeak):
            ''' Initialization.

                :param Cm0: membrane capacitance (F/m2)
                :param gLeak: leakage conductance (S/m2)
                :param ELeak: leakage revwersal potential (mV)
            '''
            cls.Cm0 = Cm0
            cls.gLeak = gLeak
            cls.ELeak = ELeak
            return super(PassiveNeuron, cls).__new__(cls)

        def copy(self):
            return self.__class__(self.Cm0, self.gLeak, self.ELeak)

        def pdict(self):
            return {
                'Cm0': f'{self.Cm0 * 1e2:.1f} uF/cm2',
                'gLeak': f'{self.gLeak:.1f} S/m2',
                'ELeak': f'{self.ELeak:.1f} mV'
            }

        def __repr__(self):
            params_str = ', '.join([f'{k} = {v}' for k, v in self.pdict().items()])
            return f'{self.__class__.__name__}({params_str})'

        def code(self, pdict):
            pdict = {k: v.replace(' ', '').replace('/', '_') for k, v in pdict.items()}
            s = '_'.join([f'{k}_{v}' for k, v in pdict.items()])
            return f'pas_{s}'

        @property
        def name(self):
            return self.code(self.pdict())

        @property
        def lookup_name(self):
            pdict = self.pdict()
            del pdict['gLeak']
            return self.code(pdict)

        @property
        def Cm0(self):
            return self._Cm0

        @Cm0.setter
        def Cm0(self, value):
            self._Cm0 = value

        @property
        def Vm0(self):
            return self.ELeak

        @classmethod
        def derStates(cls):
            return {}

        @classmethod
        def steadyStates(cls):
            return {}

        @classmethod
        def iLeak(cls, Vm):
            ''' non-specific leakage current '''
            return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

        @classmethod
        def currents(cls):
            return {'iLeak': lambda Vm, _: cls.iLeak(Vm)}

        @property
        def is_passive(self):
            return True

    return PassiveNeuron(Cm0, gLeak, ELeak)


def getDefaultPassiveNeuron():
    Cm0 = 1e-2   # F/m2
    gLeak = 1e2  # S/m2
    ELeak = -70  # mV
    return passiveNeuron(Cm0, gLeak, ELeak)
