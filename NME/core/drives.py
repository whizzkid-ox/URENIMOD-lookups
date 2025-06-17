# Author: Ryo Segawa
# Inspired by work of: https://github.com/tjjlemaire/PySONIC

import abc
import numpy as np

from .stimobj import StimObject, StimObjArray
from ..constants import *
from .batches import Batch


class Drive(StimObject):
    ''' Generic interface to drive object. '''

    @abc.abstractmethod
    def compute(self, t):
        ''' Compute the input drive at a specific time.

            :param t: time (s)
            :return: specific input drive
        '''
        raise NotImplementedError

    @classmethod
    def createQueue(cls, *args):
        ''' Create a list of Drive objects for combinations of input parameters. '''
        if len(args) == 1:
            return [cls(item) for item in args[0]]
        else:
            return [cls(*item) for item in Batch.createQueue(*args)]

    @property
    def is_searchable(self):
        return False


class XDrive(Drive):
    ''' Drive object that can be titrated to find the threshold value of one of its inputs. '''

    xvar_initial = None
    xvar_rel_thr = None
    xvar_thr = None
    xvar_precheck = False

    @property
    @abc.abstractmethod
    def xvar(self):
        raise NotImplementedError

    @xvar.setter
    @abc.abstractmethod
    def xvar(self, value):
        raise NotImplementedError

    def updatedX(self, value):
        other = self.copy()
        other.xvar = value
        return other

    @property
    def is_searchable(self):
        return True

    @property
    def is_resolved(self):
        return self.xvar is not None

    def nullCopy(self):
        return self.copy().updatedX(0.)


class ElectricDrive(XDrive):
    ''' Electric drive object with constant amplitude. '''

    xkey = 'I'
    xvar_initial = ESTIM_AMP_INITIAL
    xvar_rel_thr = ESTIM_REL_CONV_THR
    xvar_range = (0., ESTIM_AMP_UPPER_BOUND)

    def __init__(self, I):
        ''' Constructor.

            :param I: current density (mA/m2)
        '''
        self.I = I

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, value):
        if value is not None:
            value = self.checkFloat('I', value)
        self._I = value

    @property
    def xvar(self):
        return self.I

    @xvar.setter
    def xvar(self, value):
        self.I = value

    def copy(self):
        return self.__class__(self.I)

    @staticmethod
    def inputs():
        return {
            'I': {
                'desc': 'current density amplitude',
                'label': 'I',
                'unit': 'A/m2',
                'factor': 1e-3,
                'precision': 1
            }
        }

    def compute(self, t):
        return self.I


class VoltageDrive(Drive):
    ''' Voltage drive object with a held potential and a step potential. '''

    def __init__(self, Vhold, Vstep):
        ''' Constructor.

            :param Vhold: held voltage (mV)
            :param Vstep: step voltage (mV)
        '''
        self.Vhold = Vhold
        self.Vstep = Vstep

    @property
    def Vhold(self):
        return self._Vhold

    @Vhold.setter
    def Vhold(self, value):
        value = self.checkFloat('Vhold', value)
        self._Vhold = value

    @property
    def Vstep(self):
        return self._Vstep

    @Vstep.setter
    def Vstep(self, value):
        value = self.checkFloat('Vstep', value)
        self._Vstep = value

    def copy(self):
        return self.__class__(self.Vhold, self.Vstep)

    @staticmethod
    def inputs():
        return {
            'Vhold': {
                'desc': 'held voltage',
                'label': 'V_{hold}',
                'unit': 'V',
                'precision': 0,
                'factor': 1e-3
            },
            'Vstep': {
                'desc': 'step voltage',
                'label': 'V_{step}',
                'unit': 'V',
                'precision': 0,
                'factor': 1e-3
            }
        }

    @property
    def filecodes(self):
        return {
            'Vhold': f'{self.Vhold:.1f}mV',
            'Vstep': f'{self.Vstep:.1f}mV',
        }

    def compute(self, t):
        return self.Vstep


class AcousticDrive(XDrive):
    ''' Acoustic drive object with intrinsic frequency and amplitude. '''

    xkey = 'A'
    xvar_initial = ASTIM_AMP_INITIAL
    xvar_rel_thr = ASTIM_REL_CONV_THR
    xvar_thr = ASTIM_ABS_CONV_THR
    xvar_precheck = True

    def __init__(self, f, A=None, phi=np.pi):
        ''' Constructor.

            :param f: carrier frequency (Hz)
            :param A: peak pressure amplitude (Pa)
            :param phi: phase (rad)
        '''
        self.f = f
        self.A = A
        self.phi = phi

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        value = self.checkFloat('f', value)
        self.checkStrictlyPositive('f', value)
        self._f = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is not None:
            value = self.checkFloat('A', value)
            self.checkPositiveOrNull('A', value)
        self._A = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        value = self.checkFloat('phi', value)
        self._phi = value

    def pdict(self, **kwargs):
        d = super().pdict(**kwargs)
        if self.phi == np.pi:
            del d['phi']
        return d

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.A = value

    def copy(self):
        return self.__class__(self.f, self.A, phi=self.phi)

    @staticmethod
    def inputs():
        return {
            'f': {
                'desc': 'US drive frequency',
                'label': 'f',
                'unit': 'Hz',
                'precision': 0
            },
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'Pa',
                'precision': 2
            },
            'phi': {
                'desc': 'US drive phase',
                'label': '\Phi',
                'unit': 'rad',
                'precision': 2
            }
        }

    @property
    def dt(self):
        ''' Determine integration time step. '''
        return 1 / (NPC_DENSE * self.f)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.f)

    @property
    def periodicity(self):
        ''' Determine drive periodicity. '''
        return 1. / self.f

    @property
    def nPerCycle(self):
        return NPC_DENSE

    @property
    def modulationFrequency(self):
        return self.f

    def compute(self, t):
        return self.A * np.sin(2 * np.pi * self.f * t - self.phi)


class DriveArray(StimObjArray):

    objkey = 'drive'

    def compute(self, t):
        return sum(x.compute(t) for x in self)

    def updatedX(self, value):
        return self.__class__([d.updatedX(value) for d in self])

    def nullCopy(self):
        return self.copy().updatedX(0.)


class ElectricDriveArray(DriveArray):

    def __init__(self, objs):
        for x in objs:
            if not isinstance(x, ElectricDrive):
                raise ValueError(f'invalid instance: {x}')
        super().__init__(objs)


class AcousticDriveArray(DriveArray):

    def __init__(self, objs):
        for x in objs:
            if not isinstance(x, AcousticDrive):
                raise ValueError(f'invalid instance: {x}')
        super().__init__(objs)

    @property
    def freqs(self):
        return np.array([x.f for x in self])

    def is_monofrequency(self):
        return np.unique(self.freqs.size) == 1

    @property
    def fmax(self):
        return max(self.freqs)

    @property
    def fmin(self):
        return min(self.freqs)

    @property
    def dt(self):
        return 1 / (NPC_DENSE * self.fmax)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.fmax)

    @property
    def periodicity(self):
        if self.is_monofrequency():
            return self[0].periodicity  # s
        if self.size > 2:
            raise ValueError('cannot compute periodicity for more than two drives')
        return 1 / (self.fmax - self.fmin)

    @property
    def nPerCycle(self):
        return int(self.periodicity // self.dt)

    @property
    def modulationFrequency(self):
        return np.mean(self.freqs)


def getDriveArray(drives):
    ''' Interface function to initialize the appropriate drive/drive array '''
    if isinstance(drives, Drive):
        return drives
    if len(drives) == 1:
        return drives[0]
    refdrive = drives[0]
    if isinstance(refdrive, ElectricDrive):
        return ElectricDriveArray(drives)
    elif isinstance(refdrive, AcousticDrive):
        return AcousticDriveArray(drives)
    else:
        raise ValueError(f'cannot instantiate array from drive type: {refdrive}')