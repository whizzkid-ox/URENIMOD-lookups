# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-06 13:36:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-15 11:03:46

from types import MethodType
import inspect
import sys

from .solvers import *
from .batches import *
from .model import *
from .pneuron import *
from .bls import *
from .translators import *
from .nbls import *
from .vclamp import *
from .lookups import *
from .stimobj import *
from .protocols import *
from .drives import *
from .timeseries import *

from ..neurons import getPointNeuron


def getModelsDict():
    ''' Construct a dictionary of all model classes, indexed by simulation key. '''
    current_module = sys.modules[__name__]
    models_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'simkey') and isinstance(obj.simkey, str):
            models_dict[obj.simkey] = obj
    return models_dict


# Add an initFromMeta method to the Pointneuron class (done here to avoid circular import)
PointNeuron.initFromMeta = MethodType(
    lambda self, meta: getPointNeuron(meta['neuron']), PointNeuron)
models_dict = getModelsDict()


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    try:
        return models_dict[simkey].initFromMeta(meta['model'])
    except KeyError:
        raise ValueError(f'Unknown model type:{simkey}')
