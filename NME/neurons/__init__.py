# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-06 13:36:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-19 12:19:10

import inspect
import sys

from .template import *
from .cortical import *
from .thalamic import *
from .leech import *
from .stn import *
from .hh import *
from .fh import *
from .sweeney import *
from .sundt import *
from .mrg import *
from .pas import *


def getNeuronsDict():
    ''' Construct a dictionary of all the implemented point neuron classes, indexed by name. '''
    current_module = sys.modules[__name__]
    neurons_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'name') and isinstance(obj.name, str):
            neurons_dict[obj.name] = obj
    return neurons_dict


def getPointNeuron(name):
    ''' Return a point-neuron instance corresponding to a given name. '''
    neuron_classes = getNeuronsDict()
    try:
        return neuron_classes[name]()
    except KeyError:
        if name.startswith('pas'):
            return passiveNeuron(name)
        else:
            raise ValueError('"{}" neuron not found. Implemented neurons are: {}'.format(
                name, ', '.join(list(neuron_classes.keys()))))
