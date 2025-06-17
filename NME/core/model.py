# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-27 09:07:38

from functools import wraps
from inspect import getdoc
import abc
import inspect
import numpy as np
from boltons.strutils import cardinalize

from .batches import Batch
from ..threshold import titrate
from ..utils import *


class Model(metaclass=abc.ABCMeta):
    ''' Generic model interface. '''

    @property
    @abc.abstractmethod
    def tscale(self):
        ''' Relevant temporal scale of the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize simulations made with the model. '''
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        ''' String representation. '''
        raise NotImplementedError

    def params(self):
        ''' Return a dictionary of all model parameters (class and instance attributes) '''
        def toAvoid(p):
            return (p.startswith('__') and p.endswith('__')) or p.startswith('_abc_')
        class_attrs = inspect.getmembers(self.__class__, lambda a: not(inspect.isroutine(a)))
        inst_attrs = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        class_attrs = [a for a in class_attrs if not toAvoid(a[0])]
        inst_attrs = [a for a in inst_attrs if not toAvoid(a[0]) and a not in class_attrs]
        params_dict = {a[0]: a[1] for a in class_attrs + inst_attrs}
        return params_dict

    @classmethod
    def description(cls):
        return getdoc(cls).split('\n', 1)[0].strip()

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def inputs():
        ''' Return an informative dictionary on input variables used to simulate the model. '''
        raise NotImplementedError

    @abc.abstractmethod
    def filecodes(self, *args):
        ''' Return a dictionary of string-encoded inputs used for file naming. '''
        raise NotImplementedError

    def filecode(self, *args):
        return filecode(self, *args)

    @classmethod
    @abc.abstractmethod
    def getPltVars(self, *args, **kwargs):
        ''' Return a dictionary with information about all plot variables related to the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pltScheme(self):
        ''' Return a dictionary model plot variables grouped by context. '''
        raise NotImplementedError

    @staticmethod
    def checkOutputDir(queuefunc):
        ''' Check if an output directory is provided in input arguments, and if so, add it
            to each item of the returned queue (along with an "overwrite" boolean).
        '''
        @wraps(queuefunc)
        def wrapper(self, *args, **kwargs):
            outputdir = kwargs.get('outputdir')
            queue = queuefunc(self, *args, **kwargs)
            if outputdir is not None:
                overwrite = kwargs.get('overwrite', True)
                queue = queuefunc(self, *args, **kwargs)
                for i, params in enumerate(queue):
                    position_args, keyword_args = Batch.resolve(params)
                    keyword_args['overwrite'] = overwrite
                    keyword_args['outputdir'] = outputdir
                    queue[i] = (position_args, keyword_args)
            else:
                if len(queue) > 5:
                    logger.warning('Running more than 5 simulations without file saving')
            return queue
        return wrapper

    @classmethod
    @abc.abstractmethod
    def simQueue(cls, *args, outputdir=None, overwrite=True):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def checkInputs(self, *args):
        ''' Check the validity of simulation input parameters. '''
        raise NotImplementedError

    @abc.abstractmethod
    def derivatives(self, *args, **kwargs):
        ''' Compute ODE derivatives for a specific set of ODE states and external parameters. '''
        raise NotImplementedError

    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        ''' Simulate the model's differential system for specific input parameters
            and return output data in a dataframe. '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def meta(self, *args):
        ''' Return an informative dictionary about model and simulation parameters. '''
        raise NotImplementedError

    @staticmethod
    def addMeta(simfunc):
        ''' Add informative dictionary to simulation output '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            data, tcomp = timer(simfunc)(self, *args, **kwargs)
            logger.debug('completed in %ss', si_format(tcomp, 1))
            meta_dict = getMeta(self, simfunc, *args, **kwargs)
            meta_dict['tcomp'] = tcomp
            return data, meta_dict
        return wrapper

    @staticmethod
    def logNSpikes(simfunc):
        ''' Log number of detected spikes on charge profile of simulation output. '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            out = simfunc(self, *args, **kwargs)
            if out is None:
                return None
            data, meta = out
            nspikes = self.getNSpikes(data)
            logger.debug(f'{nspikes} {cardinalize("spike", nspikes)} detected')
            return data, meta

        return wrapper

    @staticmethod
    def checkSimParams(simfunc):
        ''' Check simulation parameters before launching simulation. '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            args, kwargs = alignWithMethodDef(simfunc, args, kwargs)
            self.checkInputs(*args, *list(kwargs.values()))
            return simfunc(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def logDesc(simfunc):
        ''' Log description of model and simulation parameters. '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            args, kwargs = alignWithMethodDef(simfunc, args, kwargs)
            logger.info(self.desc(getMeta(self, simfunc, *args, **kwargs)))
            return simfunc(self, *args, **kwargs)
        return wrapper

    def titrate(self, *args, **kwargs):
        return titrate(self, *args, **kwargs)

    @staticmethod
    def checkTitrate(simfunc):
        ''' If unresolved drive provided in the list of input parameters,
            perform a titration to find the threshold drive, and simulate with resolved drive.
        '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            # Extract drive object from args
            drive, *other_args = args

            # If drive is titratable and not fully resolved
            if drive.is_searchable:
                if not drive.is_resolved:
                    # Titrate
                    xthr = self.titrate(*args)

                    # If no threshold was found, return None
                    if np.isnan(xthr):
                        logger.error(
                            f'Could not find threshold {drive.inputs()[drive.xkey]["desc"]}')
                        return None

                    # Otherwise, update args list with resovled drive
                    args = (drive.updatedX(xthr), *other_args)

            # Execute simulation function
            return simfunc(self, *args, **kwargs)

        return wrapper

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)

    def getOutput(self, *args, **kwargs):
        ''' Get simulation output data for a specific parameters combination, by looking
            for an output file into a specific directory.

            If a corresponding output file is not found in the specified directory, the model
            is first run and results are saved in the output file.
        '''
        fpath = self.simAndSave(*args, overwrite=False, **kwargs)
        return loadData(fpath)
