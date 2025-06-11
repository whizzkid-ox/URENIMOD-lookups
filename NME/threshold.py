# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-28 16:42:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-27 09:10:29

import numpy as np

from .utils import logger, isWithin


class OutOfBoundsError(Exception):
    def __init__(self, bounds):
        msg = f'No threshold found within the [{bounds[0]:.2e} - {bounds[1]:.2e}] interval'
        super().__init__(msg)


class MaxNIterations(Exception):
    def __init__(self, max_nit, history):
        msg = f'Maximum number of iterations ({max_nit}) reached, history = {history}'
        super().__init__(msg)


class Thresholder:
    ''' Class used to determine the threshold satisfying a given condition within a
        continuous search interval, using a binary search with initial preconditioning.
    '''

    eps_machine = np.sqrt(np.finfo(float).eps)
    err_val = np.nan

    def __init__(self, feval, xbounds, x0=None, eps_thr=None, rel_eps_thr=1e-2,
                 max_nit=50, precheck=False, fbound=2):
        ''' Initialization.

            :param feval: evaluation function returning whether condition is satisfied
            :param xbounds: initial search interval for threshold
            :param x0: initial evaluation value
            :param eps_thr: maximum absolute error
            :param rel_eps_thr: maximum relative error
            :param precheck: boolean stating whether to perform an initial check
             for the existence of a threshold within the interval
            :param fbound: integer factor indicating the magnitude of the initial bounding procedure
            :return: final threshold, or full search history
        '''
        self.feval = feval
        self.xbounds = xbounds
        self.rel_eps_thr = rel_eps_thr
        self.eps_thr = eps_thr
        self.max_nit = max_nit
        self.fbound = fbound
        self.precheck = precheck
        self.x0 = x0

    @property
    def feval(self):
        return self._feval

    @feval.setter
    def feval(self, value):
        if not callable(value):
            raise ValueError('feval must be a callable object')
        self._feval = value

    @property
    def xbounds(self):
        return self._xbounds

    @xbounds.setter
    def xbounds(self, value):
        if len(value) != 2:
            raise ValueError('xbounds must be an iterbale of size 2')
        if value[0] >= value[1]:
            raise ValueError('lower bound must be smaller than upper bound')
        self._xbounds = value

    @property
    def fixed_lb(self):
        return self.xbounds[0]

    @fixed_lb.setter
    def fixed_lb(self, value):
        self.xbounds = (value, self.fixed_ub)

    @property
    def fixed_ub(self):
        return self.xbounds[1]

    @fixed_ub.setter
    def fixed_ub(self, value):
        self.xbounds = (self.fixed_lb, value)

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        if value is None:  # If not specified, set to geometric mean of search interval
            value = self.getStartPoint(self.xbounds, x=0.5, scale='log')
        if value == 0.:  # If zero, set to mid-point of search interval
            value = self.getStartPoint(self.xbounds, x=0.5, scale='lin')
        self._x0 = value

    @property
    def eps_thr(self):
        return self._eps_thr

    @eps_thr.setter
    def eps_thr(self, value):
        if value is None:  # If not specified, set to infinity
            value = np.inf
        self._eps_thr = value

    @property
    def rel_eps_thr(self):
        return self._rel_eps_thr

    @rel_eps_thr.setter
    def rel_eps_thr(self, value):
        value = isWithin('rel_eps_thr', value, (0., 1.))
        self._rel_eps_thr = value

    @property
    def max_nit(self):
        return self._max_nit

    @max_nit.setter
    def max_nit(self, value):
        if not isinstance(value, int):
            raise ValueError('max_nit must be of type int')
        if value < 1:
            raise ValueError('max_nit must be greater than 0')
        self._max_nit = value

    @property
    def precheck(self):
        return self._precheck

    @precheck.setter
    def precheck(self, value):
        if not isinstance(value, bool):
            raise ValueError('precheck must be of type bool')
        self._precheck = value

    @property
    def fbound(self):
        return self._fbound

    @fbound.setter
    def fbound(self, value):
        if value is not None:
            if value <= 1:
                raise ValueError('bounding factor must be greater than 1')
            # If fixed lower bound is zero, re-assign it to absolue threshold (if provided)
            # or to machine epsilon
            if self.fixed_lb == 0.:
                self.fixed_lb = self.eps_thr / 2 if self.eps_thr < np.inf else self.eps_machine
            # Search interval must span more than 2 times bounding factor
            if self.fixed_ub / self.fixed_lb <= 2 * value:
                raise ValueError(f'search interval too narrow for factor bounding')
        self._fbound = value

    @property
    def x(self):
        return self._x_history[-1]

    @x.setter
    def x(self, value):
        if not hasattr(self, '_x_history'):
            self._x_history = []
        self._x_history.append(value)

    @property
    def x_history(self):
        return np.array(self._x_history)

    @property
    def is_above(self):
        return self._eval_history[-1]

    @is_above.setter
    def is_above(self, value):
        if not hasattr(self, '_eval_history'):
            self._eval_history = []
        self._eval_history.append(value)

    @property
    def eval_history(self):
        return np.array(self._eval_history)

    @property
    def has_changed_eval(self):
        return len(set(self._eval_history)) > 1

    def eval(self):
        self.is_above = self.feval(self.x)
        isWithin('x', self.x, self.xbounds, raise_warning=False)
        self.checkNiterations()

    @property
    def nits(self):
        return len(self._x_history)

    @property
    def midpoint(self):
        return (self.ub + self.lb) / 2

    @property
    def eff_thr(self):
        return min(self.rel_eps_thr * self.lb, self.eps_thr)

    def hasConverged(self):
        return np.abs(self.ub - self.lb) <= 2 * self.eff_thr

    @staticmethod
    def getStartPoint(bounds, x=0.5, scale='lin'):
        ''' Define a value located at a given relative distance between two bounds.

            :param bounds: lower and upper bound values
            :param x: relative logarithmic distance, between 0 (lower bound) and 1 (upper bound)
            :param scale: scale type between bounds ('lin' / 'log')
            :return: scaled starting value
        '''
        if scale == 'log':
            bounds = np.log10(bounds)
        x0 = (1 - x) * bounds[0] + x * bounds[1]
        if scale == 'log':
            x0 = np.power(10., x0)
        return x0

    def checkNiterations(self):
        ''' Check that number of iterations does not exceed limit. '''
        if self.nits >= self.max_nit:
            raise MaxNIterations(self.max_nit, self._x_history)

    def initBounds(self):
        self.lb, self.ub = self.xbounds

    def checkAtBound(self):
        ''' Evaluate at the appropriate bound based on last evaluation result, and
            raise error if evaluation indicates no threshold within interval. '''
        last_eval = self.is_above
        self.x = self.lb if self.is_above else self.ub
        self.eval()
        if self.is_above == last_eval:
            raise OutOfBoundsError(self.xbounds)

    def preCondition(self):
        ''' Refine search interval by either multiplying or dividing x by a specific integer
            factor k until target lies within an interval [x, kx]
        '''
        # If exact match between (k * x) and ub or between (x / k) and lb, adapt k slightly
        if self.x * self.fbound == self.ub or self.lb * self.fbound == self.x:
            self.fbound *= 0.99

        # Iterate while upper bound is more than (k * x) or lower bound is less than (x / k)
        while self.lb < self.x / self.fbound or self.ub > self.x * self.fbound:
            # Refine interval and x based on feval result
            if self.is_above:
                self.ub = self.x
                self.x = self.ub / self.fbound
            else:
                self.lb = self.x
                self.x = self.fbound * self.lb
            # If lower bound greater or equal to upper bound -> raise error
            if self.lb >= self.ub:
                raise OutOfBoundsError(self.xbounds)
            # Evaluate
            self.eval()

        # Set x to interval mid-point and re-evaluate
        self.x = self.midpoint
        self.eval()

    def binSearch(self):
        ''' Binary search until interval is smaller than most stringent threshold criterion. '''
        while not self.hasConverged():
            # Refine interval based on feval result
            if self.is_above:
                self.ub = self.x
            else:
                self.lb = self.x

            # Set x to interval mid-point and re-evaluate
            self.x = self.midpoint
            self.eval()

    def refine(self):
        ''' Refine threshold once convergence has been reached. '''
        # If last value is not above threshold
        if not self.is_above:
            # Set x to interval mid-point and re-evaluate (to ensure relative convergence)
            self.lb, self.x = self.x, self.midpoint
            self.eval()
            # If last value still not above threshold, evaluate at upper bound
            if not self.is_above:
                self.x = self.ub
                self.eval()

    def run(self, output_history=False):
        self.initBounds()
        self.x = self.x0
        self.eval()
        try:
            if self.precheck:  # Run pre-check at the approprite interval bound if required
                self.checkAtBound()
                self.initBounds()  # Re-initialize bounds
            if self.fbound is not None:  # Perform initial factor bounding if required
                self.preCondition()
            self.binSearch()  # Run binary search until convergence
            if not self.has_changed_eval:  # if feval has not changed output, evaluate at the bound
                self.checkAtBound()
            self.refine()  # refine to make sure final value is above threshold
        except (OutOfBoundsError, MaxNIterations) as err:  # if error is raised, assign nan and log
            logger.error(err)
            self.x = self.err_val


def threshold(*args, output_history=False, **kwargs):
    ''' Wrapper function around the Thresholder class.

        :param output_history: boolean stating whether to return history of search procedure
        :return: final threshold, or full search history
    '''
    th = Thresholder(*args, **kwargs)
    th.run()
    if output_history:
        return th.x_history, th.eval_history
    else:
        return th.x


def titrate(model, drive, pp, **kwargs):
    ''' Use a binary search to determine the threshold amplitude needed
        to obtain neural excitation for a given duration, PRF and duty cycle.

        :param model: model object
        :param drive: unresolved drive object
        :param pp: pulsed protocol object
        :param xfunc: function determining whether condition is reached from simulation output
        :param Arange: search interval for electric current amplitude, iteratively refined
        :return: excitation amplitude (in drive units)
    '''
    xfunc = kwargs.pop('xfunc', None)
    Arange = kwargs.pop('Arange', None)

    # Default output function
    if xfunc is None:
        xfunc = model.titrationFunc

    # Default amplitude interval
    if Arange is None:
        Arange = model.getArange(drive)

    return threshold(
        lambda x: xfunc(model.simulate(drive.updatedX(x), pp, **kwargs)[0]),
        Arange,
        x0=drive.xvar_initial,
        rel_eps_thr=drive.xvar_rel_thr,
        eps_thr=drive.xvar_thr,
        precheck=drive.xvar_precheck)
