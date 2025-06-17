# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 12:05:32

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode, odeint, solve_ivp
from tqdm import tqdm

from ..utils import *
from ..constants import *
from .timeseries import TimeSeries


class ODESolver:
    ''' Generic interface to ODE solver object. '''

    def __init__(self, ykeys, dfunc, dt=None):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param dt: integration time step (s)
        '''
        self.ykeys = ykeys
        self.dfunc = dfunc
        self.dt = dt

    def checkFunc(self, key, value):
        if not callable(value):
            raise ValueError(f'{key} function must be a callable object')

    @property
    def ykeys(self):
        return self._ykeys

    @ykeys.setter
    def ykeys(self, value):
        if not isIterable(value):
            value = list(value)
        for item in value:
            if not isinstance(item, str):
                raise ValueError('ykeys must be a list of strings')
        self._ykeys = value

    @property
    def nvars(self):
        return len(self.ykeys)

    @property
    def dfunc(self):
        return self._dfunc

    @dfunc.setter
    def dfunc(self, value):
        self.checkFunc('derivative', value)
        self._dfunc = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value is None:
            self._dt = None
        else:
            if not isinstance(value, float):
                raise ValueError('time step must be float-typed')
            if value <= 0:
                raise ValueError('time step must be strictly positive')
            self._dt = value

    def getNSamples(self, t0, tend, dt=None):
        ''' Get the number of samples required to integrate across 2 times with a given time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :param dt: integration time step (s)
            :return: number of required samples, rounded to nearest integer
        '''
        if dt is None:
            dt = self.dt
        return max(int(np.round((tend - t0) / dt)), 2)

    def getTimeVector(self, t0, tend, **kwargs):
        ''' Get the time vector required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :return: vector going from current time to target time with appropriate step (s)
        '''
        return np.linspace(t0, tend, self.getNSamples(t0, tend, **kwargs))

    def initialize(self, y0, t0=0.):
        ''' Initialize global time vector, state vector and solution array.

            :param y0: dictionary of initial conditions
            :param t0: optional initial time or time vector (s)
        '''
        keys = list(y0.keys())
        if len(keys) != len(self.ykeys):
            raise ValueError("Initial conditions do not match system's dimensions")
        for k in keys:
            if k not in self.ykeys:
                raise ValueError(f'{k} is not a differential variable')
        y0 = {k: np.asarray(v) if isIterable(v) else np.array([v]) for k, v in y0.items()}
        ref_size = y0[keys[0]].size
        if not all(v.size == ref_size for v in y0.values()):
            raise ValueError('dimensions of initial conditions are inconsistent')
        self.y = np.array(list(y0.values())).T
        self.t = np.ones(self.y.shape[0]) * t0
        self.x = np.zeros(self.t.size)

    def append(self, t, y):
        ''' Append to global time vector, state vector and solution array.

            :param t: new time vector to append (s)
            :param y: new solution matrix to append
        '''
        self.t = np.concatenate((self.t, t))
        self.y = np.concatenate((self.y, y), axis=0)
        self.x = np.concatenate((self.x, np.ones(t.size) * self.xref))

    def bound(self, tbounds):
        ''' Restrict global time vector, state vector ans solution matrix within
            specific time range.

            :param tbounds: minimal and maximal allowed time restricting the global arrays (s).
        '''
        i_bounded = np.logical_and(self.t >= tbounds[0], self.t <= tbounds[1])
        self.t = self.t[i_bounded]
        self.y = self.y[i_bounded, :]
        self.x = self.x[i_bounded]

    @staticmethod
    def timeStr(t):
        return f'{t * 1e3:.5f} ms'

    def timedlog(self, s, t=None):
        ''' Add preceding time information to log string. '''
        if t is None:
            t = self.t[-1]
        return f't = {self.timeStr(t)}: {s}'

    def integrateUntil(self, target_t, remove_first=False):
        ''' Integrate system until a target time and append new arrays to global arrays.

            :param target_t: target time (s)
            :param remove_first: optional boolean specifying whether to remove the first index
            of the new arrays before appending
        '''
        if target_t < self.t[-1]:
            raise ValueError(f'target time ({target_t} s) precedes current time {self.t[-1]} s')
        elif target_t == self.t[-1]:
            t, y = self.t[-1], self.y[-1]
        if self.dt is None:
            sol = solve_ivp(
                self.dfunc, [self.t[-1], target_t], self.y[-1], method='LSODA')
            t, y = sol.t, sol.y.T
        else:
            t = self.getTimeVector(self.t[-1], target_t)
            y = odeint(self.dfunc, self.y[-1], t, tfirst=True)
        if remove_first:
            t, y = t[1:], y[1:]
        self.append(t, y)

    def resampleArrays(self, t, y, target_dt):
        ''' Resample a time vector and soluton matrix to target time step.

            :param t: time vector to resample (s)
            :param y: solution matrix to resample
            :target_dt: target time step (s)
            :return: resampled time vector and solution matrix
        '''
        tnew = self.getTimeVector(t[0], t[-1], dt=target_dt)
        ynew = np.array([np.interp(tnew, t, x) for x in y.T]).T
        return tnew, ynew

    def resample(self, target_dt):
        ''' Resample global arrays to a new target time step.

            :target_dt: target time step (s)
        '''
        tnew, self.y = self.resampleArrays(self.t, self.y, target_dt)
        self.x = interp1d(self.t, self.x, kind='nearest', assume_sorted=True)(tnew)
        self.t = tnew

    def solve(self, y0, tstop, **kwargs):
        ''' Simulate system for a given time interval for specific initial conditions.

            :param y0: dictionary of initial conditions
            :param tstop: stopping time (s)
        '''
        # Initialize system
        self.initialize(y0, **kwargs)

        # Integrate until tstop
        self.integrateUntil(tstop, remove_first=True)

    @property
    def solution(self):
        ''' Return solution as a pandas dataframe.

            :return: timeseries dataframe with labeled time, state and variables vectors.
        '''
        return TimeSeries(self.t, self.x, {k: self.y[:, i] for i, k in enumerate(self.ykeys)})

    def __call__(self, *args, target_dt=None, max_nsamples=None, **kwargs):
        ''' Specific call method: solve the system, resample solution if needed, and return
            solution dataframe. '''
        self.solve(*args, **kwargs)
        if target_dt is not None:
            self.resample(target_dt)
        elif max_nsamples is not None and self.t.size > max_nsamples:
            self.resample(np.ptp(self.t) / max_nsamples)
        return self.solution


class PeriodicSolver(ODESolver):
    ''' ODE solver that integrates periodically until a stable periodic behavior is detected.'''

    def __init__(self, T, *args, primary_vars=None, **kwargs):
        ''' Initialization.

            :param T: periodicity (s)
            :param primary_vars: keys of the primary solution variables to check for stability
        '''
        super().__init__(*args, **kwargs)
        self.T = T
        self.primary_vars = primary_vars

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if not isinstance(value, float):
            raise ValueError('periodicity must be float-typed')
        if value <= 0:
            raise ValueError('periodicity must be strictly positive')
        self._T = value

    @property
    def primary_vars(self):
        return self._primary_vars

    @primary_vars.setter
    def primary_vars(self, value):
        if value is None:  # If none specified, set all variables to be checked for stability
            value = self.ykeys
        if not isIterable(value):
            value = [value]
        for item in value:
            if item not in self.ykeys:
                raise ValueError(f'{item} is not a differential variable')
        self._primary_vars = value

    @property
    def i_primary_vars(self):
        return [self.ykeys.index(k) for k in self.primary_vars]

    @property
    def xref(self):
        return 1.

    def getNPerCycle(self, dt=None):
        ''' Compute number of samples per cycle.

            :param dt: optional integration time step (s)
            :return: number of samples per cycle, rounded to nearest integer
        '''
        # if time step not provided, compute dt from last 2 elements of time vector
        if dt is None:
            dt = self.t[-1] - self.t[-2]
        return int(np.round(self.T / dt))

    def getCycle(self, i, ivars=None):
        ''' Get time vector and solution matrix for the ith cycle.

            :param i: cycle index
            :param ivars: optional indexes of subset of variables of interest
            :return: solution matrix for ith cycle, filtered for variables of interest
        '''
        # By default, consider indexes of all variables
        if ivars is None:
            ivars = range(self.nvars)

        # Get first time index where time difference differs from solver's time step, if any
        i_diff_dt = np.where(np.invert(np.isclose(np.diff(self.t)[::-1], self.dt)))[0]

        # Determine the number of samples to consider in the backwards direction
        nsamples = i_diff_dt[0] if i_diff_dt.size > 0 else self.t.size

        npc = self.getNPerCycle()                # number of samples per cycle
        ncycles = int(np.round(nsamples / npc))  # rounded number of cycles
        ioffset = self.t.size - npc * ncycles    # corresponding initial index offset

        # Check index validity
        if i < 0:
            i += ncycles
        if i < 0 or i >= ncycles:
            raise ValueError('Invalid index')

        # Compute start and end indexes
        istart = i * npc + ioffset
        iend = istart + npc

        # Return arrays for corresponding cycle
        return self.t[istart:iend], self.y[istart:iend, ivars]

    def isPeriodicallyStable(self):
        ''' Assess the periodic stabilization of a solution, by evaluating the deviation
            of system variables between the last two periods.

            :return: boolean stating whether the solution is periodically stable or not
        '''
        # Extract the last 2 cycles of the primary variables from the solution
        y_last, y_prec = [self.getCycle(-i, ivars=self.i_primary_vars)[1] for i in [1, 2]]

        # Evaluate ratios of RMSE between the two cycles / variation range over the last cycle
        ratios = rmse(y_last, y_prec, axis=0) / np.ptp(y_last, axis=0)

        # Classify solution as periodically stable only if all ratios are below critical threshold
        return np.all(ratios < MAX_RMSE_PTP_RATIO)

    def integrateCycle(self):
        ''' Integrate system for a cycle. '''
        self.integrateUntil(self.t[-1] + self.T, remove_first=True)

    def solve(self, y0, nmax=None, nmin=None, **kwargs):
        ''' Simulate system with a specific periodicity until stopping criterion is met.

            :param y0: dictionary of initial conditions
            :param nmax: maximum number of integration cycles (optional)
        '''
        if nmax is None:
            nmax = NCYCLES_MAX
        if nmin is None:
            nmin = 2
        assert nmin <= nmax, 'incorrect bounds for number of cycles (min > max)'

        # Initialize system
        if y0 is not None:
            self.initialize(y0, **kwargs)

        # Integrate system for minimal number of cycles
        for i in range(nmin):
            self.integrateCycle()

        # Keep integrating system periodically until stopping criterion is met
        while not self.isPeriodicallyStable() and i < nmax:
            self.integrateCycle()
            i += 1

        # Log stopping criterion
        if i == nmax:
            logger.warning(self.timedlog(f'criterion not met -> stopping after {i} cycles'))
        else:
            logger.debug(self.timedlog(f'stopping criterion met after {i} cycles'))


class EventDrivenSolver(ODESolver):
    ''' Event-driven ODE solver. '''

    def __init__(self, eventfunc, *args, event_params=None, **kwargs):
        ''' Initialization.

            :param eventfunc: function called on each event
            :param event_params: dictionary of parameters used by the derivatives function
        '''
        super().__init__(*args, **kwargs)
        self.eventfunc = eventfunc
        self.assignEventParams(event_params)

    def assignEventParams(self, event_params):
        ''' Assign event parameters as instance attributes. '''
        if event_params is not None:
            for k, v in event_params.items():
                setattr(self, k, v)

    @property
    def eventfunc(self):
        return self._eventfunc

    @eventfunc.setter
    def eventfunc(self, value):
        self.checkFunc('event', value)
        self._eventfunc = value

    @property
    def xref(self):
        return self._xref

    @xref.setter
    def xref(self, value):
        self._xref = value

    def initialize(self, *args, **kwargs):
        self.xref = 0
        super().initialize(*args, **kwargs)

    def fireEvent(self, xevent):
        ''' Call event function and set new xref value. '''
        if xevent is not None:
            if xevent == 'log':
                self.logProgress()
            else:
                self.eventfunc(xevent)
                self.xref = xevent

    def initLog(self, logfunc, n):
        ''' Initialize progress logger. '''
        self.logfunc = logfunc
        if self.logfunc is None:
            setHandler(logger, TqdmHandler(my_log_formatter))
            self.pbar = tqdm(total=n)
        else:
            self.np = n
            logger.debug('integrating stimulus')

    def logProgress(self):
        ''' Log simulation progress. '''
        if self.logfunc is None:
            self.pbar.update()
        else:
            logger.debug(self.timedlog(self.logfunc(self.y[-1])))

    def terminateLog(self):
        ''' Terminate progress logger. '''
        if self.logfunc is None:
            self.pbar.close()
        else:
            logger.debug('integration completed')

    def sortEvents(self, events):
        ''' Sort events pairs by occurence time. '''
        return sorted(events, key=lambda x: x[0])

    def solve(self, y0, events, tstop, log_period=None, logfunc=None, **kwargs):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param events: list of events
            :param tstop: stopping time (s)
        '''
        # Sort events according to occurrence time
        events = self.sortEvents(events)

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')

        if log_period is not None:  # Add log events if any
            tlogs = np.arange(kwargs.get('t0', 0.), tstop, log_period)[1:]
            if tstop not in tlogs:
                tlogs = np.hstack((tlogs, [tstop]))
            events = self.sortEvents(events + [(t, 'log') for t in tlogs])
            self.initLog(logfunc, tlogs.size)
        else:  # Otherwise, add None event at tstop
            events.append((tstop, None))

        # Initialize system
        self.initialize(y0, **kwargs)

        # For each upcoming event
        for i, (tevent, xevent) in enumerate(events):
            self.integrateUntil(  # integrate until event time
                tevent,
                remove_first=i > 0 and events[i - 1][1] == 'log')
            self.fireEvent(xevent)  # fire event

        # Terminate log if any
        if log_period is not None:
            self.terminateLog()


class HybridSolver(EventDrivenSolver, PeriodicSolver):

    def __init__(self, ykeys, dfunc, dfunc_sparse, predfunc, eventfunc, T,
                 dense_vars, dt_dense, dt_sparse, **kwargs):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivatives function
            :param dfunc_sparse: derivatives function for sparse integration periods
            :param predfunc: function computing the extra arguments necessary for sparse integration
            :param eventfunc: function called on each event
            :param T: periodicity (s)
            :param dense_vars: list of fast-evolving differential variables
            :param dt_dense: dense integration time step (s)
            :param dt_sparse: sparse integration time step (s)
        '''
        PeriodicSolver.__init__(
            self, T, ykeys, dfunc, primary_vars=kwargs.get('primary_vars', None), dt=dt_dense)
        self.eventfunc = eventfunc
        self.assignEventParams(kwargs.get('event_params', None))
        self.predfunc = predfunc
        self.dense_vars = dense_vars
        self.dt_sparse = dt_sparse
        self.sparse_solver = ode(dfunc_sparse)
        self.sparse_solver.set_integrator('dop853', nsteps=SOLVER_NSTEPS, atol=1e-12)

    @property
    def predfunc(self):
        return self._predfunc

    @predfunc.setter
    def predfunc(self, value):
        self.checkFunc('prediction', value)
        self._predfunc = value

    @property
    def dense_vars(self):
        return self._dense_vars

    @dense_vars.setter
    def dense_vars(self, value):
        if value is None:  # If none specified, set all variables as dense variables
            value = self.ykeys
        if not isIterable(value):
            value = [value]
        for item in value:
            if item not in self.ykeys:
                raise ValueError(f'{item} is not a differential variable')
        self._dense_vars = value

    @property
    def is_dense_var(self):
        return np.array([x in self.dense_vars for x in self.ykeys])

    @property
    def is_sparse_var(self):
        return np.invert(self.is_dense_var)

    def integrateSparse(self, ysparse, target_t):
        ''' Integrate sparse system until a specific time.

            :param ysparse: sparse 1-cycle solution matrix of fast-evolving variables
            :paramt target_t: target time (s)
        '''
        # Compute number of samples in the sparse cycle solution
        npc = ysparse.shape[0]

        # Initialize time vector and solution array for the current interval
        n = int(np.ceil((target_t - self.t[-1]) / self.dt_sparse))
        t = np.linspace(self.t[-1], target_t, n + 1)[1:]
        y = np.empty((n, self.y.shape[1]))

        # Initialize sparse integrator
        self.sparse_solver.set_initial_value(self.y[-1, self.is_sparse_var], self.t[-1])
        for i, tt in enumerate(t):
            # Integrate to next time only if dt is above given threshold
            if tt - self.sparse_solver.t > MIN_SPARSE_DT:
                self.sparse_solver.set_f_params(self.predfunc(ysparse[i % npc]))
                self.sparse_solver.integrate(tt)
                if not self.sparse_solver.successful():
                    raise ValueError(self.timedlog('integration error', tt))

            # Assign solution values (computed and propagated) to sparse solution array
            y[i, self.is_dense_var] = ysparse[i % npc, self.is_dense_var]
            y[i, self.is_sparse_var] = self.sparse_solver.y

        # Append to global solution
        self.append(t, y)

    def solve(self, y0, events, tstop, update_interval, logfunc=None, **kwargs):
        ''' Integrate system using a hybrid scheme:

            - First, the full ODE system is integrated for a few cycles with a dense time
              granularity until a stopping criterion is met
            - Second, the profiles of all variables over the last cycle are downsampled to a
              far lower (i.e. sparse) sampling rate
            - Third, a subset of the ODE system is integrated with a sparse time granularity,
              for the remaining of the time interval, while the remaining variables are
              periodically expanded from their last cycle profile.
        '''
        # Sort events according to occurrence time
        events = self.sortEvents(events)

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')
        # Add None event at tstop
        events.append((tstop, None))

        # Initialize system
        self.initialize(y0)

        # Initialize event iterator
        ievent = iter(events)
        tevent, xevent = next(ievent)
        stop = False

        # While final event is not reached
        while not stop:
            # Determine end-time of current interval
            tend = min(tevent, self.t[-1] + update_interval)

            # If time interval encompasses at least one cycle, solve periodic system
            nmax = int(np.round((tend - self.t[-1]) / self.T))
            if nmax > 0:
                logger.debug(self.timedlog('integrating dense system'))
                PeriodicSolver.solve(self, None, nmax=nmax)

            # If end-time of current interval has been exceeded, bound solution to that time
            if self.t[-1] > tend:
                logger.debug(self.timedlog(f'bounding system at {self.timeStr(tend)}'))
                self.bound((self.t[0], tend))

            # If end-time of current interval has not been reached
            if self.t[-1] < tend:
                # Get solution over last cycle and resample it to sparse time step
                tlast, ylast = self.getCycle(-1)
                _, ysparse = self.resampleArrays(tlast, ylast, self.dt_sparse)

                # Integrate sparse system for the rest of the current interval
                logger.debug(self.timedlog(f'integrating sparse system until {self.timeStr(tend)}'))
                self.integrateSparse(ysparse, tend)

            # If end-time corresponds to event, fire it and move to next event
            if self.t[-1] == tevent:
                logger.debug(self.timedlog('firing event'))
                self.fireEvent(xevent)
                try:
                    tevent, xevent = next(ievent)
                except StopIteration:
                    stop = True
