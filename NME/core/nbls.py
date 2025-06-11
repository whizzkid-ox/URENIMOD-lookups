# Author: Ryo Segawa
# Inspired by work of: https://github.com/tjjlemaire/PySONIC

import logging
import numpy as np

from .solvers import EventDrivenSolver, HybridSolver
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from .model import Model
from .drives import AcousticDrive, ElectricDrive
from .protocols import *
from ..utils import *
from ..constants import *
from ..postpro import getFixedPoints
from .lookups import EffectiveVariablesLookup
from ..neurons import getPointNeuron


class NeuronalBilayerSonophore(BilayerSonophore):
    ''' This class inherits from the BilayerSonophore class and receives an PointNeuron instance
        at initialization, to define the electro-mechanical NICE model and its SONIC variant. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ASTIM'  # keyword used to characterize simulations made with this model

    def __init__(self, a, pneuron, embedding_depth=0.0):
        ''' Constructor of the class.

            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param pneuron: point-neuron model
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''
        self.pneuron = pneuron
        super().__init__(a, pneuron.Cm0, pneuron.Qm0, embedding_depth=embedding_depth)

    @property
    def a_str(self):
        return f'{self.a * 1e9:.1f} nm'

    def __repr__(self):
        s = f'{self.__class__.__name__}({self.a_str}, {self.pneuron}'
        if self.d > 0.:
            s += f', d={si_format(self.d, precision=1)}m'
        return f'{s})'

    def copy(self):
        return self.__class__(self.a, self.pneuron, embedding_depth=self.d)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.a == other.a and self.pneuron == other.pneuron and self.d == other.d

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        if not isinstance(value, PointNeuron):
            raise ValueError(f'{value} is not a valid PointNeuron instance')
        if not hasattr(self, '_pneuron') or value != self._pneuron:
            self._pneuron = value
            if hasattr(self, 'a'):
                super().__init__(self.a, self.pneuron.Cm0, self.pneuron.Qm0, embedding_depth=self.d)

    @property
    def meta(self):
        return {
            'neuron': self.pneuron.name,
            'a': self.a,
            'd': self.d,
        }

    @classmethod
    def initFromMeta(cls, meta):
        return cls(meta['a'], getPointNeuron(meta['neuron']), embedding_depth=meta['d'])

    def params(self):
        return {**super().params(), **self.pneuron.params()}

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright),
                **self.pneuron.getPltVars(wrapleft, wrapright)}

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    def filecode(self, *args):
        return Model.filecode(self, *args)

    @staticmethod
    def inputs():
        # Get parent input vars and supress irrelevant entries
        inputvars = BilayerSonophore.inputs()
        del inputvars['Qm']

        # Fill in current input vars in appropriate order
        inputvars.update({
            **AcousticDrive.inputs(),
            'fs': {
                'desc': 'sonophore membrane coverage fraction',
                'label': 'f_s',
                'unit': '\%',
                'factor': 1e2,
                'precision': 0
            },
            'method': None
        })
        return inputvars

    def filecodes(self, drive, pp, fs, method, qss_vars):
        codes = {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nature': pp.nature,
            'a': f'{self.a * 1e9:.0f}nm',
            **drive.filecodes,
            **pp.filecodes,
        }
        codes['fs'] = f'fs{fs * 1e2:.0f}%' if fs < 1 else None
        codes['method'] = method
        codes['qss_vars'] = qss_vars
        return codes

    @staticmethod
    def interpEffVariable(key, Qm, stim, lkp):
        ''' Interpolate Q-dependent effective variable along various stimulation states of a solution.

            :param key: lookup variable key
            :param Qm: charge density solution vector
            :param stim: stimulation state solution vector
            :param lkp: 2D lookup object
            :return: interpolated effective variable vector
        '''
        x = np.zeros(stim.size)
        stim_vals = np.unique(stim)
        for s in stim_vals:
            x[stim == s] = lkp.project('A', s).interpVar1D(Qm[stim == s], key)
        return x

    @staticmethod
    def spatialAverage(fs, x, x0):
        ''' fs-modulated spatial averaging. '''
        return fs * x + (1 - fs) * x0

    @timer
    def computeEffVars(self, drive, fs, Qm0, Qm_overtones=None):
        ''' Compute "effective" coefficients of the HH system for a specific
            acoustic stimulus and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param drive: acoustic drive object
            :param fs: list of sonophore membrane coverage fractions
            :param Qm: imposed charge density (C/m2)
            :return: list with computation time and a list of dictionaries of effective variables
        '''
        if not isIterable(fs):
            fs = [fs]
        if Qm_overtones is None:
            # Constant Qm profile
            Qm_cycle = Qm0
            novertones = 0
        else:
            # Qm profile as Fourier series
            A_Qm, phi_Qm = list(zip(*Qm_overtones))
            Qm_fft = np.hstack(([Qm0 + 0j], A_Qm * (np.cos(phi_Qm) + 1j * np.sin(phi_Qm))))
            Qm_cycle = np.fft.irfft(Qm_fft, n=drive.nPerCycle) * drive.nPerCycle
            novertones = len(A_Qm)

        # Run simulation and extract capacitance vector from last cycle
        Z_cycle = super().simCycles(drive, Qm_cycle).tail(drive.nPerCycle)['Z'].values  # m
        Cm_cycle = self.v_capacitance(Z_cycle)  # F/m2

        # For each coverage fraction
        effvars_list = []
        for x in fs:
            # Compute membrane potential vector
            Vm_cycle = Qm_cycle / self.spatialAverage(x, Cm_cycle, self.Cm0) * 1e3  # mV

            # Compute effective (cycle-average) membrane potential
            effvars = {'V': np.mean(Vm_cycle)}

            # If Qm overtones were provided, compute Vm overtones
            if novertones > 0:
                # classic Fourier coefficients
                Vm_coeffs = np.fft.rfft(Vm_cycle)[:novertones + 1] / drive.nPerCycle
                # amplitude-phase formalism
                A_Vm, phi_Vm = np.abs(Vm_coeffs), np.angle(Vm_coeffs)
                for i in range(1, novertones + 1):
                    effvars[f'A_V{i}'] = A_Vm[i]
                    effvars[f'phi_V{i}'] = phi_Vm[i]

            # Add computed effective rates
            effvars.update(self.pneuron.getEffRates(Vm_cycle))

            # Append to list
            effvars_list.append(effvars)

        # Log process
        log = f'{self}: lookups @ {drive.desc}, Qm0 = {Qm0 * 1e5:.2f} nC/cm2'
        if Qm_overtones is not None:
            log = log + ', ' + ', '.join([
                f'Qm{i + 1} = ({x[0] * 1e5:.2f} nC/cm2, {x[1]:.2f} rad)'
                for i, x in enumerate(Qm_overtones)])
        if len(fs) > 1:
            log += f', fs = {fs.min() * 1e2:.0f} - {fs.max() * 1e2:.0f}%'
        else:
            log += f', fs = {fs[0] * 1e2:.0f}%'
        logger.info(log)

        # Return effective coefficients
        return effvars_list

    def getLookupFileName(self, a=None, f=None, A=None, fs=None, novertones=0.):
        if all(x is None for x in [a, f, A, fs]):
            fs = 1.
        try:
            fname = f'{self.pneuron.lookup_name}_lookups'
        except AttributeError:
            fname = f'{self.pneuron.name}_lookups'
        if a is not None:
            fname += f'_{a * 1e9:.0f}nm'
        if f is not None:
            fname += f'_{f * 1e-3:.0f}kHz'
        if A is not None:
            fname += f'_{A * 1e-3:.0f}kPa'
        if fs is not None:
            fname += f'_fs{fs:.2f}'
        if novertones > 0:
            fname += f'_{novertones}overtones'
        return f'{fname}.pkl'

    def getLookupFilePath(self, *args, **kwargs):
        return os.path.join(LOOKUP_DIR, self.getLookupFileName(*args, **kwargs))

    def getLookup(self, *args, **kwargs):
        keep_tcomp = kwargs.pop('keep_tcomp', False)
        lookup_path = self.getLookupFilePath(*args, **kwargs)
        lkp = EffectiveVariablesLookup.fromPickle(lookup_path)
        if not keep_tcomp:
            del lkp.tables['tcomp']
        return lkp

    def getLookup2D(self, f, fs):
        proj_kwargs = {'a': self.a, 'f': f, 'fs': fs}
        proj_str = f'a = {si_format(self.a)}m, f = {si_format(f)}Hz, fs = {fs * 1e2:.0f}%'
        logger.debug(f'loading {self.pneuron} lookup for {proj_str}')
        if fs < 1.:
            kwargs = proj_kwargs.copy()
            kwargs['fs'] = None
        else:
            kwargs = {'fs': fs}
        return self.getLookup(**kwargs).projectN(proj_kwargs)

    def fullDerivatives(self, t, y, drive, fs):
        ''' Compute the full system derivatives.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param drive: acoustic drive object
            :param fs: sonophore membrane coverage fraction (-)
            :return: vector of derivatives
        '''
        dydt_mech = BilayerSonophore.derivatives(
            self, t, y[:3], drive, y[3])
        dydt_elec = self.pneuron.derivatives(
            t, y[3:], Cm=self.spatialAverage(fs, self.capacitance(y[1]), self.Cm0))
        return dydt_mech + dydt_elec

    def effDerivatives(self, t, y, lkp1d, qss_vars):
        ''' Compute the derivatives of the n-ODE effective system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param lkp: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :param qss_vars: list of QSS variables
            :return: vector of effective system derivatives at time t
        '''
        # Unpack values and interpolate lookup at current charge density
        Qm, *states = y
        lkp0d = lkp1d.interpolate1D(Qm)

        # Compute states dictionary from differential and QSS variables
        states_dict = {}
        i = 0
        for k in self.pneuron.statesNames():
            if k in qss_vars:
                states_dict[k] = self.pneuron.quasiSteadyStates()[k](lkp0d)
            else:
                states_dict[k] = states[i]
                i += 1

        # Compute charge density derivative
        dQmdt = - self.pneuron.iNet(lkp0d['V'], states_dict) * 1e-3

        # Compute states derivative vector only for differential variable
        dstates = []
        for k in self.pneuron.statesNames():
            if k not in qss_vars:
                dstates.append(self.pneuron.derEffStates()[k](lkp0d, states_dict))

        return [dQmdt, *dstates]

    def deflectionDependentVm(self, Qm, Z, fs):
        ''' Compute deflection (and sonophore coverage fraction) dependent voltage profile. '''
        return Qm / self.spatialAverage(fs, self.v_capacitance(Z), self.Cm0) * 1e3  # mV

    def fullInitialConditions(self, *args, **kwargs):
        ''' Compute simulation initial conditions. '''
        y0 = super().initialConditions(*args, **kwargs)
        y0.update({
            'Qm': [self.Qm0] * 2,
            **{k: [self.pneuron.steadyStates()[k](self.pneuron.Vm0)] * 2
               for k in self.pneuron.statesNames()}
        })
        return y0

    def __simFull(self, drive, pp, fs):
        # Compute initial conditions
        y0 = self.fullInitialConditions(drive, self.Qm0, drive.dt)

        # Initialize solver and compute solution
        solver = EventDrivenSolver(
            lambda x: setattr(solver.drive, 'xvar', drive.xvar * x),    # eventfunc
            y0.keys(),                                                  # variables list
            lambda t, y: self.fullDerivatives(t, y, solver.drive, fs),  # dfunc
            event_params={'drive': drive.copy().updatedX(0.)},          # event parameters
            dt=drive.dt)                                                # time step
        data = solver(
            y0, pp.stimEvents(), pp.tstop, target_dt=CLASSIC_TARGET_DT,
            log_period=pp.tstop / 100 if logger.getEffectiveLevel() <= logging.INFO else None,
            # logfunc=lambda y: f'Qm = {y[3] * 1e5:.2f} nC/cm2'
        )

        # Remove velocity and add voltage timeseries to solution
        del data['U']
        data.addColumn(
            'Vm', self.deflectionDependentVm(data['Qm'], data['Z'], fs), preceding_key='Qm')

        # Return solution dataframe
        return data

    def __simHybrid(self, drive, pp, fs):
        # Compute initial conditions
        y0 = self.fullInitialConditions(drive, self.Qm0, drive.dt)

        # Initialize solver and compute solution
        solver = HybridSolver(
            y0.keys(),
            lambda t, y: self.fullDerivatives(t, y, solver.drive, fs),  # dfunc
            lambda t, y, Cm: self.pneuron.derivatives(
                t, y, Cm=self.spatialAverage(fs, Cm, self.Cm0)),        # dfunc_sparse
            lambda yref: self.capacitance(yref[1]),                     # predfunc
            lambda x: setattr(solver.drive, 'xvar', drive.xvar * x),    # eventfunc
            drive.periodicity,                                          # periodicity
            ['U', 'Z', 'ng'],                                           # fast-evolving variables
            drive.dt,                                                   # dense time step
            drive.dt_sparse,                                            # sparse time step
            event_params={'drive': drive.copy().updatedX(0.)},          # event parameters
            primary_vars=['Z', 'ng']                                    # primary variables
        )
        data = solver(
            y0, pp.stimEvents(), pp.tstop, HYBRID_UPDATE_INTERVAL, target_dt=CLASSIC_TARGET_DT,
            log_period=pp.tstop / 100 if logger.getEffectiveLevel() < logging.INFO else None,
            logfunc=lambda y: f'Qm = {y[3] * 1e5:.2f} nC/cm2'
        )

        # Remove velocity and add voltage timeseries to solution
        del data['U']
        data.addColumn(
            'Vm', self.deflectionDependentVm(data['Qm'], data['Z'], fs), preceding_key='Qm')

        # Return solution dataframe
        return data

    def __simSonic(self, drive, pp, fs, qss_vars=None, pavg=False):
        # Load appropriate 2D lookup
        lkp = self.getLookup2D(drive.f, fs)

        # Adapt lookup and pulsing protocol if pulse-average mode is selected
        if pavg:
            lkp = lkp * pp.DC + lkp.project('A', 0.).tile('A', lkp.refs['A']) * (1 - pp.DC)
            tstim = (int(pp.tstim * pp.PRF) - 1 + pp.DC) / pp.PRF
            pp = TimeProtocol(tstim, pp.tstim + pp.toffset - tstim)

        # Determine QSS and differential variables, and create optional QSS lookup
        if qss_vars is None:
            qss_vars = []
        else:
            lkp_QSS = EffectiveVariablesLookup(
                lkp.refs, {k: self.pneuron.quasiSteadyStates()[k](lkp) for k in qss_vars})
        diff_vars = [item for item in self.pneuron.statesNames() if item not in qss_vars]

        # Set initial conditions
        y0 = {
            'Qm': self.Qm0,
            **{k: self.pneuron.steadyStates()[k](self.pneuron.Vm0) for k in diff_vars}
        }

        # Initialize solver and compute solution
        solver = EventDrivenSolver(
            lambda x: setattr(solver, 'lkp', lkp.project('A', drive.xvar * x)),  # eventfunc
            y0.keys(),                                                           # variables list
            lambda t, y: self.effDerivatives(t, y, solver.lkp, qss_vars),        # dfunc
            event_params={'lkp': lkp.project('A', 0.)},                          # event parameters
            dt=self.pneuron.chooseTimeStep())                                    # time step
        data = solver(
            y0, pp.stimEvents(), pp.tstop,
            log_period=pp.tstop / 100 if pp.tstop >= 5 else None,
            max_nsamples=MAX_NSAMPLES_EFFECTIVE)

        # Interpolate Vm and QSS variables along charge vector and store them in solution dataframe
        data.addColumn(
            'Vm', self.interpEffVariable('V', data['Qm'], data.stim * drive.A, lkp),
            preceding_key='Qm')
        for k in qss_vars:
            data[k] = self.interpEffVariable(k, data['Qm'], data.stim * drive.A, lkp_QSS)

        # Add dummy deflection and gas content vectors to solution
        for key in ['Z', 'ng']:
            data[key] = np.full(data['t'].size, np.nan)

        # Return solution dataframe
        return data

    def intMethods(self):
        ''' Listing of model integration methods. '''
        return {
            'full': self.__simFull,
            'hybrid': self.__simHybrid,
            'sonic': self.__simSonic
        }

    @classmethod
    @Model.checkOutputDir
    def simQueue(cls, freqs, amps, durations, offsets, PRFs, DCs, fs, methods, qss_vars, **kwargs):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param freqs: list (or 1D-array) of US frequencies
            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :param fs: sonophore membrane coverage fractions (-)
            :params methods: integration methods
            :param qss_vars: QSS variables
            :return: list of parameters (list) for each simulation
        '''
        if ('full' in methods or 'hybrid' in methods) and kwargs['outputdir'] is None:
            logger.warning('Running cumbersome simulation(s) without file saving')
        if amps is None:
            amps = [None]
        drives = AcousticDrive.createQueue(freqs, amps)
        protocols = PulsedProtocol.createQueue(durations, offsets, PRFs, DCs)
        queue = []
        for drive in drives:
            for pp in protocols:
                for cov in fs:
                    for method in methods:
                        queue.append([drive, pp, cov, method, qss_vars])
        return queue

    @classmethod
    @Model.checkOutputDir
    def simQueueBurst(cls, freqs, amps, durations, PRFs, DCs, BRFs, nbursts,
                      fs, methods, qss_vars, **kwargs):
        if ('full' in methods or 'hybrid' in methods) and kwargs['outputdir'] is None:
            logger.warning('Running cumbersome simulation(s) without file saving')
        if amps is None:
            amps = [None]
        drives = AcousticDrive.createQueue(freqs, amps)
        protocols = BurstProtocol.createQueue(durations, PRFs, DCs, BRFs, nbursts)
        queue = []
        for drive in drives:
            for pp in protocols:
                for cov in fs:
                    for method in methods:
                        queue.append([drive, pp, cov, method, qss_vars])
        return queue

    def checkInputs(self, drive, pp, fs, method, qss_vars):
        PointNeuron.checkInputs(drive, pp)
        _, xevents, = zip(*pp.stimEvents())
        if np.any(np.array([xevents]) < 0.):
            raise ValueError('Invalid time protocol: contains negative modulators')
        if not isinstance(fs, float):
            raise TypeError(f'Invalid "fs" parameter (must be float typed)')
        if qss_vars is not None:
            if not isIterable(qss_vars) or not isinstance(qss_vars[0], str):
                raise ValueError('Invalid QSS variables: must be None or an iterable of strings')
            sn = self.pneuron.statesNames()
            for item in qss_vars:
                if item not in sn:
                    raise ValueError(f'Invalid QSS variable: {item} (must be in {sn}')
        if method not in list(self.intMethods().keys()):
            raise ValueError(f'Invalid integration method: "{method}"')

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    @Model.checkSimParams
    def simulate(self, drive, pp, fs=1., method='sonic', qss_vars=None):
        ''' Simulate the electro-mechanical model for a specific set of US stimulation parameters,
            and return output data in a dataframe.

            :param drive: acoustic drive object
            :param pp: pulse protocol object
            :param fs: sonophore membrane coverage fraction (-)
            :param method: selected integration method
            :return: output dataframe
        '''
        # Set the tissue elastic modulus
        self.setTissueModulus(drive)

        # Call appropriate simulation function and return
        simfunc = self.intMethods()[method]
        simargs = [drive, pp, fs]
        if method == 'sonic':
            simargs.append(qss_vars)
        return simfunc(*simargs)

    def desc(self, meta):
        method = meta['method'] if 'method' in meta else meta['model']['method']
        fs = meta['fs'] if 'fs' in meta else meta['model']['fs']
        s = f'{self}: {method} simulation @ {meta["drive"].desc}, {meta["pp"].desc}'
        if fs < 1.0:
            s += f', fs = {(fs * 1e2):.2f}%'
        if 'qss_vars' in meta and meta['qss_vars'] is not None:
            s += f" - QSS ({', '.join(meta['qss_vars'])})"
        return s

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    def getArange(self, drive):
        return (0., self.getLookup().refs['A'].max())

    @property
    def titrationFunc(self):
        return self.pneuron.titrationFunc

    @logCache(os.path.join(os.path.split(__file__)[0], 'astim_titrations.log'))
    def titrate(self, drive, pp, fs=1., method='sonic', qss_vars=None, xfunc=None, Arange=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given frequency and pulsed protocol.

            :param drive: unresolved acoustic drive object
            :param pp: pulse protocol object
            :param fs: sonophore membrane coverage fraction (-)
            :param method: integration method
            :return: determined threshold amplitude (Pa)
        '''
        return super().titrate(drive, pp, fs=fs, method=method, qss_vars=qss_vars,
                               xfunc=xfunc, Arange=Arange)

    def getQuasiSteadyStates(self, f, amps=None, charges=None, DC=1.0, squeeze_output=False):
        ''' Compute the quasi-steady state values of the neuron's gating variables
            for a combination of US amplitudes, charge densities,
            at a specific US frequency and duty cycle.

            :param f: US frequency (Hz)
            :param amps: US amplitudes (Pa)
            :param charges: membrane charge densities (C/m2)
            :param DC: duty cycle
            :return: 4-tuple with reference values of US amplitude and charge density,
                as well as interpolated Vmeff and QSS gating variables
        '''
        # Get DC-averaged lookups interpolated at the appropriate amplitudes and charges
        lkp = self.getLookup().projectDC(amps=amps, DC=DC).projectN({'a': self.a, 'f': f})
        if charges is not None:
            lkp = lkp.project('Q', charges)

        # Specify dimensions with A as the first axis
        lkp.move('A', 0)

        # Compute QSS states using these lookups
        QSS = EffectiveVariablesLookup(
            lkp.refs,
            {k: v(lkp) for k, v in self.pneuron.quasiSteadyStates().items()})

        # Compress outputs if needed
        if squeeze_output:
            QSS = QSS.squeeze()
            lkp = lkp.squeeze()

        return lkp, QSS

    def iNetQSS(self, Qm, f, A, DC):
        ''' Compute quasi-steady state net membrane current for a given combination
            of US parameters and a given membrane charge density.

            :param Qm: membrane charge density (C/m2)
            :param f: US frequency (Hz)
            :param A: US amplitude (Pa)
            :param DC: duty cycle (-)
            :return: net membrane current (mA/m2)
        '''
        lkp, QSS = self.getQuasiSteadyStates(
            f, amps=A, charges=Qm, DC=DC, squeeze_output=True)
        return self.pneuron.iNet(lkp['V'], QSS)  # mA/m2

    def fixedPointsQSS(self, f, A, DC, lkp, dQdt):
        ''' Compute QSS fixed points along the charge dimension for a given combination
            of US parameters, and determine their stability.

            :param f: US frequency (Hz)
            :param A: US amplitude (Pa)
            :param DC: duty cycle (-)
            :param lkp: lookup dictionary for effective variables along charge dimension
            :param dQdt: charge derivative profile along charge dimension
            :return: 2-tuple with values of stable and unstable fixed points
        '''
        logger.debug(f'A = {A * 1e-3:.2f} kPa, DC = {DC * 1e2:.0f}%')

        # Extract fixed points from QSS charge variation profile
        def dfunc(Qm):
            return - self.iNetQSS(Qm, f, A, DC)
        fixed_points = getFixedPoints(
            lkp.refs['Q'], dQdt, filter='both', der_func=dfunc).tolist()
        dfunc = lambda x: np.array(self.effDerivatives(_, x, lkp))

        # classified_fixed_points = {'stable': [], 'unstable': [], 'saddle': []}
        classified_fixed_points = []

        np.set_printoptions(precision=2)

        # For each fixed point
        for i, Qm in enumerate(fixed_points):

            # Re-compute QSS at fixed point
            *_, QSS = self.getQuasiSteadyStates(f, amps=A, charges=Qm, DC=DC,
                                                squeeze_output=True)

            # Classify fixed point stability by numerically evaluating its Jacobian and
            # computing its eigenvalues
            x = np.array([Qm, *QSS.tables.values()])
            eigvals, key = classifyFixedPoint(x, dfunc)
            # classified_fixed_points[key].append(Qm)

            classified_fixed_points.append((x, eigvals, key))
            # eigenvalues.append(eigvals)
            logger.debug(f'{key} point @ Q = {(Qm * 1e5):.1f} nC/cm2')

        # eigenvalues = np.array(eigenvalues).T
        # print(eigenvalues.shape)

        return classified_fixed_points

    def isStableQSS(self, f, A, DC):
        lookups, QSS = self.getQuasiSteadyStates(
            f, amps=A, DCs=DC, squeeze_output=True)
        dQdt = -self.pneuron.iNet(lookups['V'], QSS.tables)  # mA/m2
        classified_fixed_points = self.fixedPointsQSS(f, A, DC, lookups, dQdt)
        return len(classified_fixed_points['stable']) > 0


class DrivenNeuronalBilayerSonophore(NeuronalBilayerSonophore):

    simkey = 'DASTIM'  # keyword used to characterize simulations made with this model

    def __init__(self, Idrive, *args, **kwargs):
        self.Idrive = Idrive
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().__repr__()[:-1] + f', Idrive = {self.Idrive:.2f} mA/m2)'

    @classmethod
    def initFromMeta(cls, meta):
        return cls(meta['Idrive'], meta['a'], getPointNeuron(meta['neuron']),
                   embedding_depth=meta['d'])

    def params(self):
        return {**{'Idrive': self.Idrive}, **super().params()}

    @staticmethod
    def inputs():
        return {
            **NeuronalBilayerSonophore.inputs(),
            'Idrive': ElectricDrive.inputs()['I']
        }

    @property
    def meta(self):
        return {
            **super().meta,
            'Idrive': self.Idrive
        }

    def filecodes(self, *args):
        return {
            **super().filecodes(*args),
            'Idrive': f'Idrive{self.Idrive:.1f}mAm2'
        }

    def fullDerivatives(self, *args):
        dydt = super().fullDerivatives(*args)
        dydt[3] += self.Idrive * 1e-3
        return dydt

    def effDerivatives(self, *args):
        dQmdt, *dstates = super().effDerivatives(*args)
        dQmdt += self.Idrive * 1e-3
        return [dQmdt, *dstates]
