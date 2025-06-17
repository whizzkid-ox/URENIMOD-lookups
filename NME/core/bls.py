# Author: Ryo Segawa
# Inspired by work of: https://github.com/tjjlemaire/PySONIC

from enum import Enum
import os
import json
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import brentq, curve_fit

from .model import Model
from .lookups import EffectiveVariablesLookup
from .solvers import PeriodicSolver
from .drives import Drive, AcousticDrive
from ..utils import logger, si_format, isIterable, LOOKUP_DIR
from ..constants import *


class PmCompMethod(Enum):
    ''' Enum: types of computation method for the intermolecular pressure '''
    direct = 1
    predict = 2


def LennardJones(x, beta, alpha, C, m, n):
    ''' Generic expression of a Lennard-Jones function, adapted for the context of
        symmetric deflection (distance = 2x).

        :param x: deflection (i.e. half-distance)
        :param beta: x-shifting factor
        :param alpha: x-scaling factor
        :param C: y-scaling factor
        :param m: exponent of the repulsion term
        :param n: exponent of the attraction term
        :return: Lennard-Jones potential at given distance (2x)
    '''
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))


def lookup(func):
    ''' Load parameters from lookup file, or compute them and store them in lookup file. '''

    lookup_path = os.path.join(os.path.split(__file__)[0], 'bls_lookups.json')

    def wrapper(obj):
        akey = f'{obj.a * 1e9:.1f}'
        Qkey = f'{obj.Qm0 * 1e5:.2f}'

        # Open lookup files
        try:
            with open(lookup_path, 'r') as fh:
                lookups = json.load(fh)
        except FileNotFoundError:
            lookups = {}

        # If info not in lookups, compute parameters and add them
        if akey not in lookups or Qkey not in lookups[akey]:
            func(obj)
            if akey not in lookups:
                lookups[akey] = {Qkey: {'LJ_approx': obj.LJ_approx, 'Delta_eq': obj.Delta}}
            else:
                lookups[akey][Qkey] = {'LJ_approx': obj.LJ_approx, 'Delta_eq': obj.Delta}
            logger.debug('Saving BLS derived parameters to lookup file')
            with open(lookup_path, 'w') as fh:
                json.dump(lookups, fh, indent=2)

        # If lookup exists, load parameters from it
        else:
            logger.debug('Loading BLS derived parameters from lookup file')
            obj.LJ_approx = lookups[akey][Qkey]['LJ_approx']
            obj.Delta = lookups[akey][Qkey]['Delta_eq']

    return wrapper


class BilayerSonophore(Model):
    ''' Definition of the Bilayer Sonophore Model
        - geometry
        - pressure terms
        - cavitation dynamics
    '''

    # BIOMECHANICAL PARAMETERS
    T = 309.15       # Temperature (K)
    delta0 = 2.0e-9  # Thickness of the leaflet (m)
    Delta_ = 1.4e-9  # Initial gap between the two leaflets on a non-charged membrane at equil. (m)
    pDelta = 1.0e5   # Attraction/repulsion pressure coefficient (Pa)
    m = 5.0          # Exponent in the repulsion term (dimensionless)
    n = 3.3          # Exponent in the attraction term (dimensionless)
    rhoL = 1075.0    # Density of the surrounding fluid (kg/m^3)
    muL = 7.0e-4     # Dynamic viscosity of the surrounding fluid (Pa.s)
    muS = 0.035      # Dynamic viscosity of the leaflet (Pa.s)
    kA = 0.24        # Area compression modulus of the leaflet (N/m)
    alpha = 7.56     # Tissue shear loss modulus frequency coefficient (Pa.s)
    C0 = 0.62        # Initial gas molar concentration in the surrounding fluid (mol/m^3)
    kH = 1.613e5     # Henry's constant (Pa.m^3/mol)
    P0 = 1.0e5       # Static pressure in the surrounding fluid (Pa)
    Dgl = 3.68e-9    # Diffusion coefficient of gas in the fluid (m^2/s)
    xi = 0.5e-9      # Boundary layer thickness for gas transport across leaflet (m)
    c = 1515.0       # Speed of sound in medium (m/s)

    # BIOPHYSICAL PARAMETERS
    epsilon0 = 8.854e-12  # Vacuum permittivity (F/m)
    epsilonR = 1.0        # Relative permittivity of intramembrane cavity (dimensionless)

    rel_Zmin = -0.49  # relative deflection range lower bound (in multiples of Delta)

    tscale = 'us'    # relevant temporal scale of the model
    simkey = 'MECH'  # keyword used to characterize simulations made with this model

    def __init__(self, a, Cm0, Qm0, embedding_depth=0.0):
        ''' Constructor of the class.
            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param Cm0: membrane resting capacitance (F/m2)
            :param Qm0: membrane resting charge density (C/m2)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''
        # Extract resting constants and geometry
        self.Cm0 = Cm0
        self.Qm0 = Qm0
        self.a = a
        self.d = embedding_depth
        self.S0 = np.pi * self.a**2

        # Initialize null elastic modulus for tissue
        self.kA_tissue = 0.

        # Compute Pm params
        self.computePMparams()

        # Compute initial volume and gas content
        self.V0 = np.pi * self.Delta * self.a**2
        self.ng0 = self.gasPa2mol(self.P0, self.V0)

    def copy(self):
        return self.__class__(self.a, self.Cm0, self.Qm0, embedding_depth=self.d)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value <= 0.:
            raise ValueError('Sonophore radius must be positive')
        self._a = value

    @property
    def Cm0(self):
        return self._Cm0

    @Cm0.setter
    def Cm0(self, value):
        if value <= 0.:
            raise ValueError('Resting membrane capacitance must be positive')
        self._Cm0 = value

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        if value < 0.:
            raise ValueError('Embedding depth cannot be negative')
        self._d = value

    def __repr__(self):
        s = f'{self.__class__.__name__}({self.a * 1e9:.1f} nm'
        if self.d > 0.:
            s += f', d={si_format(self.d, precision=1)}m'
        return f'{s})'

    @property
    def meta(self):
        return {
            'a': self.a,
            'd': self.d,
            'Cm0': self.Cm0,
            'Qm0': self.Qm0,
        }

    @classmethod
    def initFromMeta(cls, d):
        return cls(d['a'], d['Cm0'], d['Qm0'])

    @staticmethod
    def inputs():
        return {
            'a': {
                'desc': 'sonophore radius',
                'label': 'a',
                'unit': 'm',
                'precision': 0
            },
            'Qm': {
                'desc': 'membrane charge density',
                'label': 'Q_m',
                'unit': 'nC/cm^2',
                'factor': 1e5,
                'precision': 1
            },
            **AcousticDrive.inputs()
        }

    def filecodes(self, drive, Qm, PmCompMethod='predict'):
        if isIterable(Qm):
            Qm_code = f'{Qm.min() * 1e5:.1f}nCcm2_{Qm.max() * 1e5:.1f}nCcm2_{Qm.size}'
        else:
            Qm_code = f'{Qm * 1e5:.1f}nCcm2'
        return {
            'simkey': self.simkey,
            'a': f'{self.a * 1e9:.0f}nm',
            **drive.filecodes,
            'Qm': Qm_code
        }

    @staticmethod
    def getPltVars(wl='df["', wr='"]'):
        return {
            'Pac': {
                'desc': 'acoustic pressure',
                'label': 'P_{AC}',
                'unit': 'kPa',
                'factor': 1e-3,
                'func': f'meta["drive"].compute({wl}t{wr})'
            },
            'Z': {
                'desc': 'leaflets deflection',
                'label': 'Z',
                'unit': 'nm',
                'factor': 1e9,
                'bounds': (-1.0, 10.0)
            },

            'ng': {
                'desc': 'gas content',
                'label': 'n_g',
                'unit': '10^{-22}\ mol',
                'factor': 1e22,
                'bounds': (1.0, 15.0)
            },

            'Pmavg': {
                'desc': 'average intermolecular pressure',
                'label': 'P_M',
                'unit': 'kPa',
                'factor': 1e-3,
                'func': f'PMavgpred({wl}Z{wr})'
            },

            'Telastic': {
                'desc': 'leaflet elastic tension',
                'label': 'T_E',
                'unit': 'mN/m',
                'factor': 1e3,
                'func': f'TEleaflet({wl}Z{wr})'
            },

            'Cm': {
                'desc': 'membrane capacitance',
                'label': 'C_m',
                'unit': 'uF/cm^2',
                'factor': 1e2,
                'bounds': (0.0, 1.5),
                'func': f'v_capacitance({wl}Z{wr})'
            }
        }

    @property
    def pltScheme(self):
        return {
            'P_{AC}': ['Pac'],
            'Z': ['Z'],
            'n_g': ['ng']
        }

    @property
    def Zmin(self):
        return self.rel_Zmin * self.Delta

    def curvrad(self, Z):
        ''' Leaflet curvature radius
            (signed variable)

            :param Z: leaflet apex deflection (m)
            :return: leaflet curvature radius (m)
        '''
        if Z == 0.0:
            return np.inf
        else:
            return (self.a**2 + Z**2) / (2 * Z)

    def v_curvrad(self, Z):
        ''' Vectorized curvrad function '''
        return np.array(list(map(self.curvrad, Z)))

    def surface(self, Z):
        ''' Surface area of the stretched leaflet
            (spherical cap formula)

            :param Z: leaflet apex deflection (m)
            :return: stretched leaflet surface (m^2)
        '''
        return np.pi * (self.a**2 + Z**2)

    def volume(self, Z):
        ''' Volume of the inter-leaflet space
            (cylinder +/- 2 spherical caps)

            :param Z: leaflet apex deflection (m)
            :return: bilayer sonophore inner volume (m^3)
        '''
        return np.pi * self.a**2 * self.Delta\
            * (1 + (Z / (3 * self.Delta) * (3 + Z**2 / self.a**2)))

    def arealStrain(self, Z):
        ''' Areal strain of the stretched leaflet
            epsilon = (S - S0)/S0 = (Z/a)^2

            :param Z: leaflet apex deflection (m)
            :return: areal strain (dimensionless)
        '''
        return (Z / self.a)**2

    def logRelGap(self, Z):
        ''' Logarithm of relative sonophore deflection for a given deflection Z. '''
        return np.log((2 * Z + self.Delta) / self.Delta)

    def capacitance(self, Z):
        ''' Membrane capacitance
            (parallel-plate capacitor evaluated at average inter-layer distance)

            :param Z: leaflet apex deflection (m)
            :return: capacitance per unit area (F/m2)
        '''
        if Z == 0.0:
            return self.Cm0
        else:
            Z2 = (self.a**2 - Z**2 - Z * self.Delta) / (2 * Z)
            return self.Cm0 * self.Delta / self.a**2 * (Z + Z2 * self.logRelGap(Z))

    def v_capacitance(self, Z):
        ''' Vectorized capacitance function '''
        return np.array(list(map(self.capacitance, Z)))

    def derCapacitance(self, Z, U):
        ''' Evolution of membrane capacitance

            :param Z: leaflet apex deflection (m)
            :param U: leaflet apex deflection velocity (m/s)
            :return: time derivative of capacitance per unit area (F/m2.s)
        '''
        ratio1 = (Z**2 + self.a**2) / (Z * (2 * Z + self.Delta))
        ratio2 = (Z**2 + self.a**2) / (2 * Z**2) * self.logRelGap(Z)
        dCmdZ = self.Cm0 * self.Delta / self.a**2 * (ratio1 - ratio2)
        return dCmdZ * U

    @staticmethod
    def localDeflection(r, Z, R):
        ''' Local leaflet deflection at specific radial distance
            (signed)

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: local transverse leaflet deviation (m)
        '''
        if np.abs(Z) == 0.0:
            return 0.0
        else:
            return np.sign(Z) * (np.sqrt(R**2 - r**2) - np.abs(R) + np.abs(Z))

    def PMlocal(self, r, Z, R):
        ''' Local intermolecular pressure

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: local intermolecular pressure (Pa)
        '''
        z = self.localDeflection(r, Z, R)
        relgap = (2 * z + self.Delta) / self.Delta_
        return self.pDelta * ((1 / relgap)**self.m - (1 / relgap)**self.n)

    def PMavg(self, Z, R, S):
        ''' Average intermolecular pressure across the leaflet
            (computed by quadratic integration)

            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :param S: surface of the stretched leaflet (m^2)
            :return: averaged intermolecular resultant pressure (Pa)

            .. warning:: quadratic integration is computationally expensive.
        '''
        # Integrate intermolecular force over an infinitely thin ring of radius r from 0 to a
        fTotal, _ = integrate.quad(lambda r, Z, R: 2 * np.pi * r * self.PMlocal(r, Z, R),
                                   0, self.a, args=(Z, R))
        return fTotal / S

    def v_PMavg(self, Z, R, S):
        ''' Vectorized PMavg function '''
        return np.array(list(map(self.PMavg, Z, R, S)))

    def LJfitPMavg(self):
        ''' Determine optimal parameters of a Lennard-Jones expression
            approximating the average intermolecular pressure.

            These parameters are obtained by a nonlinear fit of the
            Lennard-Jones function for a range of deflection values
            between predetermined Zmin and Zmax.

            :return: 3-tuple with optimized LJ parameters for PmAvg prediction (Map) and
                the standard and max errors of the prediction in the fitting range (in Pascals)
        '''
        # Determine lower bound of deflection range: when Pm = Pmmax
        PMmax = LJFIT_PM_MAX  # Pa
        Zlb_range = (self.Zmin, 0.0)
        Zlb = brentq(lambda Z, Pmmax: self.PMavg(Z, self.curvrad(Z), self.surface(Z)) - PMmax,
                      *Zlb_range, args=(PMmax), xtol=1e-16)

        # Create vectors for geometric variables
        Zub = 2 * self.a
        Z = np.arange(Zlb, Zub, 1e-11)
        Pmavg = self.v_PMavg(Z, self.v_curvrad(Z), self.surface(Z))

        # Compute optimal nonlinear fit of custom LJ function with initial guess
        x0_guess = self.delta0
        C_guess = 0.1 * self.pDelta
        nrep_guess = self.m
        nattr_guess = self.n
        pguess = (x0_guess, C_guess, nrep_guess, nattr_guess)
        popt, _ = curve_fit(lambda x, x0, C, nrep, nattr:
                            LennardJones(x, self.Delta, x0, C, nrep, nattr),
                            Z, Pmavg, p0=pguess, maxfev=100000)
        (x0_opt, C_opt, nrep_opt, nattr_opt) = popt
        Pmavg_fit = LennardJones(Z, self.Delta, x0_opt, C_opt, nrep_opt, nattr_opt)

        # Compute prediction error
        residuals = Pmavg - Pmavg_fit
        ss_res = np.sum(residuals**2)
        N = residuals.size
        std_err = np.sqrt(ss_res / N)
        max_err = max(np.abs(residuals))

        logger.debug('LJ approx: x0 = %.2f nm, C = %.2f kPa, m = %.2f, n = %.2f',
                     x0_opt * 1e9, C_opt * 1e-3, nrep_opt, nattr_opt)

        LJ_approx = {"x0": x0_opt, "C": C_opt, "nrep": nrep_opt, "nattr": nattr_opt}
        return (LJ_approx, std_err, max_err)

    @lookup
    def computePMparams(self):
        # Find Delta that cancels out Pm + Pec at Z = 0 (m)
        if self.Qm0 == 0.0:
            D_eq = self.Delta_
        else:
            (D_eq, Pnet_eq) = self.findDeltaEq(self.Qm0)
            assert Pnet_eq < PNET_EQ_MAX, 'High Pnet at Z = 0 with ∆ = %.2f nm' % (D_eq * 1e9)
        self.Delta = D_eq

        # Find optimal Lennard-Jones parameters to approximate PMavg
        (self.LJ_approx, std_err, _) = self.LJfitPMavg()
        assert std_err < PMAVG_STD_ERR_MAX, 'High error in PmAvg nonlinear fit:'\
            ' std_err =  %.2f Pa' % std_err

    def PMavgpred(self, Z):
        ''' Approximated average intermolecular pressure
            (using nonlinearly fitted Lennard-Jones function)

            :param Z: leaflet apex deflection (m)
            :return: predicted average intermolecular pressure (Pa)
        '''
        return LennardJones(Z, self.Delta, self.LJ_approx['x0'], self.LJ_approx['C'],
                            self.LJ_approx['nrep'], self.LJ_approx['nattr'])

    def Pelec(self, Z, Qm):
        ''' Electrical pressure term

            :param Z: leaflet apex deflection (m)
            :param Qm: membrane charge density (C/m2)
            :return: electrical pressure (Pa)
        '''
        relS = self.S0 / self.surface(Z)
        abs_perm = self.epsilon0 * self.epsilonR  # F/m
        return - relS * Qm**2 / (2 * abs_perm)  # Pa

    def findDeltaEq(self, Qm):
        ''' Compute the Delta that cancels out the (Pm + Pec) equation at Z = 0
            for a given membrane charge density, using the Brent method to refine
            the pressure root iteratively.

            :param Qm: membrane charge density (C/m2)
            :return: equilibrium value (m) and associated pressure (Pa)
        '''
        def dualPressure(Delta):
            x = (self.Delta_ / Delta)
            return (self.pDelta * (x**self.m - x**self.n) + self.Pelec(0.0, Qm))
        Delta_eq = brentq(dualPressure, 0.1 * self.Delta_, 2.0 * self.Delta_, xtol=1e-16)
        logger.debug('∆eq = %.2f nm', Delta_eq * 1e9)
        return (Delta_eq, dualPressure(Delta_eq))

    def gasFlux(self, Z, P):
        ''' Gas molar flux through the sonophore boundary layers

            :param Z: leaflet apex deflection (m)
            :param P: internal gas pressure (Pa)
            :return: gas molar flux (mol/s)
        '''
        dC = self.C0 - P / self.kH
        return 2 * self.surface(Z) * self.Dgl * dC / self.xi

    @classmethod
    def gasmol2Pa(cls, ng, V):
        ''' Internal gas pressure for a given molar content

            :param ng: internal molar content (mol)
            :param V: sonophore inner volume (m^3)
            :return: internal gas pressure (Pa)
        '''
        return ng * Rg * cls.T / V

    @classmethod
    def gasPa2mol(cls, P, V):
        ''' Internal gas molar content for a given pressure

            :param P: internal gas pressure (Pa)
            :param V: sonophore inner volume (m^3)
            :return: internal gas molar content (mol)
        '''
        return P * V / (Rg * cls.T)

    def PtotQS(self, Z, ng, Qm, Pac, Pm_comp_method):
        ''' Net quasi-steady pressure for a given acoustic pressure
            (Ptot = Pm + Pg + Pec - P0 - Pac)

            :param Z: leaflet apex deflection (m)
            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: acoustic pressure (Pa)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: total balance pressure (Pa)
        '''
        if Pm_comp_method is PmCompMethod.direct:
            Pm = self.PMavg(Z, self.curvrad(Z), self.surface(Z))
        elif Pm_comp_method is PmCompMethod.predict:
            Pm = self.PMavgpred(Z)
        return Pm + self.gasmol2Pa(ng, self.volume(Z)) - self.P0 - Pac + self.Pelec(Z, Qm)

    def balancedefQS(self, ng, Qm, Pac=0.0, Pm_comp_method=PmCompMethod.predict):
        ''' Quasi-steady equilibrium deflection for a given acoustic pressure
            (computed by approximating the root of quasi-steady pressure)

            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: external acoustic perturbation (Pa)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: leaflet deflection canceling quasi-steady pressure (m)
        '''
        Zbounds = (self.Zmin, self.a)
        PQS = [self.PtotQS(x, ng, Qm, Pac, Pm_comp_method) for x in Zbounds]
        if not (PQS[0] > 0 > PQS[1]):
            s = 'P_QS not changing sign within [{:.2f}, {:.2f}] nm interval: '.format(
                *np.array(Zbounds) * 1e9)
            s += ', '.join([
                f'P_QS({Z * 1e9:.2f} nm) = {si_format(P, 2)}Pa' for Z, P in zip(Zbounds, PQS)])
            raise ValueError(s)
        return brentq(self.PtotQS, *Zbounds, args=(ng, Qm, Pac, Pm_comp_method), xtol=1e-16)

    def TEleaflet(self, Z):
        ''' Elastic tension in leaflet

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.kA * self.arealStrain(Z)

    def setTissueModulus(self, drive):
        ''' Set the frequency-dependent elastic modulus of the surrounding tissue. '''
        G_tissue = self.alpha * drive.modulationFrequency  # G'' (Pa)
        self.kA_tissue = 2 * G_tissue * self.d  # kA of the tissue layer (N/m)

    def TEtissue(self, Z):
        ''' Elastic tension in surrounding viscoelastic layer

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.kA_tissue * self.arealStrain(Z)

    def TEtot(self, Z):
        ''' Total elastic tension (leaflet + surrounding viscoelastic layer)

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.TEleaflet(Z) + self.TEtissue(Z)

    def PEtot(self, Z, R):
        ''' Total elastic tension pressure (leaflet + surrounding viscoelastic layer)

            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: elastic tension pressure (Pa)
        '''
        return - self.TEtot(Z) / R

    @classmethod
    def PVleaflet(cls, U, R):
        ''' Viscous stress pressure in leaflet

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: leaflet viscous stress pressure (Pa)
        '''
        return - 12 * U * cls.delta0 * cls.muS / R**2

    @classmethod
    def PVfluid(cls, U, R):
        ''' Viscous stress pressure in surrounding medium

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: fluid viscous stress pressure (Pa)
        '''
        return - 4 * U * cls.muL / np.abs(R)

    @classmethod
    def accP(cls, Ptot, R):
        ''' Leaflet transverse acceleration resulting from pressure imbalance

            :param Ptot: net pressure (Pa)
            :param R: leaflet curvature radius (m)
            :return: pressure-driven acceleration (m/s^2)
        '''
        return Ptot / (cls.rhoL * np.abs(R))

    @staticmethod
    def accNL(U, R):
        ''' Leaflet transverse nonlinear acceleration

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: nonlinear acceleration term (m/s^2)

            .. note:: A simplified version of nonlinear acceleration (neglecting
                dR/dH) is used here.
            '''
        # return - (3/2 - 2*R/H) * U**2 / R
        return -(3 * U**2) / (2 * R)

    @staticmethod
    def checkInputs(drive, Qm, Pm_comp_method):
        ''' Check validity of stimulation parameters

            :param drive: acoustic drive object
            :param Qm: imposed membrane charge density (C/m2)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
        '''
        if not isinstance(drive, Drive):
            raise TypeError(f'Invalid "drive" parameter (must be an "Drive" object)')

        if not (isinstance(Qm, float) or isIterable(Qm)):
            raise TypeError(f'Invalid "Qm" parameter (must be a scalar or T-periodic vector)')
        if isIterable(Qm):
            if len(Qm) != drive.nPerCycle:
                raise ValueError(
                    f'Qm size ({Qm.size}) differs from drive granularity ({drive.nPerCycle})')
        Qmin, Qmax = CHARGE_RANGE
        if np.min(Qm) < Qmin or np.max(Qm) > Qmax:
            raise ValueError(
                f'Invalid applied charge: {Qm * 1e5} nC/cm2 (must be within [{Qmin * 1e5}, {Qmax * 1e5}] interval')
        if not isinstance(Pm_comp_method, PmCompMethod):
            raise TypeError('Invalid Pm computation method (must be "PmCompmethod" type)')

    def derivatives(self, t, y, drive, Qm, Pm_comp_method=PmCompMethod.predict):
        ''' Evolution of the mechanical system

            :param t: time instant (s)
            :param y: vector of HH system variables at time t
            :param drive: acoustic drive object
            :param Qm: membrane charge density (F/m2)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: vector of mechanical system derivatives at time t
        '''
        # Split input vector explicitly
        U, Z, ng = y

        # Correct deflection value is below critical compression
        if Z < self.Zmin:
            logger.warning('Deflection out of range: Z = %.2f nm', Z * 1e9)
            Z = self.Zmin

        # Compute curvature radius
        R = self.curvrad(Z)

        # Compute total pressure
        Pg = self.gasmol2Pa(ng, self.volume(Z))
        if Pm_comp_method is PmCompMethod.direct:
            Pm = self.PMavg(Z, self.curvrad(Z), self.surface(Z))
        elif Pm_comp_method is PmCompMethod.predict:
            Pm = self.PMavgpred(Z)
        Pac = drive.compute(t)
        Pv = self.PVleaflet(U, R) + self.PVfluid(U, R)
        Ptot = Pm + Pg - self.P0 - Pac + self.PEtot(Z, R) + Pv + self.Pelec(Z, Qm)

        # Compute derivatives
        dUdt = self.accP(Ptot, R) + self.accNL(U, R)
        dZdt = U
        dngdt = self.gasFlux(Z, Pg)

        # Return derivatives vector
        return [dUdt, dZdt, dngdt]

    def computeInitialDeflection(self, drive, Qm, dt, Pm_comp_method=PmCompMethod.predict):
        ''' Compute non-zero deflection value for a small perturbation
            (solving quasi-steady equation).
        '''
        Pac = drive.compute(dt)
        return self.balancedefQS(self.ng0, Qm, Pac, Pm_comp_method)

    @classmethod
    @Model.checkOutputDir
    def simQueue(cls, freqs, amps, charges, **kwargs):
        drives = AcousticDrive.createQueue(freqs, amps)
        queue = []
        for drive in drives:
            for Qm in charges:
                queue.append([drive, Qm])
        return queue

    def initialConditions(self, *args, **kwargs):
        ''' Compute simulation initial conditions. '''
        # Compute initial non-zero deflection
        Z = self.computeInitialDeflection(*args, **kwargs)

        # Return initial conditions dictionary
        return {
            'U': [0.] * 2,
            'Z': [0., Z],
            'ng': [self.ng0] * 2,
        }

    def simCycles(self, drive, Qm, nmax=None, nmin=None, Pm_comp_method=PmCompMethod.predict):
        ''' Simulate for a specific number of cycles or until periodic stabilization,
            for a specific set of ultrasound parameters, and return output data in a dataframe.

            :param drive: acoustic drive object
            :param Qm: imposed membrane charge density (C/m2)
            :param n: number of cycles (optional)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: output dataframe
        '''
        # Set the tissue elastic modulus
        self.setTissueModulus(drive)

        # Adapt Qm(t) function to charge input type
        if isinstance(Qm, float):
            # If float, simply use a constant applied charge
            Qm0, Qm_t = Qm, lambda t: Qm
        elif isIterable(Qm):
            # If iterable, recast as 1d array, check size, and use a time-varying charge
            Qm0, Qm_t = Qm[0], lambda t: Qm[int((t % drive.periodicity) / drive.dt)]
        else:
            raise ValueError('unknown charge input type')

        # Compute initial conditions
        y0 = self.initialConditions(drive, Qm0, drive.dt, Pm_comp_method=Pm_comp_method)

        # Initialize solver and compute solution
        solver = PeriodicSolver(
            drive.periodicity,                                                    # periodicty
            y0.keys(),                                                            # variables list
            lambda t, y: self.derivatives(t, y, drive, Qm_t(t), Pm_comp_method),  # dfunc
            primary_vars=['Z', 'ng'],                                             # primary variables
            dt=drive.dt                                                           # time step
        )
        data = solver(y0, nmax=nmax, nmin=nmin)

        # Remove velocity timeries from solution
        del data['U']

        # Return solution dataframe
        return data

    @Model.addMeta
    @Model.logDesc
    @Model.checkSimParams
    def simulate(self, drive, Qm, Pm_comp_method=PmCompMethod.predict):
        ''' Wrapper around the simUntilConvergence method, with decorators. '''
        return self.simCycles(drive, Qm, Pm_comp_method=Pm_comp_method)

    def desc(self, meta):
        Qm = meta['Qm']
        if isIterable(Qm):
            Qstr = f'US-periodic function within [{Qm.min() * 1e5:.2f}, {Qm.max() * 1e5:.2f}] nC/cm2'
        else:
            Qstr = f'{si_format(Qm * 1e-4, 2)}C/cm2'
        return f'{self}: simulation @ {meta["drive"].desc}, Q = {Qstr}'

    @Model.logDesc
    def getZlast(self, drive, Qm):
        ''' Run simulation and extract deflection vector from last cycle. '''
        return self.simCycles(drive, Qm).tail(NPC_DENSE)['Z'].values  # m

    def getRelCmCycle(self, *args, **kwargs):
        ''' Run simulation and extract relative capacitance vector from last cycle. '''
        return self.v_capacitance(self.getZlast(*args, **kwargs)) / self.Cm0

    @property
    def Cm_lkp_filename(self):
        return f'Cm_lkp_{self.a * 1e9:.0f}nm.pkl'

    @property
    def Cm_lkp_filepath(self):
        return os.path.join(LOOKUP_DIR, self.Cm_lkp_filename)

    @property
    def Cm_lkp(self):
        return EffectiveVariablesLookup.fromPickle(self.Cm_lkp_filepath)

    def getGammaLookup(self):
        return self.Cm_lkp.reduce(lambda x, **kwargs: np.ptp(x, **kwargs) / 2, 't')
