"""Simple argument parser for mechanical simulations."""

import logging
import numpy as np
from argparse import ArgumentParser
from .utils import isIterable, logger

class Parser(ArgumentParser):
    """Basic argument parser with defaults and logging."""

    def __init__(self):
        super().__init__()
        self.defaults = {}
        self.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')

    def parse2array(self, args, key, factor=1):
        """Convert argument value to numpy array with optional scaling factor."""
        return np.array(args[key]) * factor

    def parse(self):
        args = vars(super().parse_args())
        # Apply defaults
        for k, v in self.defaults.items():
            if k in args and args[k] is None:
                args[k] = v if isIterable(v) else [v]
        # Set log level
        args['loglevel'] = logging.DEBUG if args.pop('verbose') else logging.INFO
        return args

class MechSimParser(Parser):
    """Parser for mechanical simulations."""

    def __init__(self, outputdir=None):
        super().__init__()
        # Initialize defaults
        self.defaults = {
            'fiber_length': np.array([1e-4]),        # 0.1 mm
            'fiber_diameter': np.array([1e-6]),      # 1.0 µm
            'membrane_thickness': np.array([1.4e-9]), # 1.4 nm
            'freq': np.array([1e3]),                 # 1 kHz
            'amp': np.array([1e3]),                  # 1 kPa
            'charge': np.array([0.0]),               # C/m²
            'tstim': np.array([0.0]),               # s
            'toffset': np.array([0.0]),             # s
            'PRF': np.array([0.0])                  # Hz
        }
        
        # Initialize unit conversion factors (all 1.0 since we use SI units)
        self.factors = {k: 1.0 for k in self.defaults.keys()}
        # Add additional factors for temporal parameters
        self.factors.update({
            'tstim': 1.0,    # time in seconds
            'toffset': 1.0,  # time in seconds
            'PRF': 1.0,      # frequency in Hz
            'DC': 1.0        # duty cycle (unitless)
        })

        # Add fiber parameters
        self.add_argument('--fiber_length', type=float, nargs='+', help='Fiber length (m)')
        self.add_argument('--fiber_diameter', type=float, nargs='+', help='Fiber diameter (m)')
        self.add_argument('--membrane_thickness', type=float, nargs='+', help='Membrane thickness (m)')
        self.add_argument('--freq', type=float, nargs='+', help='Frequency (Hz)')
        self.add_argument('--amp', type=float, nargs='+', help='Amplitude (Pa)')
        self.add_argument('--charge', type=float, nargs='+', help='Charge density (C/m²)')
        self.add_argument('--tstim', type=float, nargs='+', help='Stimulus duration (s)')
        self.add_argument('--toffset', type=float, nargs='+', help='Stimulus offset time (s)')
        self.add_argument('--PRF', type=float, nargs='+', help='Pulse repetition frequency (Hz)')

    def parse(self):
        """Parse arguments and convert to appropriate types."""
        args = super().parse()
        for key in self.defaults.keys():
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        """Extract simulation inputs from arguments."""
        return [args[k] for k in ['freq', 'amp', 'charge']]
        
        # Initialize unit conversion factors
        self.factors = {k: 1.0 for k in self.defaults.keys()}
        
        # Factors for unit conversion (all 1.0 since we use SI units)
        self.factors = {k: 1.0 for k in self.defaults.keys()}

        # Add fiber parameters
        self.add_argument('--fiber_length', type=float, nargs='+', help='Fiber length (m)')
        self.add_argument('--fiber_diameter', type=float, nargs='+', help='Fiber diameter (m)')
        self.add_argument('--membrane_thickness', type=float, nargs='+', help='Membrane thickness (m)')
        self.add_argument('--freq', type=float, nargs='+', help='Frequency (Hz)')
        self.add_argument('--amp', type=float, nargs='+', help='Amplitude (Pa)')
        self.add_argument('--charge', type=float, nargs='+', help='Charge density (C/m²)')

    def parse(self):
        """Parse arguments and convert to appropriate types."""
        args = super().parse()
        for key in self.defaults.keys():
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        """Extract simulation inputs from arguments."""
        return [args[k] for k in ['freq', 'amp', 'charge']]



    def addHideOutput(self):
        self.add_argument(
            '--hide', default=False, action='store_true', help='Hide output')

    def addTimeRange(self, default=None):
        self.add_argument(
            '--trange', type=float, nargs=2, default=default,
            help='Time lower and upper bounds (ms)')
        self.to_parse['trange'] = self.parseTimeRange

    def addZvar(self, default):
        self.add_argument(
            '-z', '--zvar', type=str, default=default, help='z-variable type')

    def addYscale(self, default='lin'):
        self.add_argument(
            '--yscale', type=str, choices=('lin', 'log'), default=default,
            help='y-scale type ("lin" or "log")')

    def addZscale(self, default='lin'):
        self.add_argument(
            '--zscale', type=str, choices=('lin', 'log'), default=default,
            help='z-scale type ("lin" or "log")')

    def addZbounds(self, default):
        self.add_argument(
            '--zbounds', type=float, nargs=2, default=default,
            help='z-scale lower and upper bounds')

    def addCmap(self, default=None):
        self.add_argument(
            '--cmap', type=str, default=default, help='Colormap name')

    def addCscale(self, default='lin'):
        self.add_argument(
            '--cscale', type=str, default=default, choices=('lin', 'log'),
            help='Color scale ("lin" or "log")')

    def addInputDir(self, dep_key=None):
        self.inputdir_dep_key = dep_key
        self.add_argument(
            '-i', '--inputdir', type=str, help='Input directory')
        self.to_parse['inputdir'] = self.parseInputDir

    def addOutputDir(self, dep_key=None):
        self.outputdir_dep_key = dep_key
        self.add_argument(
            '-o', '--outputdir', type=str, help='Output directory')
        self.to_parse['outputdir'] = self.parseOutputDir

    def addInputFiles(self, dep_key=None):
        self.inputfiles_dep_key = dep_key
        self.add_argument(
            '-i', '--inputfiles', type=str, help='Input files')
        self.to_parse['inputfiles'] = self.parseInputFiles

    def addPatches(self):
        self.add_argument(
            '--patches', type=str, default='one',
            help='Stimulus patching mode ("none", "one", all", or a boolean list)')
        self.to_parse['patches'] = self.parsePatches

    def addThresholdCurve(self):
        self.add_argument(
            '--threshold', default=False, action='store_true', help='Show threshold amplitudes')

    def addNeuron(self):
        self.add_argument(
            '-n', '--neuron', type=str, nargs='+', help='Neuron name (string)')
        self.to_parse['neuron'] = self.parseNeuron

    def parseNeuron(self, args):
        pneurons = []
        for n in args['neuron']:
            if n == 'pas':
                pneuron = getDefaultPassiveNeuron()
            else:
                pneuron = getPointNeuron(n)
            pneurons.append(pneuron)
        return pneurons

    def addInteractive(self):
        self.add_argument(
            '--interactive', default=False, action='store_true', help='Make interactive')

    def addLabels(self):
        self.add_argument(
            '--labels', type=str, nargs='+', default=None, help='Labels')

    def addRelativeTimeBounds(self):
        self.add_argument(
            '--rel_tbounds', type=float, nargs='+', default=None,
            help='Relative time lower and upper bounds')

    def addPretty(self):
        self.add_argument(
            '--pretty', default=False, action='store_true', help='Make figure pretty')

    def addSubset(self, choices):
        self.add_argument(
            '--subset', type=str, nargs='+', default=['all'], choices=choices + ['all'],
            help='Run specific subset(s)')
        self.subset_choices = choices
        self.to_parse['subset'] = self.parseSubset

    def parseSubset(self, args):
        if args['subset'] == ['all']:
            return self.subset_choices
        else:
            return args['subset']

    def parseTimeRange(self, args):
        if args['trange'] is None:
            return None
        return np.array(args['trange']) * 1e-3

    def parsePatches(self, args):
        if args['patches'] not in ('none', 'one', 'all'):
            return eval(args['patches'])
        else:
            return args['patches']

    def parseInputFiles(self, args):
        if self.inputfiles_dep_key is not None and not args[self.inputfiles_dep_key]:
            return None
        elif args['inputfiles'] is None:
            return OpenFilesDialog('pkl')[0]

    def parseDir(self, key, args, title, dep_key=None):
        if dep_key is not None and args[dep_key] is False:
            return None
        try:
            if args[key] is not None:
                return os.path.abspath(args[key])
            else:
                return selectDirDialog(title=title)
        except ValueError:
            raise ValueError(f'No {key} selected')

    def parseInputDir(self, args):
        return self.parseDir(
            'inputdir', args, 'Select input directory', self.inputdir_dep_key)

    def parseOutputDir(self, args):
        if hasattr(self, 'outputdir') and self.outputdir is not None:
            return self.outputdir
        else:
            if args['outputdir'] is not None and args['outputdir'] == 'dump':
                return DEFAULT_OUTPUT_FOLDER
            else:
                return self.parseDir(
                    'outputdir', args, 'Select output directory', self.outputdir_dep_key)

    def parseLogLevel(self, args):
        return logging.DEBUG if args.pop('verbose') else logging.INFO

    def parsePltScheme(self, args):
        if args['plot'] is None or args['plot'] == ['all']:
            return None
        else:
            return {x: [x] for x in args['plot']}

    def parseOverwrite(self, args):
        check_for_output = args.pop('checkout')
        return not check_for_output

    def restrict(self, args, keys):
        if sum([args[x] is not None for x in keys]) > 1:
            raise ValueError(
                f'You must provide only one of the following arguments: {", ".join(keys)}')

    def parse2array(self, args, key, factor=1):
        return np.array(args[key]) * factor

    def parse(self):
        args = vars(super().parse_args())
        for k, v in self.defaults.items():
            if k in args and args[k] is None:
                args[k] = v if isIterable(v) else [v]
        for k, parse_method in self.to_parse.items():
            args[k] = parse_method(args)
        return args

    @staticmethod
    def parsePlot(args, output):
        render_args = {}
        if 'spikes' in args:
            render_args['spikes'] = args['spikes']
        if args['compare']:
            if args['plot'] == ['all']:
                logger.error('Specific variables must be specified for comparative plots')
                return
            for key in ['cmap', 'cscale']:
                if key in args:
                    render_args[key] = args[key]
            for pltvar in args['plot']:
                comp_plot = CompTimeSeries(output, pltvar)
                comp_plot.render(**render_args)
        else:
            scheme_plot = GroupedTimeSeries(output, pltscheme=args['pltscheme'])
            scheme_plot.render(**render_args)

        # phase_plot = PhaseDiagram(output, args['plot'][0])
        # phase_plot.render(
        #     # trange=args['trange'],
        #     # rel_tbounds=args['rel_tbounds'],
        #     labels=args['labels'],
        #     prettify=args['pretty'],
        #     cmap=args['cmap'],
        #     cscale=args['cscale']
        # )

        plt.show()



    def addPRF(self):
        self.add_argument(
            '--PRF', nargs='+', type=float, help='PRF (Hz)')

    def addDC(self):
        self.add_argument(
            '--DC', nargs='+', type=float, help='Duty cycle (%%)')
        self.add_argument(
            '--spanDC', default=False, action='store_true', help='Span DC from 1 to 100%%')
        self.to_parse['DC'] = self.parseDC

    def addBRF(self):
        self.add_argument(
            '--BRF', nargs='+', type=float, help='Burst repetition frequency (Hz)')

    def addNBursts(self):
        self.add_argument(
            '--nbursts', nargs='+', type=int, help='Number of bursts')

    def addTitrate(self):
        self.add_argument(
            '--titrate', default=False, action='store_true', help='Perform titration')

    def parseAmplitude(self, args):
        raise NotImplementedError

    def parseDC(self, args):
        if args.pop('spanDC'):
            return np.arange(1, 101) * self.factors['DC']  # (-)
        else:
            return np.array(args['DC']) * self.factors['DC']  # (-)

    def parse(self, args=None):
        if args is None:
            args = super().parse()
        for key in ['tstim', 'toffset', 'PRF']:
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        keys = ['amp', 'tstim', 'toffset', 'PRF', 'DC']
        if len(args['nbursts']) > 1 or args['nbursts'][0] > 1:
            del keys[2]
            keys += ['BRF', 'nbursts']
        return [args[k] for k in keys]


