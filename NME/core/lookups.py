# Author: Ryo Segawa
# Inspired by work of: https://github.com/tjjlemaire/PySONIC

import os
from sys import getsizeof
import json
import pickle
import re
import numpy as np
from scipy.interpolate import interp1d

from ..utils import isWithin, isIterable, moveItem


class Lookup:
    ''' Multidimensional lookup object allowing to store, project,
        interpolate and retrieve several lookup tables along multiple
        reference input vectors.
    '''

    interp_choices = ('linear', 'quadratic', 'cubic', 'poly1', 'poly2', 'poly3')

    def __init__(self, refs, tables, interp_method='linear', extrapolate=False):
        ''' Constructor.

            :param refs: dictionary of reference one-dimensional input vectors.
            :param tables: dictionary of multi-dimensional lookup tables
            :param interp_method: interpolation method
            :param extrapolate: boolean stating whether tables can be extrapolated outside
                of reference bounds
        '''
        self.refs = refs
        self.tables = tables
        self.interp_method = interp_method
        self.extrapolate = extrapolate
        for k, v in self.items():
            if v.shape != self.dims:
                raise ValueError(
                    f'{k} Table dimensions {v.shape} does not match references {self.dims}')

        # If no dimension, make sure tables contain scalars
        if self.ndims == 0 and isinstance(self.tables[self.outputs[0]], np.ndarray):
            self.tables = {k: v.item(0) for k, v in self.items()}

        # If single input, mark it as sole ref
        if self.ndims == 1:
            self.refkey = self.inputs[0]
            self.ref = self.refs[self.refkey]
            self.refbounds = (self.ref.min(), self.ref.max())

    def __repr__(self):
        ref_str = ', '.join([f'{x[0]}: {x[1]}' for x in zip(self.inputs, self.dims)])
        tables_str = ', '.join(self.outputs)
        return f'{self.__class__.__name__}{self.ndims}D({ref_str})[{tables_str}]'

    def __getitem__(self, key):
        ''' simplified lookup table getter. '''
        return self.tables[key]

    def __delitem__(self, key):
        ''' simplified lookup table suppressor. '''
        del self.tables[key]

    def __setitem__(self, key, value):
        ''' simplified lookup table setter. '''
        self.tables[key] = value

    def __sizeof__(self):
        ''' Return the size of the lookup in bytes. '''
        s = getsizeof(self.refs) + getsizeof(self.tables)
        for k, v in self.refitems():
            s += v.nbytes
        for k, v in self.items():
            s += v.nbytes
        return s

    def keys(self):
        return self.tables.keys()

    def values(self):
        return self.tables.values()

    def items(self):
        return self.tables.items()

    def refitems(self):
        return self.refs.items()

    def pop(self, key):
        x = self.tables[key]
        del self.tables[key]
        return x

    def rename(self, key1, key2):
        self.tables[key2] = self.tables.pop(key1)

    @property
    def dims(self):
        ''' Tuple indicating the size of each input vector. '''
        return tuple([x.size for x in self.refs.values()])

    @property
    def ndims(self):
        ''' Number of dimensions in lookup. '''
        return len(self.refs)

    @property
    def inputs(self):
        ''' Names of reference input vectors. '''
        return list(self.refs.keys())

    @property
    def outputs(self):
        ''' Names of the different output tables. '''
        return list(self.keys())

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, value):
        if value not in self.interp_choices:
            raise ValueError(f'interpolation method must be one of {self.interp_choices}')
        if self.isPolynomialMethod(value) and self.ndims > 1:
            raise ValueError(f'polynomial interpolation only available for 1D lookups')
        self._interp_method = value

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, value):
        if not isinstance(value, bool):
            raise ValueError(f'extrapolate: expected boolean')
        self._extrapolate = value

    @property
    def kwattrs(self):
        return {
            'interp_method': self.interp_method,
            'extrapolate': self.extrapolate}

    def checkAgainst(self, other):
        ''' Check self object against another lookup object for compatibility. '''
        if self.inputs != other.inputs:
            raise ValueError(f'Differing lookups (references names do not match)')
        if self.dims != other.dims:
            raise ValueError(f'Differing lookup dimensions ({self.dims} - {other.dims})')
        for k, v in self.refitems():
            if (other.refs[k] != v).any():
                raise ValueError(f'Differing {k} lookup reference')
        if self.outputs != other.outputs:
            raise ValueError(f'Differing lookups (table names do not match)')

    def operate(self, other, op):
        ''' Generic arithmetic operator. '''
        if isinstance(other, int):
            other = float(other)
        if isinstance(other, self.__class__):
            self.checkAgainst(other)
            tables = {k: getattr(v, op)(other[k]) for k, v in self.items()}
        elif isinstance(other, float):
            tables = {k: getattr(v, op)(other) for k, v in self.items()}
        else:
            raise ValueError(f'Cannot {op} {self.__class__} object with {type(other)} variable')
        return self.__class__(self.refs, tables, **self.kwattrs)

    def __add__(self, other):
        ''' Addition operator. '''
        return self.operate(other, '__add__')

    def __sub__(self, other):
        ''' Subtraction operator. '''
        return self.operate(other, '__sub__')

    def __mul__(self, other):
        ''' Multiplication operator. '''
        return self.operate(other, '__mul__')

    def __truediv__(self, other):
        ''' Division operator. '''
        return self.operate(other, '__truediv__')

    def squeeze(self):
        ''' Return a new lookup object in which all lookup dimensions that only contain
            a single value have been removed '''
        new_tables = {k: v.squeeze() for k, v in self.items()}
        new_refs = {}
        for k, v in self.refitems():
            if v.size > 1:
                new_refs[k] = v
        return self.__class__(new_refs, new_tables, **self.kwattrs)

    def getAxisIndex(self, key):
        ''' Get the axis index of a specific input key. '''
        assert key in self.inputs, f'Unkown input dimension: {key}'
        return self.inputs.index(key)

    def copy(self):
        ''' Return a copy of the current lookup object. '''
        return self.__class__(self.refs, self.tables, **self.kwattrs)

    def checkInterpMethod(self, interp_method):
        if interp_method not in self.interp_choices:
            raise ValueError(f'interpolation method must be one of {self.interp_choices}')

    @staticmethod
    def isPolynomialMethod(method):
        return method.startswith('poly')

    def getInterpolationDegree(self):
        return int(self.interp_method[-1])

    def getInterpolator(self, ref_key, table_key, axis=-1):
        ''' Return 1D interpolator function along a given reference vector for a specific table .'''
        if self.isPolynomialMethod(self.interp_method):
            return np.poly1d(np.polyfit(self.refs[ref_key], self.tables[table_key],
                                        self.getInterpolationDegree()))
        else:
            fill_value = 'extrapolate' if self.kwattrs['extrapolate'] else np.nan
            return interp1d(self.refs[ref_key], self.tables[table_key], axis=axis,
                            kind=self.interp_method, assume_sorted=True, fill_value=fill_value)

    def project(self, key, value):
        ''' Return a new lookup object in which tables are interpolated at one/several
            specific value(s) along a given dimension.

            :param key: input key
            :param value: value(s) to interpolate lookup tables at
            :return: new interpolated lookup object with adapted dimensions
        '''
        # Check if value is 0 or 1-dimensional
        if not isIterable(value):
            delete_input_dim = True
        else:
            delete_input_dim = False
            value = np.asarray(value)

        # Check that value is within the bounds of the reference vector
        if not self.kwattrs['extrapolate']:
            value = isWithin(key, value, (self.refs[key].min(), self.refs[key].max()))

        # Get the axis index of the reference vector
        axis = self.getAxisIndex(key)
        # print(f'interpolating lookup along {key} (axis {axis}) at {value}')

        # Construct new tables dictionary
        if self.refs[key].size == 1:
            # If reference vector has only 1 value, take the mean along corresponding dimension
            new_tables = {k: v.mean(axis=axis) for k, v in self.items()}
        else:
            # Otherwise, interpolate lookup tables appropriate value(s) along the reference vector
            new_tables = {k: self.getInterpolator(key, k, axis=axis)(value) for k in self.keys()}

        # Construct new refs dictionary, deleting
        new_refs = self.refs.copy()
        if delete_input_dim:
            # If interpolation value is a scalar, remove the corresponding input vector
            del new_refs[key]
        else:
            # Otherwise, update the input vector at the interpolation values
            new_refs[key] = value

        # Construct and return a lookup object with the updated refs and tables
        return self.__class__(new_refs, new_tables, **self.kwattrs)

    def projectN(self, projections):
        ''' Project along multiple dimensions simultaneously.

            :param projections: dictionary of input keys and corresponding interpolation value(s)
            :return: new interpolated lookup object with adapted dimensions
        '''
        # Construct a copy of the current lookup object
        lkp = self.copy()

        # Apply successive projections, overwriting the lookup object at each step
        for k, v in projections.items():
            lkp = lkp.project(k, v)

        # Return updated lookup object
        return lkp

    def move(self, key, index):
        ''' Move a specific input to a new index and re-organize lookup object accordingly.

            :param key: input key
            :param index: target index
        '''
        # Get absolute target axis index
        if index == -1:
            index = self.ndims - 1

        # Get reference axis index
        iref = self.getAxisIndex(key)

        # Re-organize all lookup tables, moving the reference axis to the target index
        for k in self.keys():
            self.tables[k] = np.moveaxis(self.tables[k], iref, index)

        # Re-order refs dictionary such that key falls at the appropriate index
        self.refs = {k: self.refs[k] for k in moveItem(list(self.refs.keys()), key, index)}

    def interpVar1D(self, ref_value, var_key):
        ''' Interpolate a specific lookup vector at one/several specific value(s)
            along the reference input vector.

            :param ref_value: specific input value
            :param var_key: output table key
            :return: interpolated value(s)

            .. warning:: This method can only be used for 1 dimensional lookups.
        '''
        assert self.ndims == 1, 'Cannot interpolate multi-dimensional object'
        if isinstance(ref_value, float):
            isWithin(self.inputs[0], ref_value, self.refbounds)
        return np.interp(ref_value, self.ref, self.tables[var_key], left=np.nan, right=np.nan)

    def interpolate1D(self, value):
        ''' Interpolate all lookup vectors variable at one/several specific value(s)
            along the reference input vector.

            :param value: specific input value
            :return: dictionary of output keys: interpolated value(s)

            .. warning:: This method can only be used for 1 dimensional lookups.
        '''
        return {k: self.interpVar1D(value, k) for k in self.outputs}

    def tile(self, ref_name, ref_values):
        ''' Return a new lookup object in which tables are tiled along a new input dimension.

            :param ref_name: input name
            :param ref_values: input vector
            :return: lookup object with additional input vector and tiled tables
        '''
        itiles = range(ref_values.size)
        tables = {k: np.array([v for i in itiles]) for k, v in self.items()}
        refs = {**{ref_name: ref_values}, **self.refs}
        return self.__class__(refs, tables, **self.kwattrs)

    def reduce(self, rfunc, ref_name):
        ''' Reduce lookup by applying a reduction function along a specific reference axis. '''
        iaxis = self.getAxisIndex(ref_name)
        refs = {k: v for k, v in self.refitems() if k != ref_name}
        tables = {k: rfunc(v, axis=iaxis) for k, v in self.items()}
        return self.__class__(refs, tables, **self.kwattrs)

    def toDict(self):
        ''' Translate self object into a dictionary. '''
        return {
            'refs': {k: v.tolist() for k, v in self.refs.items()},
            'tables': {k: v.tolist() for k, v in self.tables.items()},
        }

    @classmethod
    def fromDict(cls, d):
        ''' Construct lookup instance from dictionary. '''
        refs = {k: np.array(v) for k, v in d['refs'].items()}
        tables = {k: np.array(v) for k, v in d['tables'].items()}
        return cls(refs, tables)

    def toJson(self, fpath):
        ''' Save self object to a JSON file. '''
        with open(fpath, 'w') as fh:
            json.dump(self.toDict(), fh)

    @classmethod
    def fromJson(cls, fpath):
        ''' Construct lookup instance from JSON file. '''
        cls.checkForExistence(fpath)
        with open(fpath) as fh:
            d = json.load(fh)
        return cls.fromDict(d)

    def toPickle(self, fpath):
        ''' Save self object to a PKL file. '''
        with open(fpath, 'wb') as fh:
            pickle.dump({'refs': self.refs, 'tables': self.tables}, fh)

    @classmethod
    def fromPickle(cls, fpath):
        ''' Construct lookup instance from PKL file. '''
        cls.checkForExistence(fpath)
        with open(fpath, 'rb') as fh:
            d = pickle.load(fh)
        return cls(d['refs'], d['tables'])

    @staticmethod
    def checkForExistence(fpath):
        ''' Raise an error if filepath does not correspond to an existing file. '''
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'Missing lookup file: "{fpath}"')


class EffectiveVariablesLookup(Lookup):
    ''' Lookup object with added functionality to handle effective variables, namely:
        - a special EffectiveVariablesDict wrapper around the output tables
        - projectOff and projectDC methods allowing for smart projections.
    '''

    def __init__(self, refs, tables, **kwargs):
        if not isinstance(tables, EffectiveVariablesDict):
            tables = EffectiveVariablesDict(tables)
        super().__init__(refs, tables, **kwargs)

    def interpolate1D(self, value):
        return EffectiveVariablesDict(super().interpolate1D(value))

    def projectOff(self):
        ''' Project for OFF periods (zero amplitude). '''
        # Interpolate at zero amplitude
        lkp0 = self.project('A', 0.)

        # Move charge axis to end in all tables
        Qaxis = lkp0.getAxisIndex('Q')
        for k, v in lkp0.items():
            lkp0.tables[k] = np.moveaxis(v, Qaxis, -1)

        # Iterate along dimensions and take first value along corresponding axis
        for i in range(lkp0.ndims - 1):
            for k, v in lkp0.items():
                lkp0.tables[k] = v[0]

        # Keep only charge vector in references
        lkp0.refs = {'Q': lkp0.refs['Q']}

        return lkp0

    def projectDC(self, amps=None, DC=1.):
        ''' Project lookups at a given duty cycle.'''
        # Assign default values
        if amps is None:
            amps = self.refs['A']
        elif not isIterable(amps):
            amps = np.array([amps])

        # project lookups at zero and defined amps
        lkp0 = self.project('A', 0.)
        lkps_ON = self.project('A', amps)

        # Retrieve amplitude axis index, and move amplitude to first axis
        A_axis = lkps_ON.getAxisIndex('A')
        lkps_ON.move('A', 0)

        # Tile the zero-amplitude lookup to match the lkps_ON dimensions
        lkps_OFF = lkp0.tile('A', lkps_ON.refs['A'])

        # Compute a DC averaged lookup
        lkp = lkps_ON * DC + lkps_OFF * (1 - DC)

        # Move amplitude back to its original axis
        lkp.move('A', A_axis)

        return lkp


class EffectiveVariablesDict():
    ''' Wrapper around a dictionary object, allowing to return derived
        effetive variables for special keys.
    '''

    # Key patterns
    suffix_pattern = '[A-Za-z0-9_]+'
    xinf_pattern = re.compile(f'^({suffix_pattern})inf$')
    taux_pattern = re.compile(f'^tau({suffix_pattern})$')

    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(self.d.keys()) + ')'

    def items(self):
        return self.d.items()

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def alphax(self, x):
        return self.d[f'alpha{x}']

    def betax(self, x):
        return self.d[f'beta{x}']

    def taux(self, x):
        return 1 / (self.alphax(x) + self.betax(x))

    def xinf(self, x):
        return self.alphax(x) * self.taux(x)

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            m = self.taux_pattern.match(key)
            if m is not None:
                return self.taux(m.group(1))
            else:
                m = self.xinf_pattern.match(key)
                if m is not None:
                    return self.xinf(m.group(1))
                else:
                    raise KeyError(key)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def pop(self, key):
        return self.d.pop(key)
