"""
destruct
A struct parsing library.
"""
import sys
import os
import io
import collections
import itertools
import struct
import copy


__all__ = [
    # Bases.
    'Type',
    # Special types.
    'Nothing',
    # Numeric types.
    'Int', 'UInt', 'Float', 'Double',
    # Data types.
    'Sig', 'Str', 'Pad', 'Data',
    # Algebraic types.
    'Struct', 'Union',
    # List types.
    'Arr',
    # Choice types.
    'Any',
    # Helper functions.
    'parse'
]


class Type:
    def format(self):
        return None

    def parse(self, input):
        fmt = self.format()
        if fmt:
            length = struct.calcsize(fmt)
            return struct.unpack(fmt, input.read(length))[0]
        else:
            raise NotImplementedError


class Nothing(Type):
    def parse(self, input):
        return None


ORDER_MAP = {
    'le': '<',
    'be': '>',
    'native': '='
}

class Int(Type):
    SIZE_MAP = {
        8: 'b',
        16: 'h',
        32: 'i',
        64: 'l',
        128: 'q'
    }

    def __init__(self, n, signed=True, order='le'):
        self.n = n
        self.signed = signed
        self.order = order

    def format(self):
        endian = ORDER_MAP[self.order]
        kind = self.SIZE_MAP[self.n]
        if not self.signed:
            kind = kind.upper()
        return '{e}{k}'.format(e=endian, k=kind)

class UInt(Type):
    def __new__(self, *args, **kwargs):
        return Int(*args, signed=False, **kwargs)

class Float(Type):
    SIZE_MAP = {
        32: 'f',
        64: 'd'
    }

    def __init__(self, n=32, order='le'):
        self.n = n
        self.order = order

    def format(self):
        endian = ORDER_MAP[self.order]
        kind = self.SIZE_MAP[self.n]
        return '{e}{k}'.format(e=endian, k=kind)

class Double(Type):
    def __new__(self, *args, **kwargs):
        return Float(*args, n=64, **kwargs)


class Sig(Type):
    def __init__(self, sequence):
        self.sequence = sequence

    def parse(self, input):
        data = input.read(len(self.sequence))
        if data != self.sequence:
            raise ValueError('{} does not match expected {}!'.format(data, self.sequence))
        return self.sequence


class Str(Type):
    def __init__(self, length=0, exact=True, encoding='utf-8'):
        self.length = length
        self.exact = exact
        self.encoding = encoding

    def parse(self, input):
        chars = []
        for i in itertools.count(start=1):
            if self.length and i > self.length:
                break
            c = input.read(1)
            if not c or c == b'\x00':
                break
            chars.append(c)

        if self.length and self.exact:
            left = self.length - len(chars)
            if left:
                input.read(left)

        data = b''.join(chars)
        return data.decode(self.encoding)


class Pad(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input):
        data = input.read(self.length)
        if len(data) != self.length:
            raise ValueError('Padding too little (expected {}, got {})!'.format(self.length, len(data)))
        return None

class Data(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input):
        data = input.read(self.length)
        if len(data) != self.length:
            raise ValueError('Data length too little (expected {}, got {})!'.format(self.length, len(data)))
        return data


class MetaSpec(collections.OrderedDict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, item, value):
        if '__' in item:
            super().__setattr__(item, value)
        else:
            self[item] = value

class MetaStruct(type):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        return collections.OrderedDict({'_' + k: v for k, v in kwargs.items()})

    def __new__(cls, name, bases, attrs, **kwargs):
        spec = MetaSpec()
        hooks = {}
        for base in bases:
            spec.update(getattr(base, '_spec', {}))
            hooks.update(getattr(base, '_hooks', {}))

        for key, value in attrs.copy().items():
            if key.startswith('on_'):
                hkey = key.replace('on_', '', 1)
                hooks[hkey] = value
                del attrs[key]
            elif isinstance(value, Type) or value is None:
                spec[key] = value
                del attrs[key]

        attrs['_spec'] = spec
        attrs['_hooks'] = hooks
        return type.__new__(cls, name, bases, attrs)

    def __init__(cls, *args, **kwargs):
        return type.__init__(cls, *args)

class Struct(Type, metaclass=MetaStruct):
    _align = 0
    _union = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec = copy.deepcopy(self._spec)

    def parse(self, input):
        n = 0
        pos = input.tell()

        for name, parser in self._spec.items():
            if self._union:
                input.seek(pos, os.SEEK_SET)

            try:
                val = parser.parse(input)
            except Exception as e:
                traceback = sys.exc_info()[2]
                raise type(e)('{}: {}'.format(name, e)).with_traceback(traceback)
            nbytes = input.tell() - pos

            if self._union:
                n = max(n, nbytes)
            else:
                if self._align:
                    amount = self._align - (nbytes % self._align)
                    input.seek(amount, os.SEEK_CUR)
                    nbytes += amount
                n = nbytes

            setattr(self, name, val)
            if name in self._hooks:
                self._hooks[name](self, self._spec)

        input.seek(pos + n, os.SEEK_SET)
        return self

class Union(Struct, union=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, union=True, **kwargs)


class Any(Type):
    def __init__(self, children):
        self.children = children

    def parse(self, input):
        exceptions = []
        pos = input.tell()
        parsers = [to_parser(c) for c in self.children]

        for child in parsers:
            input.seek(pos, os.SEEK_SET)

            child = to_parser(child)
            try:
                val = parse(child, input)
                return val
            except Exception as e:
                exceptions.append(e)

        messages = []
        for c, e in zip(parsers, exceptions):
            message = str(e)
            if '\n' in message:
                first, _, others = message.partition('\n')
                message = '{}\n{}'.format(first, '\n'.join('  {}'.format(line) for line in others.split('\n')))
            messages.append('- {}: {}: {}'.format(type(c).__name__, type(e).__name__, message))
        raise ValueError('Expected any of the following, nothing matched:\n{}'.format('\n'.join(messages)))


class Arr(Type):
    def __init__(self, child, count=0, max_length=0, pad_count=0, pad_to=0):
        self.child = child
        self.count = count
        self.max_length = max_length
        self.pad_count = pad_count
        self.pad_to = pad_to

    def parse(self, input):
        res = []
        i = n = 0
        pos = input.tell()

        while True:
            if self.count and i >= self.count:
                break
            if self.max_length and input.tell() - pos > self.max_length:
                break

            start = input.tell()
            child = to_parser(self.child)
            try:
                v = parse(child, input)
            except:
                # Check EOF.
                if input.read(1) == b'':
                    break

                input.seek(-1, os.SEEK_CUR)
                raise

            if self.pad_count:
                input.seek(self.pad_count, os.SEEK_CUR)

            if self.pad_to:
                diff = input.tell() - start
                padding = self.pad_to - (diff % self.pad_to)
                input.seek(padding, os.SEEK_CUR)

            if self.max_length and input.tell() - pos > self.max_length:
                break

            res.append(v)
            i += 1

        return res


def to_input(input):
    if not isinstance(input, io.IOBase):
        input = io.BytesIO(input)
    return input

def to_parser(spec):
    if isinstance(spec, Type):
        return spec
    elif issubclass(spec, Type):
        return spec()
    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def parse(spec, input):
    return to_parser(spec).parse(to_input(input))
