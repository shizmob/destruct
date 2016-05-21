"""
destruct
A struct parsing library.
"""
import collections
import struct
import itertools
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
    _consumed = 0

    def format(self):
        return None

    def parse(self, input):
        fmt = self.format()
        if fmt:
            self._consumed = struct.calcsize(fmt)
            return struct.unpack(fmt, input[:self._consumed])[0]
        else:
            raise NotImplementedError

class Nothing(Type):
    def parse(self, input):
        self._consumed = 0
        return None

ENDIAN_MAP = {
    'little': '<',
    'big': '>',
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

    def __init__(self, n, signed=True, endian='little'):
        self.n = n
        self.signed = signed
        self.endian = endian

    def format(self):
        endian = ENDIAN_MAP[self.endian]
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

    def __init__(self, n=32, endian='little'):
        self.n = n

    def format(self):
        endian = ENDIAN_MAP[self.endian]
        kind = self.SIZE_MAP[self.n]
        return '{e}{k}'.format(e=endian, k=kind)

class Double(Type):
    def __new__(self, *args, **kwargs):
        return Float(*args, n=64, **kwargs)


class Sig(Type):
    def __init__(self, sequence):
        self.sequence = sequence

    def parse(self, input):
        if input[:len(self.sequence)] == self.sequence:
            self._consumed = len(self.sequence)
            return self.sequence
        raise ValueError('{} does not match expected {}!'.format(input[:len(self.sequence)], self.sequence))

class Str(Type):
    def __init__(self, max_length=0, encoding='utf-8'):
        self.max_length = max_length
        self.encoding = encoding

    def parse(self, input):
        if self.max_length:
            n = input.find(b'\x00', 0, self.max_length)
            if n < 0:
                n = self.max_length
        else:
            n = input.find(b'\x00')
            if n < 0:
                n = len(input)
        self._consumed = n
        return bytes(input[:n]).rstrip(b'\x00').decode(self.encoding)


class Pad(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input):
        if len(input) < self.length:
            raise ValueError('Padding too little (expected {}, got {})!'.format(self.length, len(input)))
        self._consumed = self.length
        return None

class Data(Type):
    def __init__(self, length=0):
        self.length = 0

    def parse(self, input):
        if len(input) < self.length:
            raise ValueError('Data length too little (expected {}, got {})!'.format(self.length, len(input)))
        self._consumed = self.length
        return input[:self.length]


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
        for base in bases:
            spec.update(getattr(base, '_spec', {}))

        for key, value in attrs.copy().items():
            if isinstance(value, Type) or value is None:
                spec[key] = value
                del attrs[key]

        attrs['_spec'] = spec
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
        for name, parser in self._spec.items():
            val = parser.parse(input)
            nbytes = parser._consumed
            if self._align:
                nbytes += self._align - (nbytes % self._align)

            if self._union:
                n = max(n, nbytes)
            else:
                n += nbytes
                input = input[nbytes:]

            setattr(self, name, val)
            if hasattr(self, 'on_' + name):
                getattr(self, 'on_' + name)(self._spec)

        self._consumed = n
        return self

class Union(Struct, union=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, union=True, **kwargs)


class Any(Type):
    def __init__(self, children):
        self.children = children

    def parse(self, input):
        exceptions = []
        for child in self.children:
            try:
                val = child.parse(input)
                self._consumed = child._consumed
                return val
            except Exception as e:
                exceptions.append(e)

        messages = []
        for c, e in zip(self.children, exceptions):
            message = str(e)
            if '\n' in message:
                first, _, others = message.partition('\n')
                message = '{}\n{}'.format(first, '\n'.join('  {}'.format(line) for line in others.split('\n')))
            messages.append('- {}: {}: {}'.format(type(c).__name__, type(e).__name__, message))
        raise ValueError('Expected any of the following, nothing matched:\n{}'.format('\n'.join(messages)))


class Arr(Type):
    def __init__(self, child, count=0, max_length=0):
        self.child = child
        self.count = count
        self.max_length = max_length

    def parse(self, input):
        res = []
        if self.count:
            elems = range(self.count)
        else:
            elems = itertools.repeat(0)

        i = n = 0
        while (not self.count or i < self.count) and (not self.max_length or n < self.max_length):
            if not input:
                if self.count:
                    raise ValueError('Not enough elements in array, expected {} and got {}.'.format(self.count, n - 1))
                else:
                    break

            child = to_parser(self.child)
            try:
                v = parse(child, input)
            except:
                if self.max_length:
                    break
            res.append(v)
            input = input[child._consumed:]

            i += 1
            n += child._consumed

        self._consumed = n
        return res


def to_parser(spec):
    if isinstance(spec, Type):
        return spec
    elif issubclass(spec, Type):
        return spec()
    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def parse(spec, input):
    return to_parser(spec).parse(input)
