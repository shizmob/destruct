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
    'Nothing', 'Static', 'Offset',
    # Numeric types.
    'Int', 'UInt', 'Float', 'Double', 'Enum',
    # Data types.
    'Sig', 'Str', 'Pad', 'Data',
    # Algebraic types.
    'Struct', 'Union',
    # List types.
    'Arr',
    # Choice types.
    'Maybe', 'Any',
    # Helper functions.
    'parse', 'to_parser'
]


def indent(s, count, start=False):
    """ Indent all lines of a string. """
    lines = s.splitlines()
    for i in range(0 if start else 1, len(lines)):
        lines[i] = ' ' * count + lines[i]
    return '\n'.join(lines)

def format_value(f, formatter, indentation=0):
    """ Format containers to use the given formatter function instead of always repr(). """
    if isinstance(f, (dict, collections.Mapping)):
        if f:
            fmt = '{{\n{}\n}}'
            values = [indent(',\n'.join('{}: {}'.format(formatter(k), formatter(v)) for k, v in f.items()), 2, True)]
        else:
            fmt = '{{}}'
            values = []
    elif isinstance(f, (list, set, frozenset)):
        if f:
            fmt = '{{\n{}\n}}' if isinstance(f, (set, frozenset)) else '[\n{}\n]'
            values = [indent(',\n'.join(formatter(v) for v in f), 2, True)]
        else:
            fmt = '{{}}' if isinstance(f, (set, frozenset)) else '[]'
            values = []
    elif isinstance(f, bytes):
        fmt = '[{}]'
        values = [' '.join(hex(b)[2:].zfill(2) for b in f)]
    else:
        fmt = '{}'
        values = [formatter(f)]
    return indent(fmt.format(*values), indentation)

def propagate_exception(e, prefix):
    traceback = sys.exc_info()[2]
    try:
        e = type(e)('{}: {}'.format(prefix, e))
    except:
        e = ValueError('{}: {}: {}'.format(prefix, type(e).__name__, e))
    raise e.with_traceback(traceback) from None


class Type:
    def format(self):
        return None

    def parse(self, input, context):
        fmt = self.format()
        if fmt:
            length = struct.calcsize(fmt)
            vals = struct.unpack(fmt, input.read(length))
            if len(vals) == 1:
                return vals[0]
            return vals
        else:
            raise NotImplementedError


class Nothing(Type):
    def parse(self, input, context):
        return None

class Static(Type):
    def __init__(self, value):
        self.value = value

    def parse(self, input, context):
        return self.value

class Offset(Type):
    def __init__(self, child, offset=0, relative=False, to=0):
        self.offset = offset
        self.child = child
        self.relative = relative
        self.to = to

    def parse(self, input, context):
        if isinstance(self.offset, Type):
            offset = parse(self.offset, input, context)
        else:
            offset = self.offset

        pos = input.tell()
        if self.relative:
            input.seek(self.to + offset, os.SEEK_SET)
        else:
            input.seek(offset, os.SEEK_SET)

        try:
            return parse(self.child, input, context)
        finally:
            input.seek(pos, os.SEEK_SET)


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
        64: 'q'
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

class Enum(Type):
    def __init__(self, enum, child):
        self.child = child
        self.enum = enum

    def parse(self, input, context):
        return self.enum(parse(self.child, input, context))


class Sig(Type):
    def __init__(self, sequence):
        self.sequence = sequence

    def parse(self, input, context):
        data = input.read(len(self.sequence))
        if data != self.sequence:
            raise ValueError('{} does not match expected {}!'.format(data, self.sequence))
        return self.sequence


class Str(Type):
    def __init__(self, length=0, kind='c', exact=True, encoding='utf-8'):
        self.length = length
        self.kind = kind
        self.exact = exact
        self.encoding = encoding

    def parse(self, input, context):
        if self.kind == 'c':
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
        elif self.kind == 'pascal':
            length = input.read(1)[0]
            if self.length:
                length = min(self.length, length)
            if self.length and self.exact:
                left = self.length - length
            else:
                left = 0
            data = input.read(length)
            if left:
                input.read(left)
        else:
            raise ValueError('Unknown string kind: {}'.format(self.kind))
        return data.decode(self.encoding)


class Pad(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input, context):
        data = input.read(self.length)
        if len(data) != self.length:
            raise ValueError('Padding too little (expected {}, got {})!'.format(self.length, len(data)))
        return None

class Data(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input, context):
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
        self.__ordered__ = collections.OrderedDict(self.__dict__)
        super().__init__(*args, **kwargs)
        self._spec = copy.deepcopy(self._spec)

    def __setattr__(self, n, v):
        # Store new sets in ordered dict.
        super().__setattr__(n, v)
        self.__ordered__[n] = v

    def parse(self, input, context):
        n = 0
        pos = input.tell()

        for name, parser in self._spec.items():
            if parser is None:
                setattr(self, name, None)

        for name in self._spec.keys():
            parser = self._spec[name]
            if parser is None:
                continue

            if self._union:
                input.seek(pos, os.SEEK_SET)

            try:
                val = parse(parser, input, context)
            except Exception as e:
                propagate_exception(e, name)
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
                self._hooks[name](self, self._spec, context)

        input.seek(pos + n, os.SEEK_SET)
        return self

    def __str__(self, fieldfunc=str):
        # Filter out fields we don't want to print: private (_xxx), const (XXX), methods
        fields = [k for k in self.__ordered__ if not k.startswith('_') and not k[0].isupper() and not callable(getattr(self, k))]
        # Format their values with fancy colouring according to type.
        args = []
        for k in fields:
            val = getattr(self, k)
            val = format_value(val, fieldfunc, 2)
            args.append('  {}: {}'.format(k, val))
        args = ',\n'.join(args)
        # Format final value.
        if args:
            return '{} {{\n{}\n}}'.format(self.__class__.__name__, args)
        else:
            return '{} {{}}'.format(self.__class__.__name__)

    def __repr__(self):
        return self.__str__(repr)

class Union(Struct, union=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, union=True, **kwargs)


class Maybe(Type):
    def __init__(self, child):
        self.child = child

    def parse(self, input, context):
        pos = input.tell()

        try:
            return parse(self.child, input, context)
        except:
            input.seek(pos, os.SEEK_SET)
            return None

class Any(Type):
    def __init__(self, children, *args, **kwargs):
        self.children = children
        self.args = args
        self.kwargs = kwargs

    def parse(self, input, context):
        exceptions = []
        pos = input.tell()
        parsers = [to_parser(c, *self.args, **self.kwargs) for c in self.children]

        for child in parsers:
            input.seek(pos, os.SEEK_SET)

            try:
                return parse(child, input, context)
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
    def __init__(self, child, count=-1, max_length=-1, stop_value=None, pad_count=0, pad_to=0, *args, **kwargs):
        self.child = child
        self.count = count
        self.max_length = max_length
        self.pad_count = pad_count
        self.pad_to = pad_to
        self.args = args
        self.kwargs = kwargs
        self.stop_value = stop_value

    def parse(self, input, context):
        res = []
        i = 0
        pos = input.tell()

        while self.count < 0 or i < self.count:
            if self.max_length >= 0 and input.tell() - pos >= self.max_length:
                break

            start = input.tell()
            child = to_parser(self.child, *self.args, **self.kwargs)
            try:
                v = parse(child, input, context)
            except Exception as e:
                # Check EOF.
                if input.read(1) == b'':
                    break
                input.seek(-1, os.SEEK_CUR)
                propagate_exception(e, 'index: {}'.format(i))

            if self.pad_count:
                input.seek(self.pad_count, os.SEEK_CUR)

            if self.pad_to:
                diff = input.tell() - start
                padding = self.pad_to - (diff % self.pad_to)
                if padding != self.pad_to:
                    input.seek(padding, os.SEEK_CUR)

            if v == self.stop_value or (self.max_length >= 0 and input.tell() - pos > self.max_length):
                break

            res.append(v)
            i += 1

        return res


def to_input(input):
    if not isinstance(input, io.IOBase):
        input = io.BytesIO(input)
    return input

def to_parser(spec, *args, **kwargs):
    if isinstance(spec, Type):
        return spec
    elif issubclass(spec, Type):
        return spec(*args, **kwargs)
    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def parse(spec, input, context=None):
    return to_parser(spec).parse(to_input(input), context)
