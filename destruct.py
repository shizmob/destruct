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
    'Nothing', 'Static', 'RefPoint', 'Ref',
    # Numeric types.
    'Int', 'UInt', 'Float', 'Double', 'Enum',
    # Data types.
    'Sig', 'Str', 'Pad', 'Data',
    # Algebraic types.
    'Struct', 'Union', 'Tuple',
    # List types.
    'Arr',
    # Choice types.
    'Maybe', 'Any',
    # Helper functions.
    'parse', 'to_parser', 'emit'
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
        fmt = self.format(input, context)
        if fmt:
            length = struct.calcsize(fmt)
            vals = struct.unpack(fmt, input.read(length))
            if len(vals) == 1:
                return vals[0]
            return vals
        else:
            raise NotImplementedError
    
    def emit(self, value, output, context):
        fmt = self.format(output, context)
        if fmt:
            output.write(struct.pack(fmt, value))
        else:
            raise NotImplementedError


class Nothing(Type):
    def parse(self, input, context):
        return None
    
    def emit(self, value, output, context):
        pass

class Static(Type):
    def __init__(self, value):
        self.value = value

    def parse(self, input, context):
        return to_value(self.value, input, context)

    def emit(self, value, output, context):
        pass

class RefPoint(Type):
    def __init__(self, default=None):
        self.pos = default

    def parse(self, input, context):
        self.pos = input.tell()
        return self.pos

    def emit(self, value, output, context):
        pass

class Ref(Type):
    def __init__(self, child, offset=0, reference=None, reset=True):
        self.child = child
        self.offset = offset
        self.reference = reference
        self.reset = reset

    def parse(self, input, context):
        offset = to_value(self.offset, input, context)
        reference = to_value(self.reference, input, context)

        pos = input.tell()
        if reference is not None:
            input.seek(reference + offset, os.SEEK_SET)
        else:
            input.seek(offset, os.SEEK_CUR)

        try:
            return parse(self.child, input, context)
        finally:
            if to_value(self.reset, input, context):
                input.seek(pos, os.SEEK_SET)

    def emit(self, value, output, context):
        # TODO
        pass


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

    def format(self, input, context):
        endian = ORDER_MAP[to_value(self.order, input, context)]
        kind = self.SIZE_MAP[to_value(self.n, input, context)]
        if not to_value(self.signed, input, context):
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

    def format(self, input, context):
        endian = ORDER_MAP[to_value(self.order, input, context)]
        kind = self.SIZE_MAP[to_value(self.n, input, context)]
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

    def emit(self, value, output, context):
        return to_parser(self.child).emit(value.value, output, context)


class Sig(Type):
    def __init__(self, sequence):
        self.sequence = sequence

    def parse(self, input, context):
        sequence = to_value(self.sequence, input, context)
        data = input.read(len(sequence))
        if data != sequence:
            raise ValueError('{} does not match expected {}!'.format(data, sequence))
        return sequence
    
    def emit(self, value, output, context):
        output.write(to_value(self.sequence, output, context))

class Str(Type):
    def __init__(self, length=0, kind='c', exact=True, encoding='utf-8'):
        self.length = length
        self.kind = kind
        self.exact = exact
        self.encoding = encoding

    def parse(self, input, context):
        length = to_value(self.length, input, context)
        kind = to_value(self.kind, input, context)
        exact = to_value(self.exact, input, context)
        encoding = to_value(self.encoding, input, context)

        if kind in ('raw', 'c'):
            chars = []
            for i in itertools.count(start=1):
                if length and i > length:
                    break
                c = input.read(1)
                if not c or (kind == 'c' and c == b'\x00'):
                    break
                chars.append(c)

            if length and exact:
                left = length - len(chars)
                if left:
                    input.read(left)

            data = b''.join(chars)
        elif kind == 'pascal':
            outlen = input.read(1)[0]
            if length:
                outlen = min(length, outlen)
            if length and exact:
                left = length - outlen
            else:
                left = 0
            data = input.read(length)
            if left:
                input.read(left)
        else:
            raise ValueError('Unknown string kind: {}'.format(kind))
        return data.decode(encoding)

    def emit(self, value, output, context):
        length = to_value(self.length, output, context)
        kind = to_value(self.kind, output, context)
        exact = to_value(self.exact, output, context)
        encoding = to_value(self.encoding, output, context)

        if not length:
            length = len(value) + 1
        value = value[:length].encode(encoding)

        if kind in ('raw', 'c'):
            output.write(value)
            if kind == 'c' and len(value) < length:
                output.write(b'\x00')
        elif kind == 'pascal':
            output.write(chr(length))
            output.write(value)
            

class Pad(Type):
    BLOCK_SIZE = 2048

    def __init__(self, length=0, value='\x00'):
        self.length = length
        self.value = value

    def parse(self, input, context):
        length = to_value(length, input, context)
        data = input.read(length)
        if len(data) != length:
            raise ValueError('Padding too little (expected {}, got {})!'.format(length, len(data)))
        return None
    
    def emit(self, value, output, context):
        length = to_value(length, output, context)
        value = to_value(length, output, context)

        for i in range(0, length, self.BLOCK_SIZE):
            n = min(self.BLOCK_SIZE, length - self.BLOCK_SIZE)
            output.write(value * n)

class Data(Type):
    def __init__(self, length=0):
        self.length = length

    def parse(self, input, context):
        length = to_value(self.length, input, context)
        data = input.read(length)
        if len(data) != length:
            raise ValueError('Data length too little (expected {}, got {})!'.format(length, len(data)))
        return data
    
    def emit(self, value, output, context):
        output.write(value)


class Proxy(Type):
    def __init__(self, parent, path, stack=None):
        self._stack = [] if stack is None else stack
        self._parent = parent
        self._path = path

    def parse(self, input, context):
        obj = self._stack[-1]
        for x in self._path:
            obj = getattr(obj, x)
        return obj

    def emit(self, value, output, context):
        child = self._parent
        for x in self._path:
            child = child._spec[x]
        return child.emit(value, output, context)

    def __getattr__(self, name):
        return Proxy(self._parent, self._path + [name], stack=self._stack)
        
    def __deepcopy__(self, memo):
        return self

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

class MetaAttrs(collections.OrderedDict):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.proxies = []

    def __getitem__(self, item):
        if item in self and not item.startswith('_'):
            proxy = Proxy(self.cls, [item])
            self.proxies.append(proxy)
            return proxy
        return super().__getitem__(item)

class MetaStruct(type):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        return MetaAttrs(cls, {'_' + k: v for k, v in kwargs.items()})

    def __new__(cls, name, bases, attrs, **kwargs):
        spec = MetaSpec()
        hooks = {}
        proxies = attrs.proxies[:]
        attrs = collections.OrderedDict(attrs)

        for base in bases:
            spec.update(getattr(base, '_spec', {}))
            hooks.update(getattr(base, '_hooks', {}))
            proxies.extend(getattr(base, '_proxies', []))

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
        attrs['_proxies'] = proxies
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

        for proxy in self._proxies:
            proxy._stack.append(self)

        for name, parser in self._spec.items():
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

        for proxy in self._proxies:
            proxy._stack.pop()

        input.seek(pos + n, os.SEEK_SET)
        return self
    
    def emit(self, value, output, context):
        n = 0
        pos = output.tell()

        for name, parser in self._spec.items():
            if self._union:
                output.seek(pos, os.SEEK_SET)

            field = getattr(value, name)
            try:
                emit(parser, field, output, context)
            except Exception as e:
                propagate_exception(e, name)
            
            nbytes = output.tell() - pos
            if self._union:
                n = max(n, nbytes)
            else:
                if self._align:
                    amount = self._align - (nbytes % self._align)
                    output.write('\x00' * amount)
                    nbytes += amount
                n = nbytes
        
        output.seek(pos + n, os.SEEK_SET)

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

class Tuple(Type):
    def __init__(self, children):
        self.children = children

    def parse(self, input, context):
        vals = []
        for i, child in enumerate(self.children):
            try:
                vals.append(parse(child, input, context))
            except Exception as e:
                propagate_exception(e, '[index {}]'.format(i))
        return vals

    def emit(self, value, output, context):
        for i, (child, val) in enumerate(zip(self.children, value)):
            try:
                emit(child, val, output, context)
            except Exception as e:
                propagate_exception(e, '[index {}]'.format(i))


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
    
    def emit(self, value, output, context):
        if value is None:
            return
        return emit(self.child, value, output, context)

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

    def emit(self, output, value, context):
        exceptions = []
        pos = output.tell()
        parsers = [to_parser(c, *self.args, **self.kwargs) for c in self.children]

        for child in parsers:
            output.seek(pos, os.SEEK_SET)

            try:
                return emit(child, value, output, context)
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
    def __init__(self, child, count=-1, max_length=-1, stop_value=None, pad_count=0, pad_to=0, spawner=None):
        self.child = child
        self.count = count
        self.max_length = max_length
        self.stop_value = stop_value
        self.pad_count = pad_count
        self.pad_to = pad_to
        self.spawner = spawner

    def parse(self, input, context):
        res = []
        i = 0
        pos = input.tell()
        
        count = to_value(self.count, input, context)
        max_length = to_value(self.count, input, context)
        stop_value = to_value(self.stop_value, input, context)
        pad_count = to_value(self.pad_count, input, context)
        pad_to = to_value(self.pad_to, input, context)

        while count < 0 or i < count:
            if max_length >= 0 and input.tell() - pos >= max_length:
                break
            start = input.tell()

            if self.spawner:
                child = self.spawner(i, self.child)
            else:
                child = self.child
            child = to_parser(child)

            try:
                v = parse(child, input, context)
            except Exception as e:
                # Check EOF.
                if input.read(1) == b'':
                    break
                input.seek(-1, os.SEEK_CUR)
                propagate_exception(e, '[index {}]'.format(i))

            if pad_count:
                input.seek(pad_count, os.SEEK_CUR)

            if pad_to:
                diff = input.tell() - start
                padding = pad_to - (diff % pad_to)
                if padding != pad_to:
                    input.seek(padding, os.SEEK_CUR)

            if v == stop_value or (max_length >= 0 and input.tell() - pos > max_length):
                break

            res.append(v)
            i += 1

        return res
    
    def emit(self, value, output, context):
        stop_value = to_value(self.stop_value, output, context)
        pad_count = to_value(self.pad_count, output, context)
        pad_to = to_value(self.pad_to, output, context)

        if stop_value:
            value = value + [stop_value]

        for i, elem in enumerate(value):
            start = output.tell()

            if self.spawner:
                child = self.spawner(i, self.child)
            else:
                child = self.child

            try:
                emit(child, output, value, context)
            except Exception as e:
                propagate_exception(e, '[index {}]'.format(i))
            
            if pad_count:
                output.write('\x00' * pad_count)
            if pad_to:
                diff = output.tell() - start
                padding = pad_to - (diff % pad_to)
                if padding != pad_to:
                    output.write('\x00' * padding)


def to_input(input):
    if not isinstance(input, io.IOBase):
        input = io.BytesIO(input)
    return input

def to_parser(spec, *args, **kwargs):
    if isinstance(spec, (list, tuple)):
        return Tuple(spec)
    elif isinstance(spec, Type):
        return spec
    elif issubclass(spec, Type):
        return spec(*args, **kwargs)

    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def to_value(p, input, context):
    if isinstance(p, Type):
        return p.parse(input, context)
    return p

def parse(spec, input, context=None):
    return to_parser(spec).parse(to_input(input), context)

def emit(spec, value, output, context=None):
    return to_parser(spec).emit(value, to_input(output), context)
