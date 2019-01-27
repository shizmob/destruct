"""
destruct
A struct parsing library.
"""
import sys
import os
import io
import types
import collections
import inspect
import itertools
import struct
import copy
import datetime


__all__ = [
    # Bases.
    'Type', 'Context', 'Proxy',
    # Special types.
    'Nothing', 'Static', 'RefPoint', 'Ref', 'Process',
    # Numeric types.
    'Int', 'UInt', 'Float', 'Double', 'Enum',
    # Data types.
    'Sig', 'Str', 'Pad', 'Data',
    # Misc types.
    'DateTime',
    # Algebraic types.
    'Struct', 'Union', 'Tuple', 'Switch',
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


class Context:
    def __init__(self):
        self.parents = []
        self.user = types.SimpleNamespace()


class Type:
    def format(self, input, context):
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

class Process(Type):
    def __init__(self, child=None, parse=None, emit=None):
        self.child = child
        self.do_parse = parse
        self.do_emit = emit

    def parse(self, input, context):
        val = parse(self.child, input, context)
        if self.do_parse:
            val = self.do_parse(val)
        return val

    def emit(self, value, output, context):
        if self.do_emit:
            value = self.do_emit(value)
        return emit(self.child, value, output, context)


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
    type = str

    def __init__(self, length=0, kind='c', exact=True, encoding='utf-8', length_type=UInt(8)):
        self.length = length
        self.kind = kind
        self.exact = exact
        self.encoding = encoding
        self.length_type = length_type

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
                left = length - len(chars) - (kind == 'c' and c == b'\x00')
                if left:
                    input.read(left)

            data = b''.join(chars)
        elif kind == 'pascal':
            outlen = parse(self.length_type, input)
            if length:
                outlen = min(length, outlen)
            if length and exact:
                left = length - outlen
            else:
                left = 0
            data = input.read(outlen)
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
            written = len(value)
            if kind == 'c' and written < length:
                output.write(b'\x00')
                written += 1
            if self.exact and written < length:
                output.write(b'\x00' * (length - written))
        elif kind == 'pascal':
            output.write(chr(length))
            output.write(value)
            

class Pad(Type):
    BLOCK_SIZE = 2048

    def __init__(self, length=0, value=b'\x00', reference=None):
        self.length = length
        self.value = value
        self.reference = reference

    def parse(self, input, context):
        length = to_value(self.length, input, context)
        reference = to_value(self.reference, input, context)

        if self.reference is not None:
            length = length - (input.tell() - reference)

        data = input.read(length)
        if len(data) != length:
            raise ValueError('Padding too little (expected {}, got {})!'.format(length, len(data)))
        return None
    
    def emit(self, value, output, context):
        length = to_value(self.length, output, context)
        value = to_value(self.value, output, context)

        if self.reference is not None:
            length = length - (output.tell() - reference)

        amount, remainder = divmod(length, len(value))
        output.write(value * amount)
        if remainder:
            output.write(value[:remainder])

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


class DateTime(Type):
    def __init__(self, child=None, format=None, timestamp=False):
        self.child = child
        self.format = format
        self.timestamp = timestamp

    def parse(self, input, context):
        val = parse(self.child, input, context)
        if self.timestamp:
            return datetime.datetime.fromtimestamp(val)
        else:
            return datetime.datetime.strptime(val,  self.format)

    def emit(self, value, output, context):
        if self.timestamp:
            val = value.timestamp()
        else:
            val = value.strftime(self.format)
        return emit(self.child, val, output, context)



def proxy_magic_method(name, type=None):
    def inner(self, *args, **kwargs):
        return self._check_magic_method(name, type, *args, **kwargs)
    return inner

class Proxy(Type):
    def __init__(self, type):
        self.__type = type

    def parse(self, input, context):
        return context.parents[-1]

    def emit(self, value, output, context):
        pass

    def __getattr__(self, name):
        return ProxyAttr(to_type(getattr(self.__type, name)), self, name)

    def __getitem__(self, key):
        return ProxyItem(None, self, key)

    def __call__(self, *args, **kwargs):
        if hasattr(self.__type, '__annotations__'):
            type = self.__type.__annotations__.get('return')
        else:
            type = None
        return ProxyCall(type, self, args, kwargs)

    def __deepcopy__(self, memo):
        return self
 
    def _check_magic_method(self, name, type, *args, **kwargs):
        if hasattr(self.__type, name):
            try:
                getattr(self.__type(), name)(*args, **kwargs)
                proxy = self.__getattr__(name).__call__(*args, **kwargs)
                proxy.__type = proxy.__type or type or self.__type
                return proxy
            except:
                raise NotImplemented
        raise AttributeError

MAGIC_ARITH_METHODS = (
    'add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow',
    'lshift', 'rshift', 'and', 'xor', 'or'
)
MAGIC_METHODS = {
    # misc arithmethic
    'neg': None, 'pos': None, 'abs': None, 'invert': None,
    'complex': None, 'int': int, 'float': float, 'round': int, 'trunc': int, 'floor': int, 'ceil': int,
    # attributes, iteration
    'index': int, 'dir': list, 'next': None,
    # context management
    'enter': None, 'exit': None,
    # async
    'await': None, 'aiter': None, 'anext': None, 'aenter': None, 'aexit': None,
    # representations
    #'str', 'repr', 'bytes', 'format', 'hash',
    # comparison
    'lt': bool, 'ge': bool, 'eq': bool, 'ne': bool, 'gt': bool, 'ge': bool,
}
for pfx in ('', 'r', 'i'):
    for x in MAGIC_ARITH_METHODS:
        MAGIC_METHODS[pfx + x] = None
for x, t in MAGIC_METHODS.items():
    x = '__' + x + '__'
    setattr(Proxy, x, proxy_magic_method(x, t))

class ProxyAttr(Proxy):
    def __init__(self, type, parent, name):
        super().__init__(type)
        self.__name = name
        self.__parent = parent

    def parse(self, input, context):
        return getattr(parse(self.__parent, input, context), self.__name)

class ProxyCall(Proxy):
    def __init__(self, type, parent, args, kwargs):
        super().__init__(type)
        self.__parent = parent
        self.__args = args
        self.__kwargs = kwargs
    
    def parse(self, input, context):
        args = [to_value(a, input, context) for a in self.__args]
        kwargs = {k: to_value(v, input, context) for k, v in self.__kwargs.items()}
        return parse(self.__parent, input, context)(*args, **kwargs)

class ProxyItem(Proxy):
    def __init__(self, type, parent, name):
        super().__init__(type)
        self.__parent = parent
        self.__name = name
    
    def parse(self, input, context):
        return parse(self.__parent, input, context)[self.__name]


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
        attrs = MetaSpec({'_' + k: v for k, v in kwargs.items()})
        attrs['self'] = Proxy(attrs)
        return attrs

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
            elif key == 'self' and isinstance(value, Proxy):
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

        context.parents.append(self)

        for name, parser in self._spec.items():
            if parser is None:
                setattr(self, name, None)

        for name, parser in self._spec.items():
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
        
        context.parents.pop()

        input.seek(pos + n, os.SEEK_SET)
        return self
    
    def emit(self, value, output, context):
        n = 0
        pos = output.tell()

        context.parents.append(self)

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
        
        context.parents.pop()
        
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

class Switch(Type):
    def __init__(self, default=None, **kwargs):
        self.options = kwargs
        self.selector = default

    def parse(self, input, context):
        if self.selector is None:
            raise ValueError('Selector not set!')
        if self.selector not in self.options:
            raise ValueError('Selector {} is invalid! [options: {}]'.format(
                self.selector, ', '.join(self.options.keys())
            ))
        return parse(self.options[self.selector], input, context)

    def emit(self, value, output, context):
        if self.selector is None:
            raise ValueError('Selector not set!')
        if self.selector not in self.options:
            raise ValueError('Selector {} is invalid! [options: {}]'.format(
                self.selector, ', '.join(self.options.keys())
            ))
        return emit(self.options[self.selector], value, output, context)


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
    def __init__(self, children):
        self.children = children

    def parse(self, input, context):
        exceptions = []
        pos = input.tell()
        parsers = [to_parser(c, i) for i, c in enumerate(self.children)]

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

    def emit(self, value, output, context):
        exceptions = []
        pos = output.tell()
        parsers = [to_parser(c, i) for i, c in enumerate(self.children)]

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
    def __init__(self, child, count=-1, max_length=-1, stop_value=None, pad_count=0, pad_to=0):
        self.child = child
        self.count = count
        self.max_length = max_length
        self.stop_value = stop_value
        self.pad_count = pad_count
        self.pad_to = pad_to

    def parse(self, input, context):
        res = []
        i = 0
        pos = input.tell()
        
        count = to_value(self.count, input, context)
        max_length = to_value(self.max_length, input, context)
        stop_value = to_value(self.stop_value, input, context)
        pad_count = to_value(self.pad_count, input, context)
        pad_to = to_value(self.pad_to, input, context)

        while count < 0 or i < count:
            if max_length >= 0 and input.tell() - pos >= max_length:
                break
            start = input.tell()
            child = to_parser(self.child, i)

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
            child = to_parser(self.child, i)
            try:
                emit(child, elem, output, context)
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

def to_parser(spec, ident=None):
    if isinstance(spec, (list, tuple)):
        return Tuple(spec)
    elif isinstance(spec, Type):
        return spec
    elif inspect.isclass(spec) and issubclass(spec, Type):
        return spec()
    elif callable(spec):
        return spec(ident)

    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def to_value(p, input, context):
    if isinstance(p, Type):
        return p.parse(input, context)
    return p

def parse(spec, input, context=None):
    return to_parser(spec).parse(to_input(input), context or Context())

def emit(spec, value, output, context=None):
    return to_parser(spec).emit(value, to_input(output), context or Context())
