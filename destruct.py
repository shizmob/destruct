"""
destruct
A struct parsing library.
"""
import os
import io
import types
import collections
import inspect
import itertools
import struct
import copy
import datetime
import errno
from contextlib import contextmanager


__all__ = [
    # Bases.
    'Type', 'Context', 'Proxy',
    # Nil types.
    'Nothing', 'Static',
    # Input management types.
    'RefPoint', 'Ref', 'AlignTo', 'AlignedTo', 'Capped',
    # Post-processing types.
    'Process', 'Map',
    # Misc types.
    'Generic', 'WithFile', 'Lazy',
    # Numeric types.
    'Bool', 'Int', 'UInt', 'Float', 'Double', 'Enum',
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
            values = [indent(',\n'.join('{}: {}'.format(
                format_value(k, formatter),
                format_value(v, formatter)
            ) for k, v in f.items()), 2, True)]
        else:
            fmt = '{{}}'
            values = []
    elif isinstance(f, (list, set, frozenset)):
        l = len(f)
        if l > 3:
            fmt = '{{\n{}\n}}' if isinstance(f, (set, frozenset)) else '[\n{}\n]'
            values = [indent(',\n'.join(format_value(v, formatter) for v in f), 2, True)]
        elif l > 0:
            fmt = '{{{}}}' if isinstance(f, (set, frozenset)) else '[{}]'
            values = [','.join(format_value(v, formatter) for v in f)]
        else:
            fmt = '{{}}' if isinstance(f, (set, frozenset)) else '[]'
            values = []
    elif isinstance(f, (bytes, bytearray)):
        fmt = '{}'
        values = [format_bytes(f)]
    else:
        fmt = '{}'
        values = [formatter(f)]
    return indent(fmt.format(*values), indentation)

def format_bytes(bs):
    return '[' + ' '.join(hex(b)[2:].zfill(2) for b in bs) + ']'

def format_path(path):
    s = ''
    first = True
    for p in path:
        sep = '.'
        if isinstance(p, int):
            p = '[' + str(p) + ']'
            sep = ''
        if sep and not first:
            s += sep
        s += p
        first = False
    return s

def class_name(s: object, module_whitelist=['builtins']):
    module = s.__class__.__module__
    name = s.__class__.__qualname__
    if module in module_whitelist:
        return name
    return module + '.' + name

def friendly_name(s: object) -> str:
    if hasattr(s, '__name__'):
        return s.__name__
    return str(s)

@contextmanager
def seeking(fd, pos, whence=os.SEEK_SET):
    oldpos = fd.tell()
    fd.seek(pos, whence)
    try:
        yield fd
    finally:
        fd.seek(oldpos, os.SEEK_SET)


class Context:
    __slots__ = ('root', 'value', 'path', 'user', 'size')

    def __init__(self, root, value=None):
        self.root = root
        self.value = value
        self.path = []
        self.user = types.SimpleNamespace()
        self.size = None

    @contextmanager
    def enter(self, name, parser):
        self.path.append((name, parser))
        yield
        self.path.pop()

    @contextmanager
    def add_ref(self, size):
        if self.size is None:
            self.size = sizeof(self.root, self.value)
        offset = self.size
        yield offset
        self.size += size

    def format_path(self):
        return format_path(name for name, parser in self.path)


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

    def sizeof(self, value, context):
        return None


class Nothing(Type):
    def parse(self, input, context):
        return None
    
    def emit(self, value, output, context):
        pass

    def sizeof(self, value, context):
        return 0

    def __repr__(self):
        return '<{}>'.format(class_name(self))

class Static(Type):
    def __init__(self, value):
        self.value = value

    def parse(self, input, context):
        return to_value(self.value, input, context)

    def emit(self, value, output, context):
        pass

    def sizeof(self, value, context):
        return 0

    def __repr__(self):
        return '<{}({!r})>'.format(class_name(self), self.value)

class RefPoint(Type):
    def __init__(self, child, reference=None):
        self.child = child
        self.reference = reference

    def parse(self, input, context):
        if self.reference is not None:
            pos = input.tell() + self.reference
        else:
            pos = 0
        pos += parse(self.child, input, context)
        return pos

    def emit(self, value, output, context):
        pos = value
        if self.reference is not None:
            pos -= input.tell() + reference
        return emit(self.child, pos, input, context)

    def sizeof(self, value, context):
        return sizeof(self.child, value, context)

class Ref(Type):
    def __init__(self, child, point=None, reference=os.SEEK_SET):
        self.child = child
        self.point = point
        self.reference = reference

    def parse(self, input, context):
        point = to_value(self.point, input, context)

        with seeking(input, point, self.reference) as f:
            return parse(self.child, f, context)

    def emit(self, value, output, context):
        point = to_value(self.point, output, context)

        with seeking(output, point, self.reference) as f:
            return emit(self.child, value, f, context)

    def sizeof(self, value, context):
        return 0 # sizeof(self.child, value, context)

    def __repr__(self):
        return '<{}: {!r} (point: {}, reference: {})>'.format(class_name(self), self.child, self.point, self.reference)

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

    def sizeof(self, value, context):
        return sizeof(self.child, value, context)

    def __repr__(self):
        return '<{}{}{}>'.format(
            class_name(self),
            ', parse: {}'.format(self.do_parse) if self.do_parse else '',
            ', emit: {}'.format(self.do_emit) if self.do_emit else ''
        )


class Map(Type):
    def __init__(self, child=None, mapping={}):
        self.child = child
        self.mapping = mapping
        self.reverse = {}
        # Do it somewhat awkwardly to support all kinds of iterators
        for k in mapping:
            self.reverse[mapping[k]] = k

    def parse(self, input, context):
        value = parse(self.child, input, context)
        return self.mapping.get(value, value)

    def emit(self, value, output, context):
        value = self.reverse.get(value, value)
        return emit(self.child, value, output, context)

    def sizeof(self, value, context):
        value = self.reverse.get(value, value)
        return sizeof(self.child, value, context)

    def __repr__(self):
        return '<{}: {}>'.format(class_name(self), self.mapping)

class Generic(Type):
    def __init__(self):
        self.child = []

    def resolve(self, v):
        if isinstance(v, Generic):
            self.child.append(v.child[-1])
        else:
            self.child.append(v)

    def pop(self):
        self.child.pop()

    def __parser__(self, ident):
        return to_parser(self.child[-1])

    def parse(self, input, context):
        if not self.child:
            raise ValueError('unresolved generic')
        return parse(self.child[-1], input, context)

    def emit(self, value, output, context):
        if not self.child:
            raise ValueError('unresolved generic')
        return emit(self.child[-1], value, output, context)

    def sizeof(self, value, context):
        return sizeof(self.child[-1], value, context)

    def to_value(self):
        return self.child[-1]

    def __repr__(self):
        if self.child:
            return '<{} @ 0x{:x}: {!r}>'.format(class_name(self), id(self), self.child[-1])
        return '<{} @ 0x{:x}: unresolved>'.format(class_name(self), id(self))

    def __deepcopy__(self, memo):
        return self


class CappedFile:
    def __init__(self, file, max, exact=False):
        self._file = file
        self._pos = 0
        self._max = max
        self._start = file.tell()

    def read(self, n=-1):
        remaining = self._max - self._pos
        if n < 0:
            n = remaining
        n = min(n, remaining)
        self._pos += n
        return self._file.read(n)

    def write(self, data):
        remaining = self._max - self._pos
        if len(data) > remaining:
            raise ValueError('trying to write past limit by {} bytes'.format(len(data) - remaining))
        self._pos += len(data)
        return self._file.write(data)

    def seek(self, offset, whence):
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._start + self._pos + offset
        elif whence == os.SEEK_SET:
            pos = self._start + self._max - offset
        if pos < self._start:
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), offset)
        self._pos = pos - self._start
        return self._file.seek(pos, os.SEEK_SET)

    def __getattr__(self, n):
        return getattr(self._file, n)

class WithFile(Type):
    def __init__(self, child, file):
        self.child = child
        self.file = file

    def parse(self, input, context):
        return parse(self.child, self.file(input), context)

    def emit(self, output, value, context):
        return emit(self.child, self.file(output), value, context)

class LazyEntry:
    __slots__ = ('child', 'input', 'pos', 'context')
    def __init__(self, child, input, context):
        self.child = child
        self.input = input
        self.pos = input.tell()
        self.context = context

    def __call__(self):
        with seeking(self.input, self.pos):
            return parse(self.child, self.input, self.context)

    def __str__(self):
        return '~~{}'.format(self.child)

    def __repr__(self):
        return '<{}: {!r}>'.format(class_name(self), self.child)

class Lazy(Type):
    def __init__(self, child, length=0):
        self.child = child
        self.length = length

    def parse(self, input, context):
        entry = LazyEntry(to_parser(self.child), input, context)
        input.seek(self.length, os.SEEK_CUR)
        return entry

    def __str__(self):
        return '~{}'.format(self.child)

    def __repr__(self):
        return '<{}: {!r}>'.format(class_name(self), self.child)


class Capped(Type):
    def __init__(self, child, limit=None, exact=False):
        self.child = child
        self.limit = limit
        self.exact = exact

    def parse(self, input, context):
        start = input.tell()
        capped = CappedFile(input, self.limit)
        value = parse(self.child, capped, context)
        if self.exact:
            input.seek(start + self.limit, os.SEEK_SET)
        return value

    def emit(self, value, output, context):
        start = output.tell()
        capped = CappedFile(output, self.limit)
        ret = emit(self.child, value, capped, context)
        if self.exact:
            output.seek(start + self.limit, os.SEEK_SET)
        return ret

    def sizeof(self, value, context):
        child = sizeof(self.child, value, context)
        if child is None or self.exact:
            return self.limit
        if self.limit is None:
            return child
        return min(child, self.limit)

    def __repr__(self):
        return '<{}: {!r} (limit={})>'.format(class_name(self), self.child, self.limit)

class AlignTo(Type):
    def __init__(self, child, alignment, value=b'\x00'):
        self.child = child
        self.alignment = alignment
        self.value = value

    def parse(self, input, context):
        res = parse(self.child, input, context)
        adjustment = input.tell() % self.alignment
        if adjustment:
            input.seek(self.alignment - adjustment, os.SEEK_CUR)
        return res

    def emit(self, value, output, context):
        emit(self.child, value, output, context)
        adjustment = output.tell() % self.alignment
        if adjustment:
            output.write(self.value * (self.alignment - adjustment))

    def __repr__(self):
        return '<{}: {!r} (n={})>'.format(class_name(self), self.child, self.alignment)

class AlignedTo(Type):
    def __init__(self, child, alignment, value=b'\x00'):
        self.child = child
        self.alignment = alignment
        self.value = value

    def parse(self, input, context):
        adjustment = input.tell() % self.alignment
        if adjustment:
            input.seek(self.alignment - adjustment, os.SEEK_CUR)
        res = parse(self.child, input, context)
        return res

    def emit(self, value, output, context):
        adjustment = output.tell() % self.alignment
        if adjustment:
            output.write(self.value * (self.alignment - adjustment))
        emit(self.child, value, output, context)

    def __repr__(self):
        return '<{}: {!r} (n={})>'.format(class_name(self), self.child, self.alignment)

ORDER_MAP = {
    'le': 'little',
    'be': 'big',
}

class Int(Type):
    def __init__(self, n, signed=True, order='le'):
        self.n = n
        self.signed = signed
        self.order = order

    def parse(self, input, context):
        n, rem = divmod(self.n, 8)
        if rem != 0:
            raise ValueError('{} can only decode byte-multiple integers, got: {} bits'.format(class_name(self), self.n))
        data = input.read(n)
        if len(data) != n:
            raise ValueError('too little data ({} bytes) to parse {}-bit integer'.format(len(data), self.n))
        return int.from_bytes(data, byteorder=ORDER_MAP[self.order], signed=self.signed)

    def emit(self, value, output, context):
        n, rem = divmod(self.n, 8)
        if rem != 0:
            raise ValueError('{} can only encode byte-multiple integers, got: {} bits'.format(class_name(self), self.n))
        output.write(value.to_bytes(n, ORDER_MAP[self.order], signed=self.signed))

    def sizeof(self, value, context):
        return self.n // 8

    def __repr__(self):
        return '<{}{}, {}, {}>'.format(
            class_name(self),
            self.n,
            'unsigned' if not self.signed else 'signed',
            self.order.upper()
        )

class UInt(Type):
    def __new__(self, *args, **kwargs):
        return Int(*args, signed=False, **kwargs)

class Bool(Type):
    def __init__(self, child=UInt(8), false=0, true=1):
        self.child = child
        self.false = false
        self.true = true

    def parse(self, input, context):
        value = parse(self.child, input, context)
        if value == self.true:
            return True
        elif value == self.false:
            return False
        raise ValueError('value {} is neither true nor false'.format(value))

    def emit(self, output, value, context):
        return emit(self.child, self.true if value else self.false, output, context)

    def sizeof(self, value, context):
        return sizeof(self.child, self.true if value else self.false, context)

    def __repr__(self):
        return '<{}({!r}, false: {}, true: {})>'.format(
            class_name(self),
            self.child,
            self.false,
            self.true
        )

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

    def sizeof(self, value, context):
        return self.n // 8

    def __repr__(self):
        return '<{}{}, {}>'.format(class_name(self), self.n, self.order.upper())

class Double(Type):
    def __new__(self, *args, **kwargs):
        return Float(*args, n=64, **kwargs)


class Enum(Type):
    def __init__(self, enum, child, exhaustive=True):
        self.child = child
        self.enum = enum
        self.exhaustive = exhaustive

    def parse(self, input, context):
        val = parse(self.child, input, context)
        try:
            return to_value(self.enum, input, context)(val)
        except ValueError:
            if self.exhaustive:
                raise
            return val

    def emit(self, value, output, context):
        if isinstance(value, self.enum):
            value = value.value
        return emit(self.child, value, output, context)

    def sizeof(self, value, context):
        if isinstance(value, self.enum):
            value = value.value
        return sizeof(self.child, value, context)

    def __repr__(self):
        return '<{}: {}>'.format(class_name(self), self.enum)

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

    def sizeof(self, value, context):
        return len(self.sequence)

    def __repr__(self):
        return '<{}: {}>'.format(class_name(self), format_bytes(self.sequence))

class Str(Type):
    type = str

    def __init__(self, length=None, kind='c', elem_size=1, exact=True, encoding='utf-8', length_type=UInt(8)):
        self.length = length
        self.kind = kind
        self.exact = exact
        self.encoding = encoding
        self.length_type = length_type
        self.elem_size = elem_size

    def parse(self, input, context):
        length = to_value(self.length, input, context)
        kind = to_value(self.kind, input, context)
        exact = to_value(self.exact, input, context)
        encoding = to_value(self.encoding, input, context)

        if kind in ('raw', 'c'):
            chars = []
            for i in itertools.count(start=1):
                if length is not None and i > length:
                    break
                c = input.read(self.elem_size)
                if not c or (kind == 'c' and c == b'\x00' * self.elem_size):
                    break
                chars.append(c)

            if length is not None and exact:
                left = length - len(chars) - (kind == 'c' and c == b'\x00' * self.elem_size)
                if left:
                    input.read(left * self.elem_size)

            data = b''.join(chars)
        elif kind == 'pascal':
            outlen = parse(self.length_type, input)
            if length is not None:
                outlen = min(length, outlen)
            if length is not None and exact:
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

    def sizeof(self, value, context):
        if self.exact and self.length is not None:
            base = self.length
        elif value is not None:
            base = len(value)
        else:
            return None

        if self.kind == 'pascal':
            size_len = sizeof(self.length_type, base, context)
            if size_len is None:
                return None
        elif self.kind == 'c' and not self.exact:
            size_len = self.elem_size
        else:
            size_len = 0

        return base * self.elem_size + size_len

    def __repr__(self):
        return '<{}({})>'.format(class_name(self), self.kind)


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

    def sizeof(self, value, context):
        if self.reference is not None:
            return None
        return self.length

    def __repr__(self):
        return '<{}: {} * {}>'.format(class_name(self), self.length, format_bytes(self.value))

class Data(Type):
    type = bytes

    def __init__(self, length=0):
        self.length = length

    def parse(self, input, context):
        length = to_value(self.length, input, context)
        if length is None:
            length = -1
        data = input.read(length)
        if length >= 0 and len(data) != length:
            raise ValueError('Data length too little (expected {}, got {})!'.format(length, len(data)))
        return data
    
    def emit(self, value, output, context):
        output.write(value)

    def sizeof(self, value, context):
        if self.length is None or self.length < 0:
            if value is not None:
                return len(value)
            return None
        return self.length

    def __repr__(self):
        return '<{}{}>'.format(class_name(self), ': ' + str(self.length) if self.length is not None else '')

class DateTime(Type):
    def __init__(self, child=None, format=None, timestamp=False, timezone=None):
        self.child = child
        self.format = format
        self.timestamp = timestamp
        self.timezone = timezone

    def parse(self, input, context):
        val = parse(self.child, input, context)
        if self.timestamp:
            return datetime.datetime.fromtimestamp(val, tz=self.timezone)
        else:
            return datetime.datetime.strptime(val,  self.format)

    def emit(self, value, output, context):
        if self.timestamp:
            val = value.timestamp()
        else:
            val = value.strftime(self.format)
        return emit(self.child, val, output, context)

    def sizeof(self, value, context):
        if self.timestamp:
            return sizeof(self.child, value.timestamp(), context)
        else:
            return sizeof(self.child, value.strftime(self.format), context)

    def __repr__(self):
        return '<{}: {}>'.format(class_name(self), 'UNIX timestamp' if self.timestamp else self.format)


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
    def __prepare__(mcls, name, bases, generics=[], **kwargs):
        attrs = MetaSpec({'_' + k: v for k, v in kwargs.items()})
        attrs['self'] = Proxy(attrs)
        attrs['_generics'] = generics
        attrs.update({n: Generic() for n in generics})
        attrs.update({k: globals().get(k) for k in __all__ if k[0].isupper()})
        return attrs

    def __new__(cls, name, bases, attrs, **kwargs):
        spec = MetaSpec()
        hooks = {}
        generics = collections.OrderedDict()

        for base in bases:
            spec.update(getattr(base, '_spec', {}))
            hooks.update(getattr(base, '_hooks', {}))
            generics.update(getattr(base, '_generics', collections.OrderedDict()))

        del attrs['self']
        for k in __all__:
            if k[0].isupper():
                del attrs[k]

        for n in attrs['_generics']:
            g = attrs.pop(n)
            generics[g] = None
        attrs['_generics'] = generics

        for key, value in attrs.copy().items():
            if key.startswith('on_'):
                hkey = key.replace('on_', '', 1)
                hooks[hkey] = value
                del attrs[key]
            elif isinstance(value, Type) or (inspect.isclass(value) and issubclass(value, Type)) or value is None:
                spec[key] = value
                del attrs[key]

        attrs['_spec'] = spec
        attrs['_hooks'] = hooks

        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, *args, **kwargs):
        return super().__init__(*args)

    def __getitem__(cls, ty):
        if not isinstance(ty, tuple):
            ty = (ty,)

        amount = len(ty)
        if amount > len(cls._generics):
            raise TypeError('too many generics arguments for {}: {}'.format(
                cls.__name__, amount
            ))

        new_name = '{}[{}]'.format(cls.__name__, ', '.join(friendly_name(r) for r in ty))
        new = type(new_name, (cls,), cls.__class__.__prepare__(new_name, (cls,)))
        generics = collections.OrderedDict()
        for g, child in zip(cls._generics, ty):
            generics[g] = child
        new._generics = generics
        return new


class Struct(Type, metaclass=MetaStruct):
    _align = 0
    _union = False
    _hide = []
    _partial = False
    _generics = collections.OrderedDict()

    def __init__(self, *args, **kwargs):
        self.__ordered__ = collections.OrderedDict(self.__dict__)
        super().__init__()
        self._spec = copy.deepcopy(self._spec)
        for n in self._spec:
            setattr(self, n, None)
        for n, v in kwargs.items():
            setattr(self, n, v)

    def __setattr__(self, n, v):
        # Store new sets in ordered dict.
        super().__setattr__(n, v)
        self.__ordered__[n] = v

    def parse(self, input, context):
        n = 0
        pos = input.tell()

        for g, child in self._generics.items():
            g.resolve(child)

        for name, parser in self._spec.items():
            if parser is None:
                setattr(self, name, None)

        try:
            for name, parser in self._spec.items():
                with context.enter(name, parser):
                    if parser is None:
                        continue
                    if self._union:
                        input.seek(pos, os.SEEK_SET)

                    val = parse(parser, input, context)

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
        except Exception as e:
            # Check EOF and allow if partial.
            b = input.read(1)
            if not self._partial or b:
                if b:
                    input.seek(-1, os.SEEK_CUR)
                raise
            # allow EOF if partial

        for g in self._generics:
            g.pop()

        input.seek(pos + n, os.SEEK_SET)
        return self
    
    def emit(self, value, output, context):
        n = 0
        pos = output.tell()

        for name, parser in self._spec.items():
            with context.enter(name, parser):
                if self._union:
                    output.seek(pos, os.SEEK_SET)

                field = getattr(value, name)
                emit(parser, field, output, context)

                nbytes = output.tell() - pos
                if self._union:
                    n = max(n, nbytes)
                else:
                    if self._align:
                        amount = self._align - (nbytes % self._align)
                        output.write('\x00' * amount)
                        nbytes += amount
                    n = nbytes

                if name in self._hooks:
                    self._hooks[name](value, self._spec, context)

        for g in self._generics:
            g.child.pop()

        output.seek(pos + n, os.SEEK_SET)

    def sizeof(self, value, context):
        n = 0

        for g, child in self._generics.items():
            g.child.append(child)

        for name, parser in self._spec.items():
            with context.enter(name, parser):
                if value:
                    field = getattr(value, name)
                else:
                    field = None

                nbytes = sizeof(parser, field, context)
                if nbytes is None:
                    n = None
                    break

                if self._union:
                    n = max(n, nbytes)
                else:
                    if self._align:
                        amount = self._align - (nbytes % self._align)
                        nbytes += amount
                    n += nbytes

        return n

    def __iter__(self):
        # Filter out fields we don't want to print: private (_xxx), const (XXX), methods
        return (k for k in self.__ordered__ if not k.startswith('__') and k != '_spec' and not k[0].isupper() and not callable(getattr(self, k)))

    def __eq__(self, other):
        for k in self:
            try:
                ov = getattr(self, k)
                tv = getattr(other, k)
            except AttributeError:
                return False
            if ov != tv:
                return False
        return True

    def __hash__(self):
        return hash(tuple(self))

    def __fmt__(self, fieldfunc):
        # Format our values with fancy colouring according to type.
        args = []
        for k in self:
            if k.startswith('_'):
                continue
            val = getattr(self, k)
            if val in self._hide:
                continue
            val = format_value(val, fieldfunc, 2)
            args.append('  {}: {}'.format(k, val))
        args = ',\n'.join(args)
        # Format final value.
        if args:
            return '{} {{\n{}\n}}'.format(self.__class__.__name__, args)
        else:
            return '{} {{}}'.format(self.__class__.__name__)

    def __str__(self):
        return self.__fmt__(str)

    def __repr__(self):
        return self.__fmt__(repr)

class Union(Struct, union=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, n, v):
        super().__setattr__(n, v)

        val = io.BytesIO()
        try:
            emit(self._spec[n], v, val)
        except:
            return

        for fn, fs in self._spec.items():
            if fn != n:
                val.seek(0)
                try:
                    super().__setattr__(fn, parse(fs, val))
                except:
                    pass

class Tuple(Type):
    def __init__(self, children):
        self.children = children

    def parse(self, input, context):
        vals = []
        for i, child in enumerate(self.children):
            with context.enter(i, child):
                vals.append(parse(child, input, context))
        return vals

    def emit(self, value, output, context):
        for i, (child, val) in enumerate(zip(self.children, value)):
            with context.enter(i, child):
                emit(child, val, output, context)

    def sizeof(self, value, context):
        n = 0
        if value is None:
            value = [None] * len(self.children)

        for i, (child, val) in enumerate(zip(self.children, value)):
            with context.enter(i, child):
                nbytes = sizeof(child, val, context)
                if nbytes is None:
                    n = None
                    break
                n += nbytes

        return n

    def __repr__(self):
        return '<{}({})>'.format(class_name(self), ', '.join(repr(c) for c in self.children))

class Switch(Type):
    def __init__(self, default=None, fallback=None, options=None):
        self.options = options or {}
        self.selector = default
        self.fallback = fallback

    @property
    def current(self):
        if self.selector is None and not self.fallback:
            raise ValueError('Selector not set!')
        if self.selector not in self.options and not self.fallback:
            raise ValueError('Selector {} is invalid! [options: {}]'.format(
                self.selector, ', '.join(repr(x) for x in self.options.keys())
            ))
        return self.options[self.selector] if self.selector is not None and self.selector in self.options else self.fallback

    def parse(self, input, context):
        return parse(self.current, input, context)

    def emit(self, value, output, context):
        return emit(self.current, value, output, context)

    def sizeof(self, value, context):
        return sizeof(self.current, value, context)

    def __repr__(self):
        return '<{}: {}>'.format(class_name(self), ', '.join(repr(k) + ': ' + repr(v) for k, v in self.options.items()))

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

    def sizeof(self, value, context):
        if value is None:
            return 0
        return sizeof(self.child, value, context)

    def __repr__(self):
        return '<{!r}?>'.format(self.child)

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

    def __repr__(self):
        return '<{}[{}]>'.format(class_name(self), ', '.join(repr(c) for c in self.children))

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
            if isinstance(self.child, list):
                child = to_parser(self.child[i], i)
            else:
                child = to_parser(self.child, i)
            with context.enter(i, child):
                try:
                    v = parse(child, input, context)
                except Exception as e:
                    # Check EOF.
                    if input.read(1) == b'':
                        break
                    input.seek(-1, os.SEEK_CUR)
                    raise

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

            with context.enter(i, child):
                emit(child, elem, output, context)
                if pad_count:
                    output.write('\x00' * pad_count)
                if pad_to:
                    diff = output.tell() - start
                    padding = pad_to - (diff % pad_to)
                    if padding != pad_to:
                        output.write('\x00' * padding)

    def sizeof(self, value, context):
        if self.count >= 0:
            length = self.count
        elif value is not None:
            length = len(value) + (1 if self.stop_value else 0)
        else:
            return None

        n = 0
        if value is None:
            value = [None] * length
        for i, v in enumerate(itertools.chain(value, [self.stop_value])):
            if i >= length:
                break

            child = to_parser(self.child, i)
            with context.enter(i, child):
                nbytes = sizeof(self.child, v, context)
                if nbytes is None:
                    n = None
                    break

                if self.pad_count:
                    nbytes += self.pad_count
                if self.pad_to:
                    padding = self.pad_to - (nbytes % self.pad_to)
                    if padding != self.pad_to:
                        nbytes += padding
                n += nbytes

        return n

    def __repr__(self):
        return '<[]{!r}>'.format(self.child)


def to_input(input):
    if isinstance(input, (bytes, bytearray)):
        input = io.BytesIO(input)
    return input

def to_parser(spec, ident=None):
    if isinstance(spec, (list, tuple)):
        return Tuple(spec)
    elif hasattr(spec, '__parser__'):
        return spec.__parser__(ident)
    elif isinstance(spec, Type):
        return spec
    elif inspect.isclass(spec) and issubclass(spec, Type):
        return spec()
    elif callable(spec):
        return spec(ident)

    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def to_value(p, input, context):
    if isinstance(p, Generic):
        return p.to_value()
    if isinstance(p, Type):
        return p.parse(input, context)
    return p


class DestructError(Exception):
    def __init__(self, context, inner):
        super().__init__()
        self.context = context
        self.inner = inner

    def __str__(self):
        return '[{}]: {}: {}'.format(
            self.context.format_path(), class_name(self.inner), str(self.inner)
        )

    def __repr__(self):
        return '{}(context={!r}, inner={!r})'.format(
            class_name(self), self.context, self.inner
        )

class ParseError(DestructError):
    pass

class EmitError(DestructError):
    pass

class SizeError(DestructError):
    pass

def parse(spec, input, context=None):
    parser = to_parser(spec)
    context = context or Context(parser)
    at_start = not context.path
    try:
        return parser.parse(to_input(input), context)
    except Exception as e:
        if at_start:
            raise ParseError(context, e)
        else:
            raise

def emit(spec, value, output=None, context=None):
    parser = to_parser(spec)
    ctx = context or Context(parser, value)
    if output:
        op = to_input(output)
    else:
        op = io.BytesIO()
    try:
        parser.emit(value, op, ctx)
        return op
    except Exception as e:
        if not context:
            raise EmitError(ctx, e)
        else:
            raise

def sizeof(spec, value=None, context=None):
    parser = to_parser(spec)
    ctx = context or Context(parser, value)
    try:
        s = parser.sizeof(value, ctx)
    except Exception as e:
        if not context:
            raise SizeError(ctx, e)
        else:
            raise

    if s is None:
        raise ValueError('size was None')
    return s
