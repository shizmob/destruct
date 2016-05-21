import os
import sys
from os import path
from destruct import *

class ARDHeader(Struct):
    _1    = Pad(4)
    sig   = Any([Sig(b'ARD0'), Sig(b'QDA0')])
    count = UInt(32)
    _2    = Pad(256 - 4 - 4 - 4)

class ARDIndexEntry(Struct):
    offset = UInt(32)
    size   = UInt(32)
    _      = UInt(32)
    name   = Str(33)

class ARDArchive(Struct):
    header = ARDHeader()
    index  = Arr(ARDIndexEntry)

    def on_header(self, spec):
        spec.index.count = self.header.count

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: {} <archive> <dest>'.format(sys.argv[0]))
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        archive = parse(ARDArchive, f)

        for file in archive.index:
            print('Reading {} (offset = {}, size = {})...'.format(file.name, file.offset, file.size))
            dest = path.join(sys.argv[2], file.name)
            with open(dest, 'wb') as t:
                f.seek(file.offset, os.SEEK_SET)
                t.write(f.read(file.size))
