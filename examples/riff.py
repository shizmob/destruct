from destruct import *


CHUNKS = []
def chunk(cls):
    CHUNKS.append(cls)
    return cls

METACHUNKS = []
def metachunk(cls):
    METACHUNKS.append(cls)
    return cls


class Chunk(Struct):
    id     = Sig(b'')
    length = UInt(32)

class MetaChunk(Chunk):
    type   = Str(4)
    chunks = Arr(Any(CHUNKS))

    def on_length(self, spec):
        spec.chunks.max_length = self.length - 4


@chunk
class FormatChunk(Chunk):
    id   = Sig(b'fmt ')
    data = Data()

    def on_length(self, spec):
        spec.data.length = self.length

@chunk
class DataChunk(Chunk):
    id   = Sig(b'data')
    data = Data()

    def on_length(self, spec):
        spec.data.length = self.length

@chunk
class UnknownChunk(Chunk):
    id    = Str(4)
    data  = Data()

    def on_length(self, spec):
        spec.data.length = self.length

@metachunk
class RIFFChunk(MetaChunk):
    id = Sig(b'RIFF')

@metachunk
class ListChunk(MetaChunk):
    id = Sig(b'LIST')


RIFFFile = Any(METACHUNKS)
