from destruct import *


CHUNKS = []
def chunk(cls):
    CHUNKS.append(cls())
    return cls


class Chunk(Struct):
    id     = Sig(b'')
    length = UInt(32)

class MetaChunk(Chunk):
    type   = Str(4)
    chunks = Arr(Any(CHUNKS))

    def on_size(self, spec):
        spec.chunks.max_length = self.length - 4


@chunk
class FormatChunk(Chunk):
    id   = Sig(b'fmt ')
    data = Data()

    def on_size(self, spec):
        spec.data.length = self.length

@chunk
class DataChunk(Chunk):
    id   = Sig(b'data')
    data = Data()

    def on_size(self, spec):
        spec.data.length = self.length

@chunk
class RIFFChunk(MetaChunk):
    id = Sig(b'RIFF')

@chunk
class ListChunk(MetaChunk):
    id = Sig(b'LIST')


RIFFFile = Any(CHUNKS)
