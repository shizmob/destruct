from destruct import *


CHUNKS = []
def chunk(cls):
    CHUNKS.append(cls())
    return cls


class Chunk(Struct):
    id = Sig(b'')
    size = UInt(32)

class MetaChunk(Chunk):
    type = Str(4)
    chunks = Seq(Any(CHUNKS))

    def on_size(self, spec):
        spec.chunks.limit = self.size - 4


@chunk
class FormatChunk(Chunk):
    id = Sig(b'fmt ')
    data = Data()

    def on_size(self, spec):
        spec.data.amount = self.size

@chunk
class DataChunk(Chunk):
    id = Sig(b'data')
    data = Data()

    def on_size(self, spec):
        spec.data.amount = self.size

@chunk
class RIFFChunk(MetaChunk):
    id = Sig(b'RIFF')

@chunk
class ListChunk(MetaChunk):
    id = Sig(b'LIST')


RIFFFile = Any(CHUNKS)
