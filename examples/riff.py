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
        spec.chunks.max_length = self.length


@chunk
class FormatChunk(Chunk):
    id              = Sig(b'fmt ')
    compression     = UInt(16)
    channel_count   = UInt(16)
    sample_rate     = UInt(32)
    bytes_per_sec   = UInt(32)
    alignment       = UInt(16)
    bits_per_sample = UInt(16)
    padding         = Data()

    def on_length(self, spec):
        spec.padding.length = self.length - (2 + 2 + 4 + 4 + 2 + 2)

@chunk
class FactChunk(Chunk):
    id           = Sig(b'fact')
    sample_count = UInt(32)
    padding      = Data()

    def on_length(self, spec, context):
        spec.padding.length = self.length - 4

class SampleLoop(Struct):
    id       = UInt(32)
    type     = UInt(32)
    start    = UInt(32)
    end      = UInt(32)
    fraction = UInt(32)
    count    = UInt(32)

@chunk
class SampleChunk(Chunk):
    id                = Sig(b'smpl')
    manufacturer      = UInt(32)
    product           = UInt(32)
    period            = UInt(32)
    unity_note        = UInt(32)
    pitch_fraction    = UInt(32)
    smpte_format      = UInt(32)
    smpte_offset      = UInt(32)
    sample_loop_count = UInt(32)
    padding_length    = UInt(32)
    sample_loops      = Arr(SampleLoop)
    padding           = Data()

    def on_sample_loop_count(self, spec, context):
        spec.sample_loops.count = self.sample_loop_count

    def on_padding_length(self, spec, context):
        spec.padding.length = self.padding_length - self.sample_loop_count * sizeof(SampleLoop)

@chunk
class DataChunk(Chunk):
    id   = Sig(b'data')
    data = Data()

    def on_length(self, spec, context):
        spec.data.length = self.length

@chunk
class UnknownChunk(Chunk):
    id    = Str(4)
    data  = Data()

    def on_length(self, spec, context):
        spec.data.length = self.length

@metachunk
class RIFFChunk(MetaChunk):
    id = Sig(b'RIFF')

@metachunk
class ListChunk(MetaChunk):
    id = Sig(b'LIST')


RIFFFile = Any(METACHUNKS)
