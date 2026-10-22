"""G.711 A-law and μ-law, written out by hand so you can see how companding works.

Both codecs map one 16-bit linear sample to one 8-bit code. The trick is that
the 8 bits are not spread evenly: the code is split into
    [sign: 1 bit][segment: 3 bits][mantissa: 4 bits]
The segment is roughly log2 of the amplitude. Each higher segment covers twice
the amplitude range of the one below it, using the same 16 mantissa steps. So
quiet sounds get fine steps and loud sounds get coarse steps, which is the
"logarithmic" part.

Follows the ITU-T G.711 reference algorithm (as in the classic Sun g711.c),
vectorised with numpy. The input is int16 PCM and the output is uint8 codes.
"""

import numpy as np

# Upper bound of each segment, in the reduced-precision domain each law works in.
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_ULAW_BIAS = 0x84   # μ-law adds a bias so that segment 0 starts above zero
_ULAW_CLIP = 8159


def linear_to_alaw(pcm: np.ndarray) -> np.ndarray:
    x = pcm.astype(np.int32) >> 3                # A-law works on 13-bit magnitude
    positive = x >= 0
    mask = np.where(positive, 0xD5, 0x55)        # sign bit plus even-bit inversion (see below)
    x = np.where(positive, x, -x - 1)

    seg = np.searchsorted(_ALAW_SEG_END, x)       # which logarithmic segment, 0..7 (8 = overflow)
    shift = np.where(seg < 2, 1, seg)             # segments 0 and 1 share the same step size
    mantissa = (x >> shift) & 0x0F
    code = np.where(seg >= 8, 0x7F, (np.minimum(seg, 7) << 4) | mantissa)
    # A-law inverts every other bit (XOR 0x55). On old lines, silence would
    # otherwise be long runs of zeros, which made the line lose clock sync.
    return (code ^ mask).astype(np.uint8)


def alaw_to_linear(codes: np.ndarray) -> np.ndarray:
    a = codes.astype(np.int32) ^ 0x55
    seg = (a & 0x70) >> 4
    t = (a & 0x0F) << 4
    t = np.where(seg == 0, t + 8, t + 0x108)      # +8 / +0x108 decodes to the middle of the step
    t = np.where(seg > 1, t << np.maximum(seg - 1, 0), t)
    return np.where(a & 0x80, t, -t).astype(np.int16)


def linear_to_ulaw(pcm: np.ndarray) -> np.ndarray:
    x = pcm.astype(np.int32) >> 2                # μ-law works on 14-bit magnitude
    negative = x < 0
    mask = np.where(negative, 0x7F, 0xFF)
    x = np.minimum(np.abs(x), _ULAW_CLIP) + (_ULAW_BIAS >> 2)

    seg = np.searchsorted(_ULAW_SEG_END, x)
    mantissa = (x >> (np.minimum(seg, 7) + 1)) & 0x0F
    code = np.where(seg >= 8, 0x7F, (np.minimum(seg, 7) << 4) | mantissa)
    return (code ^ mask).astype(np.uint8)


def ulaw_to_linear(codes: np.ndarray) -> np.ndarray:
    u = ~codes.astype(np.int32) & 0xFF            # μ-law codes are stored bit-inverted
    t = ((u & 0x0F) << 3) + _ULAW_BIAS
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _ULAW_BIAS - t, t - _ULAW_BIAS).astype(np.int16)
