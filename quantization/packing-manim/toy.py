"""The notebook's algorithm (L5_packing_2bit_weights.ipynb) + a trace of every step.

Plain Python ints stand in for torch.uint8: every value stays in 0..255, so the
results are bit-for-bit the same. The lessons only draw what these traces record.
"""
from dataclasses import dataclass

EX1 = [1, 0, 3, 2]                  # → [177]          (1011 0001)
EX2 = [1, 0, 3, 2, 3, 3, 3, 3]      # → [177, 255]
BAD = [1, 0, 3, 2, 1]               # 5 values · 2 bits = 10 bits: not a whole byte


def pack_weights(values, bits):
    """Same code as the notebook, without torch."""
    if len(values) * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple of {8 / bits} - got {len(values)}")
    num_values = len(values) * bits // 8
    num_steps = 8 // bits
    unpacked_idx = 0
    packed = [0] * num_values
    for i in range(num_values):
        for j in range(num_steps):
            packed[i] |= values[unpacked_idx] << (bits * j)
            unpacked_idx += 1
    return packed


def unpack_before_mask(packed, bits):
    """The course's loop body: unpacked[idx] |= packed[i] >> (bits * j)  (|= into zeros = store)."""
    out = [0] * (len(packed) * 8 // bits)
    idx = 0
    for p in packed:
        for j in range(8 // bits):
            out[idx] |= p >> (bits * j)
            idx += 1
    return out


def unpack_weights(packed, bits):
    """Course order: shift everything in the loop, then ONE `unpacked_tensor &= mask` at the end."""
    mask = 2 ** bits - 1
    return [v & mask for v in unpack_before_mask(packed, bits)]


@dataclass
class PackStep:
    i: int          # which packed byte
    j: int          # which slot inside it
    idx: int        # unpacked_idx (keeps counting across bytes)
    value: int      # uint8tensor[unpacked_idx]
    shift: int      # bits * j
    shifted: int    # value << shift
    before: int     # packed[i] before |=
    after: int      # packed[i] after |=


@dataclass
class UnpackStep:
    i: int
    j: int
    idx: int        # position in the unpacked output
    shift: int
    before: int     # packed[i]
    shifted: int    # packed[i] >> shift   (stored by the loop)
    masked: int     # shifted & mask       (after the final &= mask)


def pack_trace(values, bits=2):
    pack_weights(values, bits)  # same shape check
    steps, packed, idx = [], [0] * (len(values) * bits // 8), 0
    for i in range(len(packed)):
        for j in range(8 // bits):
            v, s = values[idx], bits * j
            before = packed[i]
            packed[i] |= v << s
            steps.append(PackStep(i, j, idx, v, s, v << s, before, packed[i]))
            idx += 1
    return steps


def unpack_trace(packed, bits=2):
    mask, steps, idx = 2 ** bits - 1, [], 0
    for i, p in enumerate(packed):
        for j in range(8 // bits):
            s = bits * j
            steps.append(UnpackStep(i, j, idx, s, p, p >> s, (p >> s) & mask))
            idx += 1
    return steps


def bits8(n):
    """[b7, b6, …, b0] — MSB first, the way the byte is drawn."""
    return [(n >> k) & 1 for k in range(7, -1, -1)]


def slots(n, bits=2):
    """2-bit slot values, slot 0 = lowest bits (rightmost on screen)."""
    return [(n >> (bits * k)) & (2 ** bits - 1) for k in range(8 // bits)]


def bstr(n):
    """'1011 0001'"""
    b = f"{n:08b}"
    return f"{b[:4]} {b[4:]}"
