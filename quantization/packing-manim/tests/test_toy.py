"""Every byte state shown in the lecture, checked against the notebook's numbers."""
import pytest

import toy


def test_pack_matches_notebook():
    assert toy.pack_weights(toy.EX1, 2) == [177]
    assert toy.pack_weights(toy.EX2, 2) == [177, 255]


def test_trace_states_match_notebook_comments():
    t = toy.pack_trace(toy.EX1)
    assert [s.after for s in t] == [0b00000001, 0b00000001, 0b00110001, 0b10110001]
    assert [s.shifted for s in t] == [0b1, 0b0, 0b00110000, 0b10000000]
    assert [s.shift for s in t] == [0, 2, 4, 6]


def test_two_byte_trace_carries_idx():
    t = toy.pack_trace(toy.EX2)
    assert [s.idx for s in t] == list(range(8))
    assert [s.i for s in t] == [0] * 4 + [1] * 4
    assert t[4].before == 0 and t[-1].after == 255


def test_shape_check():
    with pytest.raises(ValueError):
        toy.pack_weights(toy.BAD, 2)
    assert toy.pack_weights([0] * 4, 2) == [0]


def test_unpack_roundtrip_and_mask():
    assert toy.unpack_weights([177], 2) == toy.EX1
    assert toy.unpack_weights([177, 255], 2) == toy.EX2
    u = toy.unpack_trace([177])
    assert u[0].shifted == 177 and u[0].masked == 1
    assert [s.shifted for s in u] == [177, 44, 11, 2]
    assert [s.masked for s in u] == toy.EX1


def test_bits_and_slots():
    assert toy.bits8(177) == [1, 0, 1, 1, 0, 0, 0, 1]
    assert toy.slots(177) == [1, 0, 3, 2]


def test_unpack_course_order_masks_once_at_the_end():
    # inside the loop the course only shifts (|= into zeros); one `&= mask` after the loop
    assert toy.unpack_before_mask([177], 2) == [177, 44, 11, 2]
    assert toy.unpack_before_mask([177, 255], 2) == [177, 44, 11, 2, 255, 63, 15, 3]
    assert [v & 3 for v in toy.unpack_before_mask([177, 255], 2)] == toy.EX2


def test_unpack_occupancy_matches_right_shift():
    # Byte.play_shift(RIGHT) maps occupied slots s -> s - k//2 (dropping < 0); a section that
    # rebuilds unpacked[j] after its shift must show the same occupied slots
    for j in range(4):
        after_shift = {s - j for s in range(4) if s - j >= 0}
        assert toy.unpack_occupancy(j) == after_shift
    assert toy.unpack_occupancy(3) == {0}
