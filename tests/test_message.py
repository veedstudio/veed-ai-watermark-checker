import numpy as np
import pytest

from veed_videoseal import message


def test_tag_to_bits_has_requested_length():
    assert len(message.tag_to_bits("VEED:test", nbits=96)) == 96


def test_tag_to_bits_values_are_binary():
    bits = message.tag_to_bits("VEED:test", nbits=96)
    assert set(np.unique(bits).tolist()).issubset({0, 1})


def test_tag_to_bits_is_deterministic():
    a = message.tag_to_bits("VEED:AI-generated", nbits=256)
    b = message.tag_to_bits("VEED:AI-generated", nbits=256)
    assert np.array_equal(a, b)


def test_tag_to_bits_differs_for_different_tags():
    a = message.tag_to_bits("tag-one", nbits=128)
    b = message.tag_to_bits("tag-two", nbits=128)
    assert not np.array_equal(a, b)


def test_bit_accuracy_identical_is_one():
    bits = message.tag_to_bits("x", nbits=64)
    assert message.bit_accuracy(bits, bits) == 1.0


def test_bit_accuracy_inverted_is_zero():
    bits = message.tag_to_bits("x", nbits=64)
    assert message.bit_accuracy(1 - bits, bits) == 0.0


def test_bit_accuracy_half_matching():
    expected = np.array([0, 0, 1, 1])
    recovered = np.array([0, 1, 1, 0])
    assert message.bit_accuracy(recovered, expected) == 0.5


def test_bit_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        message.bit_accuracy(np.array([0, 1]), np.array([0, 1, 1]))


def test_classify_watermark_gates_on_bit_accuracy():
    # The VideoSeal detection bit is uninformative for our checkpoint (≈0.5 even when
    # watermarked), so presence is decided by bit-accuracy alone.
    assert message.classify_watermark(0.95, bit_thresh=0.75) is True
    assert message.classify_watermark(0.75, bit_thresh=0.75) is True
    assert message.classify_watermark(0.53, bit_thresh=0.75) is False
