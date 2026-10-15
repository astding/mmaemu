"""Bit comparisons for the vendored probes and their JSON diagnostics."""
from mmaemu.testing import assert_bitwise_equal


def same_bits(actual, expected) -> bool:
    try:
        assert_bitwise_equal(actual, expected)
    except AssertionError:
        return False
    return True
