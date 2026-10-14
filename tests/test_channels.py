"""Tests for l-channel, j-channel and projector identifiers."""

from fractions import Fraction

import pytest

from pyoncvpsp.channels import ChannelKey, ProjectorKey, j_channels

HALF = Fraction(1, 2)


@pytest.mark.parametrize(
    "kappa, channel",
    [
        (-1, ChannelKey(0, HALF)),
        (-2, ChannelKey(1, 3 * HALF)),
        (1, ChannelKey(1, HALF)),
        (-4, ChannelKey(3, 7 * HALF)),
        (3, ChannelKey(3, 5 * HALF)),
    ],
)
def test_kappa_round_trip(kappa, channel):
    assert ChannelKey.from_kappa(kappa) == channel
    assert channel.kappa == kappa


def test_printed_kappa():
    assert ChannelKey.from_printed_kappa(2, None) == ChannelKey(2)
    assert ChannelKey.from_printed_kappa(2, 0) == ChannelKey(2)
    assert ChannelKey.from_printed_kappa(2, -3) == ChannelKey(2, 5 * HALF)
    with pytest.raises(ValueError):
        ChannelKey.from_printed_kappa(1, -3)


@pytest.mark.parametrize(
    "l, j", [(0, Fraction(-1, 2)), (1, Fraction(5, 2)), (1, Fraction(1)), (-1, None)]
)
def test_invalid_channel(l, j):  # noqa: E741
    with pytest.raises(ValueError):
        ChannelKey(l, j)


def test_plot_marker():
    assert ChannelKey.from_plot_marker(1, "-", relativistic=True) == ChannelKey(
        1, 3 * HALF
    )
    assert ChannelKey.from_plot_marker(1, " ", relativistic=True) == ChannelKey(1, HALF)
    # l = 0 has only j = 1/2, whatever the marker
    assert ChannelKey.from_plot_marker(0, " ", relativistic=True) == ChannelKey(0, HALF)
    assert ChannelKey.from_plot_marker(1, "-", relativistic=False) == ChannelKey(1)


def test_j_channels():
    assert j_channels(2, relativistic=False) == [ChannelKey(2)]
    assert j_channels(0, relativistic=True) == [ChannelKey(0, HALF)]
    assert j_channels(2, relativistic=True) == [
        ChannelKey(2, 5 * HALF),
        ChannelKey(2, 3 * HALF),
    ]


def test_record_round_trip():
    for channel in (ChannelKey(2), ChannelKey(2, 3 * HALF)):
        assert ChannelKey.from_record(channel.as_dict()) == channel
        projector = ProjectorKey(channel, 2)
        assert ProjectorKey.from_record(projector.as_dict()) == projector
    assert ChannelKey(1, 3 * HALF).as_dict() == {"l": 1, "j": 1.5}
    assert ProjectorKey(ChannelKey(1), 2).as_dict() == {"l": 1, "j": None, "iproj": 2}


def test_channel_from_float_j():
    channel = ChannelKey.from_record({"l": 1, "j": 1.5})
    assert channel == ChannelKey(1, Fraction(3, 2))
    assert len({channel, ChannelKey(1, Fraction(3, 2))}) == 1
    assert channel.l_channel == ChannelKey(1)
