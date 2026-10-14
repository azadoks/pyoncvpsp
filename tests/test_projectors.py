"""Tests of the projector states predicted from the input, one per rule of the Fortran source."""

from fractions import Fraction

import pytest

from pyoncvpsp import OncvpspInput
from pyoncvpsp.channels import ChannelKey, ProjectorKey
from pyoncvpsp.projectors import Energy, Solver, predict_projector_layout

HALF = Fraction(1, 2)


def make_input(
    states: list[tuple[int, int, float]],
    nc: int,
    nproj: list[int],
    lloc: int = 4,
    debl: float = 1.0,
) -> OncvpspInput:
    """Input with the reference configuration `states` = (n, l, f), of which the first `nc` are
    the core, and `nproj` projectors for l = 0, 1, ..."""
    lmax = len(nproj) - 1
    ls = list(range(lmax + 1))
    return OncvpspInput.from_dict(
        {
            "oncvpsp": {
                "atsym": "X",
                "z": 20.0,
                "nc": nc,
                "nv": len(states) - nc,
                "iexc": 4,
                "psfile": "psp8",
            },
            "reference_configuration": {
                "n": [s[0] for s in states],
                "l": [s[1] for s in states],
                "f": [s[2] for s in states],
            },
            "pseudopotentials": {
                "lmax": lmax,
                "l": ls,
                "rc": [2.0] * len(ls),
                "ep": [0.1] * len(ls),
                "ncon": [4] * len(ls),
                "nbas": [8] * len(ls),
                "qcut": [6.0] * len(ls),
            },
            "local_potential": {"lloc": lloc, "lpopt": 5, "rcloc": 2.0, "dvloc0": 0.0},
            "vkb_projectors": {"l": ls, "nproj": nproj, "debl": [debl] * len(ls)},
            "model_core_charge": {"icmod": 0, "fcfact": 0.0},
            "log_derivative_analysis": {"epsh1": -2.0, "epsh2": 2.0, "depsh": 0.02},
            "output_grid": {"rlmax": 6.0, "drl": 0.01},
            "test_configurations": [],
        }
    )


CORE = [(1, 0, 2.0), (2, 0, 2.0), (2, 1, 6.0)]


def summary(layout, l: int, j: Fraction | None = None) -> list[tuple]:  # noqa: E741
    """Slots of a channel as (n, solver, energy, valence_index)."""
    return [
        (s.n, s.solver, s.energy, s.valence_index)
        for s in layout.channel(ChannelKey(l, j))
    ]


V4 = [("ONCVPSP", "4.0.1"), ("METAPSP", "1.0.1")]


@pytest.mark.parametrize("program, version", V4)
def test_v4_every_valence_state_is_a_projector_state(program, version):
    # However small its occupation (ONCVPSP rejects f <= 0)
    inp = make_input(CORE + [(3, 0, 2.0), (4, 0, 0.01)], nc=3, nproj=[2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 0) == [
        (3, Solver.BOUND, Energy.VALENCE, 3),
        (4, Solver.BOUND, Energy.VALENCE, 4),
    ]
    assert not layout.uses_ep(0) and not layout.uses_debl(0)


@pytest.mark.parametrize("program, version", V4)
def test_v4_valence_states_capped_at_nproj(program, version):
    # Issue #2: valence states beyond nproj are not projector states
    inp = make_input(CORE + [(3, 0, 2.0), (4, 0, 1.0), (5, 0, 0.01)], nc=3, nproj=[2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert [s.valence_index for s in layout.channel(ChannelKey(0))] == [3, 4]


@pytest.mark.parametrize("program, version", V4)
def test_v4_wellstates_after_valence_states(program, version):
    inp = make_input(CORE + [(3, 0, 2.0), (3, 1, 1.0)], nc=3, nproj=[3, 2, 2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 0) == [
        (3, Solver.BOUND, Energy.VALENCE, 3),
        (4, Solver.WELLSTATE, Energy.DEBL, None),
        (5, Solver.WELLSTATE, Energy.DEBL, None),
    ]
    # No valence d state: the first wellstate is at ep, one above l + 1 (no d core)
    assert summary(layout, 2) == [
        (3, Solver.WELLSTATE, Energy.EP, None),
        (4, Solver.WELLSTATE, Energy.DEBL, None),
    ]
    assert layout.uses_ep(2) and not layout.uses_ep(0)
    assert layout.uses_debl(0) and layout.uses_debl(0, iproj=2)
    assert not layout.uses_debl(0, iproj=1)


def test_v4_no_valence_state_starts_above_the_core():
    inp = make_input(CORE + [(3, 0, 2.0)], nc=3, nproj=[1, 2])
    layout = predict_projector_layout(inp, "ONCVPSP", "4.0.1", relativistic=False)
    assert [s.n for s in layout.channel(ChannelKey(1))] == [3, 4]


def test_v4_local_channel_has_one_slot():
    inp = make_input(CORE + [(3, 0, 2.0), (3, 1, 2.0)], nc=3, nproj=[2, 2, 0], lloc=2)
    layout = predict_projector_layout(inp, "ONCVPSP", "4.0.1", relativistic=False)
    assert summary(layout, 2) == [(3, Solver.WELLSTATE, Energy.EP, None)]


def test_v4_fully_relativistic_partner_well():
    inp = make_input(CORE + [(3, 0, 2.0), (3, 1, 2.0)], nc=3, nproj=[2, 2])
    layout = predict_projector_layout(inp, "ONCVPSP", "4.0.1", relativistic=True)
    assert [s.projector.channel for s in layout] == [ChannelKey(0, HALF)] * 2 + [
        ChannelKey(1, 3 * HALF)
    ] * 2 + [ChannelKey(1, HALF)] * 2
    assert summary(layout, 1, 3 * HALF)[1][1] == Solver.WELLSTATE
    # j = l - 1/2 is solved in the well of its j = l + 1/2 partner
    assert summary(layout, 1, HALF)[1][1] == Solver.PARTNER_WELL
    assert summary(layout, 0, HALF)[1][1] == Solver.WELLSTATE


V3 = [("ONCVPSP", "3.3.1"), ("ONCVPSP", "2.1.1")]


@pytest.mark.parametrize("program, version", V3)
def test_v3_first_valence_state_then_debl(program, version):
    inp = make_input(CORE + [(3, 0, 2.0), (3, 1, 0.01)], nc=3, nproj=[1, 2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 1) == [
        (3, Solver.BOUND, Energy.VALENCE, 4),
        (4, Solver.BY_ENERGY, Energy.DEBL, None),
    ]
    assert not layout.uses_ep(1) and layout.uses_debl(1)


@pytest.mark.parametrize("program, version", V3)
def test_v3_second_valence_state_overrides_debl(program, version):
    inp = make_input(CORE + [(3, 0, 2.0), (4, 0, 1.0)], nc=3, nproj=[2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 0) == [
        (3, Solver.BOUND, Energy.VALENCE, 3),
        (4, Solver.BOUND, Energy.VALENCE, 4),
    ]
    assert not layout.uses_debl(0)
    # The last further valence state sets the energy, but n is one above the first: the bound
    # state found is not that valence state unless their n agree
    inp = make_input(CORE + [(3, 0, 2.0), (4, 0, 0.01), (5, 0, 1.0)], nc=3, nproj=[2])
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 0)[1] == (4, Solver.BY_ENERGY, Energy.VALENCE, 5)
    assert not layout.uses_debl(0)


@pytest.mark.parametrize("program, version", V3)
def test_v3_local_channel_has_one_slot(program, version):
    inp = make_input(CORE + [(3, 0, 2.0), (3, 1, 2.0)], nc=3, nproj=[2, 2, 2], lloc=2)
    layout = predict_projector_layout(inp, program, version, relativistic=False)
    assert summary(layout, 2) == [(3, Solver.WELLSTATE, Energy.EP, None)]


def test_v3_fully_relativistic_calls_wellstate_per_kappa():
    inp = make_input(CORE + [(3, 0, 2.0)], nc=3, nproj=[1, 2])
    layout = predict_projector_layout(inp, "ONCVPSP", "3.3.1", relativistic=True)
    for j in (3 * HALF, HALF):
        assert summary(layout, 1, j) == [
            (3, Solver.WELLSTATE, Energy.EP, None),
            (4, Solver.BY_ENERGY, Energy.DEBL, None),
        ]


def test_layout_lookup():
    inp = make_input(CORE + [(3, 0, 2.0)], nc=3, nproj=[2])
    layout = predict_projector_layout(inp, "ONCVPSP", "4.0.1", relativistic=False)
    slot = layout.get(ProjectorKey(ChannelKey(0), 1))
    assert slot is not None and slot.is_valence
    assert layout.get(ProjectorKey(ChannelKey(0), 3)) is None


def test_unsupported_version():
    inp = make_input(CORE + [(3, 0, 2.0)], nc=3, nproj=[1])
    with pytest.raises(ValueError):
        predict_projector_layout(inp, "ONCVPSP", "5.0.0", relativistic=False)
