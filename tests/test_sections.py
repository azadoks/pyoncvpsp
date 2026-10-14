"""Tests for the blocks located by the stage state machine and the quantities bounded by them."""

import pytest

from pyoncvpsp.io.output import OncvpspTextParser
from pyoncvpsp.io.output._sections import BlockKind, blocks, line_contexts

LINES = [
    "Begin loop to  construct optimized pseudo wave functions",
    "Calculating optimized projector #   1",
    "",
    " for l=   0",
    "   Wellstate for l =  1  n =  3",
    "Calculating first optimized projector for l=   1",
    "Construct Vanderbilt / Kleinmman-Bylander projectors",
    "B matrix, ll=   1, kap=  -2",
    "B matrix Hermiticity error, ll=  1  kap=  1",
    "Pseudoatom total energy   -3.760249",
    "Diagnostic tests using semi-local pseudopotentials",
    "Diagnostic tests using Vanderbilt-Kleinman-Bylander pseudopotentials",
    " Testing for bound ghosts",
    "Test configuration 0",
    "Test configuration 1",
    "DATA FOR PLOTTING",
]


def _spans(kind: str) -> list[tuple[int, int]]:
    return [(start, stop) for _, start, stop in blocks(line_contexts(LINES), kind)]


def test_blocks_end_at_next_block_or_stage():
    # The projector block for l = 0 ends at the wellstate printed before the next header
    assert _spans(BlockKind.PROJECTOR) == [(1, 4), (5, 6)]
    assert _spans(BlockKind.VKB) == [(7, 8), (8, 9)]
    assert _spans(BlockKind.DIAG_SEMILOCAL) == [(10, 11)]
    assert _spans(BlockKind.DIAG_VKB) == [(11, 12)]
    assert _spans(BlockKind.GHOSTS) == [(12, 13)]
    assert _spans(BlockKind.TEST_CONFIG) == [(13, 14), (14, 15)]


def test_block_attributes():
    contexts = line_contexts(LINES)
    projectors = [(b.l, b.iproj) for b, _, _ in blocks(contexts, BlockKind.PROJECTOR)]
    assert projectors == [(0, 1), (1, 1)]
    vkb = [(b.l, b.kappa) for b, _, _ in blocks(contexts, BlockKind.VKB)]
    assert vkb == [(1, -2), (1, 1)]
    configs = [b.config_index for b, _, _ in blocks(contexts, BlockKind.TEST_CONFIG)]
    assert configs == [0, 1]


def test_skipped_test_configuration_is_empty():
    # A configuration without results must not take the rows of the next one
    text = """\
Test configuration 0

   n   l     f        eae           eps        diff
   2   0  2.0000   -0.83173598   -0.83172430    1.17D-05
      PSP excitation error=  4.26D-14

Test configuration 1

run_config: WARNING for fully non-local PS atom, stg. 2
 WARNING no output for configuration 2

Test configuration 2

   n   l     f        eae           eps        diff
   2   0  2.0000   -0.83173598   -0.83172430    1.17D-05
   2   1  2.5000   -0.40324483   -0.40295809    2.87D-04
      PSP excitation error=  5.05D-05

DATA FOR PLOTTING
"""
    parser = OncvpspTextParser(text=text, check=False)
    assert [len(config) for config in parser.test_configurations] == [1, 0, 2]


@pytest.mark.parametrize("name", ["oncvpsp-2.1.1-fr_08_O_r", "oncvpsp-4.0.1-fr_74_W_r"])
def test_relativistic_projector_spin_follows_kappa(parse_output, name):
    # Each channel is optimized for kap = -(l+1) (j = l+1/2) before kap = l, as its VKB blocks are
    parser = parse_output(name)
    projectors = [
        (ell, spin_sign)
        for ell, iproj, spin_sign, _, _ in parser._optimize_line_indices
        if iproj == 1
    ]
    vkb = [(ell, spin_sign) for ell, spin_sign, _, _ in parser._run_vkb_indices]
    assert projectors == vkb
    assert projectors[:3] == [(0, 1), (1, 1), (1, -1)]
