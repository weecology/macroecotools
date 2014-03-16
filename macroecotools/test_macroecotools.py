"""Test suite for macroecotools"""

from pandas import Series, DataFrame
from numpy import array, array_equal
from macroecotools import *

def test_combined_spID_multiple_str():
    """Test combining multiple strings in a single species identifiers"""
    assert combined_spID("fam", "gen", "sp") == "famgensp"

def test_combined_spID_str():
    """Test that a single string remains unchanged as a species identifier"""
    assert combined_spID("Dipodomys spectabilis") == "Dipodomys spectabilis"

def test_combined_spID_single_series():
    """Test that a single column data frame/series returns the same"""
    ids = Series(['gen1 sp1', 'gen1 sp2', 'gen2 sp3'])
    assert ids.equals(combined_spID(ids))

def test_combined_spID_multiple_series():
    """Test that multiple series return the right combined series"""
    ids1 = Series(['gen1', 'gen1', 'gen2', 'gen2'])
    ids2 = Series(['sp1', 'sp1', 'sp2', 'sp3'])
    combined_ids = Series(['gen1sp1', 'gen1sp1', 'gen2sp2', 'gen2sp3'])
    assert combined_ids.equals(combined_spID(ids1, ids2))

def test_combined_spID_lists():
    """Test that multiple lists return the right combined list"""
    ids1 = ['gen1', 'gen1', 'gen2', 'gen2']
    ids2 = ['sp1', 'sp1', 'sp2', 'sp3']
    ids3 = ['subsp1', 'subsp2', 'subsp3', 'subsp4']
    combined_ids = ['gen1sp1subsp1', 'gen1sp1subsp2', 'gen2sp2subsp3', 'gen2sp3subsp4']
    assert combined_ids == combined_spID(ids1, ids2, ids3)

def test_combined_spID_arrays():
    """Test that multiple arrays return the right combined array"""
    ids1 = array(['gen1', 'gen1', 'gen2', 'gen2'])
    ids2 = array(['sp1', 'sp1', 'sp2', 'sp3'])
    combined_ids = array(['gen1sp1', 'gen1sp1', 'gen2sp2', 'gen2sp3'])
    assert array_equal(combined_ids, combined_spID(ids1, ids2))
