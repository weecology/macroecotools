"""Test suite for macroecotools"""

from pandas import Series, DataFrame
from numpy import array, array_equal
from macroecotools import *

comp_data = DataFrame({'site': [1, 1, 2, 3, 3, 3, 4],
                       'year': [1, 2, 1, 1, 1, 2, 2],
                       'genus': ['a', 'a', 'a', 'a', 'a', 'd', 'f'],
                       'species': ['b', 'b', 'b', 'b', 'c', 'e', 'g'],
                       'spid': ['ab', 'ab', 'ab', 'ab', 'ac', 'de', 'fg'],
                       'counts': [1, 2, 5, 5, 4, 3, 10]})

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

def test_richness_in_group_single_spid_single_group():
    """Test richness_in_group with a single species identifier column, one group"""
    richness = DataFrame({'site': [1, 2, 3, 4], 'richness': [1, 1, 3, 1]},
                         columns=['site', 'richness'])
    assert richness.equals(richness_in_group(comp_data, ['site'], ['spid']))

def test_richness_in_group_multiple_spid_single_group():
    """Test richness_in_group with a multiple species id columns, one group"""
    richness = DataFrame({'site': [1, 2, 3, 4], 'richness': [1, 1, 3, 1]},
                         columns=['site', 'richness'])
    assert richness.equals(richness_in_group(comp_data, ['site'], ['genus', 'species']))

def test_richness_in_group_multiple_groups():
    """Test richness_in_group with a multiple groups"""
    richness = DataFrame({'site': [1, 1, 2, 3, 3, 4],
                          'year': [1, 2, 1, 1, 2, 2],
                          'richness': [1, 1, 1, 2, 1, 1]},
                         columns=['site', 'year', 'richness'])
    assert richness.equals(richness_in_group(comp_data, ['site', 'year'], ['spid']))

def test_abundance_in_group_no_abund_col():
    """Test abundance_in_group with no abundance column provided"""
    abundance = DataFrame({'site': [1, 2, 3, 4],
                           'abundance': [2, 1, 3, 1]},
                           columns=['site', 'abundance'])
    assert abundance.equals(abundance_in_group(comp_data, ['site']))

def test_abundance_in_group_abund_col():
    """Test abundance_in_group with a single group and an abundance column"""
    abundance = DataFrame({'site': [1, 2, 3, 4],
                           'abundance': [3, 5, 12, 10]},
                           columns=['site', 'abundance'])
    assert abundance.equals(abundance_in_group(comp_data, ['site'], ['counts']))

def test_abundance_in_group_multi_group_no_abund_col():
    """Test abundance_in_group w/multiple group columns and no abundance column"""
    abundance = DataFrame({'genus': ['a', 'a', 'd', 'f'],
                           'species': ['b', 'c', 'e', 'g'],
                           'abundance': [4, 1, 1, 1]},
                           columns=['genus', 'species', 'abundance'])
    assert abundance.equals(abundance_in_group(comp_data, ['genus', 'species']))

def test_abundance_in_group_multi_group_abund_col():
    """Test abundance_in_group w/multiple group columns and an abundance column"""
    abundance = DataFrame({'genus': ['a', 'a', 'd', 'f'],
                           'species': ['b', 'c', 'e', 'g'],
                           'abundance': [13, 4, 3, 10]},
                           columns=['genus', 'species', 'abundance'])
    assert abundance.equals(abundance_in_group(comp_data, ['genus', 'species'], ['counts']))