"""Tests for the macroeco_distributions module"""

from __future__ import division
from macroeco_distributions import *
import nose
from nose.tools import assert_almost_equals, assert_equals
from math import log
from decimal import Decimal

#Test values for Poisson lognomal are chosen from Table 1 and Table 2 
#in Grundy Biometrika 38:427-434.
#In Table 1 the values are deducted from 1 which give p(0).
pln_table1 = [[-2.0, 2, '0.9749'],
              [-2.0, 8, '0.9022'],
              [-2.0, 16, '0.8317'],
              [0.5, 2, '0.1792'],
              [0.5, 8, '0.2908'],
              [0.5, 16, '0.3416'],
              [3, 2, '0.0000'],
              [3, 8, '0.0069'],
              [3, 16, '0.0365']]

pln_table2 = [[-2.0, 2, '0.0234'],
              [-2.0, 8, '0.0538'],
              [-2.0, 16, '0.0593'],
              [0.5, 2, '0.1512'],
              [0.5, 8, '0.1123'],
              [0.5, 16, '0.0879'],
              [3, 2, '0.0000'],
              [3, 8, '0.0065'],
              [3, 16, '0.0193']]

def test_pln_pmf1():
    """Tests pmf of pln against values from Table 2 in Grundy Biometrika 38:427-434.
    
    The table of test values is structured as log10(exp(mu)), sigma ** 2, p(1). 
    
    """
    data = set([(log(10 ** line[0]), line[1] ** 0.5, line[2]) for line in pln_table2])
    for line in data:
        yield check_pln_pmf, 1, line[0], line[1], 0, line[2]

def test_pln_pmf2():
    """Tests cdf of pln against values from Table 1 in Grundy Biometrika 38:427-434.
    
    The table of test values is structured as log10(exp(mu)), sigma ** 2, p(0). 
    
    """
    data = set([(log(10 ** line[0]), line[1] ** 0.5, line[2]) for line in pln_table1])
    for line in data:
        yield check_pln_pmf, 0, line[0], line[1], 0, line[2]

def check_pln_pmf(x, mu, sigma, lower_trunc, p_known):
    p_val = pln.pmf(x, mu, sigma, lower_trunc)
    
    p_rounded = round(p_val, 4)
    assert_almost_equals(p_rounded, float(p_known), places=4)

if __name__ == "__main__":
    nose.run()