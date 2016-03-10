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

# The following test values are obtained from R 3.1.0.
# Truncated distributions are derived using the upper or lower bound.
trunc_logser_pmf_table = [[1, 0.1, 10, '0.9491'],
                          [2, 0.3, 5, '0.1262'],
                          [3, 0.9, 20, '0.1074']]
trunc_logser_cdf_table = [[1, 0.1, 10, '0.9491'],
                          [2, 0.3, 5, '0.9677'],
                          [3, 0.9, 20, '0.6839']]
trunc_expon_pdf_table = [[2, 0.1, 1, '0.0905'],
                          [3, 0.5, 0.2, '0.1233'],
                          [4, 0.7, 0.4, '0.05632']]
trunc_expon_cdf_table = [[2, 0.1, 1, '0.0952'],
                          [3, 0.5, 0.2, '0.7534'],
                          [4, 0.7, 0.4, '0.9195']]
trunc_pareto_pdf_table = [[2, 1, 1, '0.25'],
                          [3, 2, 0.2, '0.0030'],
                          [4, 3, 0.7, '0.0040']]
trunc_pareto_cdf_table = [[2, 1, 1, '0.50'],
                          [3, 2, 0.2, '0.9956'],
                          [4, 3, 0.7, '0.9946']]
trunc_geom_with_zeros_pmf_table = [[1, 0.1, 10, '0.1312'],
                          [2, 0.3, 5, '0.1666'],
                          [3, 0.7, 20, '0.0189']]
trunc_geom_with_zeros_cdf_table = [[1, 0.1, 10, '0.2769'],
                          [2, 0.3, 5, '0.7446'],
                          [3, 0.7, 20, '0.9919']]
trunc_geom_pmf_table = [[1, 0.1, 10, '0.1535'],
                          [2, 0.3, 5, '0.2524'],
                          [3, 0.7, 20, '0.063']]
trunc_geom_cdf_table = [[1, 0.1, 10, '0.1535'],
                          [2, 0.3, 5, '0.6130'],
                          [3, 0.7, 20, '0.973']]
nbinom_lower_trunc_pmf_table = [[20, 10, 0.2, '0.0118'],
                          [2, 8, 0.3, '0.0012'],
                          [5, 20, 0.8, '0.1586']]
nbinom_lower_trunc_cdf_table = [[20, 10, 0.2, '0.0611'],
                          [2, 8, 0.3, '0.0015'],
                          [5, 20, 0.8, '0.6122']]
trunc_weibull_pdf_table = [[3, 1, 1, 2, '0.3679'],
                          [1.7, 5, 2, 1, '0.8640'],
                          [2, 3, 4, 0.7, '0.1664']]
trunc_weibull_cdf_table = [[3, 1, 1, 2, '0.6321'],
                          [1.7, 5, 2, 1, '0.3380'],
                          [2, 3, 4, 0.7, '0.1128']]

def test_pln_pmf1():
    """Tests pmf of pln against values from Table 2 in Grundy Biometrika 38:427-434.
    
    The table of test values is structured as log10(exp(mu)), sigma ** 2, p(1). 
    
    """
    data = set([(log(10 ** line[0]), line[1] ** 0.5, line[2]) for line in pln_table2])
    for line in data:
        yield check_dist, pln.pmf, 1, line[2], line[0], line[1], 0

def test_pln_pmf2():
    """Tests cdf of pln against values from Table 1 in Grundy Biometrika 38:427-434.
    
    The table of test values is structured as log10(exp(mu)), sigma ** 2, p(0). 
    
    """
    data = set([(log(10 ** line[0]), line[1] ** 0.5, line[2]) for line in pln_table1])
    for line in data:
        yield check_dist, pln.pmf, 0, line[2], line[0], line[1], 0

def test_trunc_logser_pmf():
    for line in trunc_logser_pmf_table:
        yield check_dist, trunc_logser.pmf, line[0],  float(line[3]), line[1], line[2]
        
def test_trunc_logser_cdf():
    for line in trunc_logser_cdf_table:
        yield check_dist, trunc_logser.cdf, line[0], float(line[3]), line[1], line[2]

def test_trunc_expon_pdf():
    for line in trunc_expon_pdf_table:
        yield check_dist, trunc_expon.pdf, line[0], float(line[3]), line[1], line[2]
        
def test_trunc_expon_cdf():
    for line in trunc_expon_cdf_table:
        yield check_dist, trunc_expon.cdf, line[0], float(line[3]), line[1], line[2]

def test_trunc_pareto_pdf():
    for line in trunc_pareto_pdf_table:
        yield check_dist, trunc_pareto.pdf, line[0], float(line[3]), line[1], line[2]
        
def test_trunc_pareto_cdf():
    for line in trunc_pareto_cdf_table:
        yield check_dist, trunc_pareto.cdf, line[0], float(line[3]), line[1], line[2]

def test_trunc_geom_with_zeros_pmf():
    for line in trunc_geom_with_zeros_pmf_table:
        yield check_dist, trunc_geom_with_zeros.pmf, line[0], float(line[3]), line[1], line[2]
        
def test_trunc_geom_with_zeros_cdf():
    for line in trunc_geom_with_zeros_cdf_table:
        yield check_dist, trunc_geom_with_zeros.cdf, line[0], float(line[3]), line[1], line[2]

def test_trunc_geom_pmf():
    for line in trunc_geom_pmf_table:
        yield check_dist, trunc_geom.pmf, line[0], float(line[3]), line[1], line[2]
        
def test_trunc_geom_cdf():
    for line in trunc_geom_cdf_table:
        yield check_dist, trunc_geom.cdf, line[0], float(line[3]), line[1], line[2]

def test_nbinom_lower_trunc_pmf():
    for line in nbinom_lower_trunc_pmf_table:
        yield check_dist, nbinom_lower_trunc.pmf, line[0], float(line[3]), line[1], line[2]
        
def test_nbinom_lower_trunc_cdf():
    for line in nbinom_lower_trunc_cdf_table:
        yield check_dist, nbinom_lower_trunc.cdf, line[0], float(line[3]), line[1], line[2]

def test_trunc_weibull_pdf():
    for line in trunc_weibull_pdf_table:
        yield check_dist, trunc_weibull.pdf, line[0], float(line[4]), line[1], line[2], line[3]
        
def test_trunc_weibull_cdf():
    for line in trunc_weibull_cdf_table:
        yield check_dist, trunc_weibull.cdf, line[0], float(line[4]), line[1], line[2], line[3]
        
def check_dist(dist, x, p_known, *pars):
    """Check the pmf/pdf or cdf of a distribution in macroeco_distribution.
    
    dist should take the form dist_name.pmf, dist_name.pdf, or dist_name.cdf (e.g., trunc_logser.pmf)
    
    """
    p_val = dist(x, *pars)
    p_rounded = round(p_val, 4)
    assert_almost_equals(p_rounded, float(p_known), places = 4)
    
if __name__ == "__main__":
    nose.run()