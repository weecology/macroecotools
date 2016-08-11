"""Tests for the macroeco_distributions module"""

from __future__ import division
from macroeco_distributions import *
import nose
from nose.tools import assert_almost_equals, assert_equals
from math import log

dist_dic = {'pln': pln, 'trunc_logser': trunc_logser, 'trunc_expon': trunc_expon, 
            'trunc_pareto': trunc_pareto, 'trunc_geom': trunc_geom, 'trunc_nbinom': nbinom_lower_trunc, 
            'trunc_weibull': trunc_weibull, 'trunc_geom_zeros': trunc_geom_with_zeros}
        
def check_dist(dist, x, p_known, pars):
    """Check the pmf/pdf or cdf of a distribution in macroeco_distribution.
    
    dist should take the form dist_name.pmf, dist_name.pdf, or dist_name.cdf (e.g., trunc_logser.pmf)
    
    """
    p_val = dist(x, *pars)
    p_rounded = round(p_val, 4)
    assert_almost_equals(p_rounded, float(p_known), places = 4)

def test_multi_dists():
    dat_dir = 'test_data.csv'
    with open(dat_dir) as f:
        for line in f:
            line = line.strip('\n').split(',')
            dist_name, func = line[:2]
            dist = dist_dic[dist_name]
            x, p = [float(i) for i in line[2:4]]
            pars = [float(i) for i in line[4:]]
            if dist_name == 'pln': 
                pars = [log(10 ** pars[0]), pars[1] ** 0.5, pars[2]]
            if func == 'pmf': dist_func = dist.pmf
            elif func == 'pdf': dist_func = dist.pdf
            elif func == 'cdf': dist_func = dist.cdf
            yield check_dist, dist_func, x, p, pars
        
if __name__ == "__main__":
    nose.run()