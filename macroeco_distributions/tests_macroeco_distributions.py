"""Tests for the macroeco_distributions module"""

from __future__ import division
from macroeco_distributions import *
import nose
from nose.tools import assert_almost_equals, assert_equals
from math import log

dist_dic = {'pln': pln, 'trunc_logser': trunc_logser, 'trunc_expon': trunc_expon, 
            'trunc_pareto': trunc_pareto, 'trunc_geom': trunc_geom, 'trunc_nbinom': nbinom_lower_trunc, 
            'trunc_weibull': trunc_weibull, 'trunc_geom_zeros': trunc_geom_with_zeros}

dist_solver_dic = {'pln': pln_solver, 'logser': logser_solver, 'trunc_logser': trunc_logser_solver, 
                   'trunc_geom': trunc_geom_solver, 'trunc_expon': trunc_expon_solver, 
                   'trunc_pareto': trunc_pareto_solver, 'zipf': zipf_solver,
                   'trunc_nbinom': nbinom_lower_trunc_solver}

# Testing SAD data from BBS (5-year total for one route)
# Parameter values obtained by using the solvers in macroeco_distributions ver. 4e5fc52
sad_1 = [109, 14, 4, 4, 680, 195, 13, 3, 123, 116, 1, 5, 105, 26, 14, 2, 9, 29, 15, 133, 5, 41, 45, 33, 
         17, 27, 37, 11, 169, 1, 27, 7, 19, 23, 100, 4, 8, 5, 19, 1, 21, 12, 6, 1, 10, 2, 1, 94, 2, 4, 28, 1, 3, 
         34, 3, 20, 72, 21, 1, 84, 10, 528, 18, 1, 1, 10, 10, 48, 7]

sad_1_pars_dic = {'pln': (2.27027, 1.83484), 'logser': 0.99621, 'trunc_logser': 0.99621, 
                   'trunc_geom': 0.021218, 'trunc_expon': 0.021678, 
                   'trunc_pareto': 0.38849, 'zipf': 1.32303, 'trunc_nbinom': (0.12298, 0.0054890)}

sad_2 = [71, 27, 88, 21, 1, 2, 6, 2, 54, 3, 4, 33, 21, 12, 36, 29, 11, 14, 3, 3, 1, 10, 1, 28, 2, 119, 235, 
         23, 1, 30, 11, 1, 5, 1, 3, 1, 24, 73, 7, 11, 17, 327, 143, 35, 19, 4, 63, 18, 88, 10, 416, 124, 7, 5, 8, 
         16, 20, 234, 72, 3, 9, 57, 30, 8, 6, 11, 1, 2, 154]

sad_2_pars_dic = {'pln': (2.2937, 1.8070), 'logser': 0.99571, 'trunc_logser': 0.99571, 
                   'trunc_geom': 0.023509, 'trunc_expon': 0.024075, 
                   'trunc_pareto': 0.38769, 'zipf': 1.32247, 'trunc_nbinom': (0.17369, 0.0070226)}

def check_dist(dist, x, p_known, pars):
    """Check the pmf/pdf or cdf of a distribution in macroeco_distribution.
    
    dist should take the form dist_name.pmf, dist_name.pdf, or dist_name.cdf (e.g., trunc_logser.pmf)
    
    """
    p_val = dist(x, *pars)
    p_rounded = round(p_val, 4)
    assert_almost_equals(p_rounded, float(p_known), places = 4)

def check_solver(dist_name, num_list, par_dic):
    """Check that changes on solvers do not break them.
    
    The input dist_name has to be one of those listed in dist_solver_dic.
    
    """
    dist_solver = dist_solver_dic[dist_name]
    par_known = par_dic[dist_name]
    if dist_name in ['trunc_expon', 'trunc_pareto']:
        par = dist_solver(num_list, min(num_list))
    elif dist_name == 'trunc_geom':
        par = dist_solver(num_list, sum(num_list))
    else: par = dist_solver(num_list)
    
    if dist_name in ['pln', 'trunc_nbinom']:
        for i, par_single in enumerate(list(par)):
            assert_almost_equals(par_single, list(par_known)[i], places = 4)
    else: assert_almost_equals(par, par_known, places = 4)
    
def test_multi_dists():
    dat_dir = 'macroeco_distributions/test_data.csv'
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
 
def test_multi_solvers():
    for dist_name in dist_solver_dic.keys():
        yield check_solver, dist_name, sad_1, sad_1_pars_dic
        yield check_solver, dist_name, sad_2, sad_2_pars_dic
         
if __name__ == "__main__":
    nose.run()