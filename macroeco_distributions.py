"""Probability and Likelihood Functions for Distribution Testing"""

from __future__ import division
import sys
from math import factorial, floor
from numpy import exp, histogram, log, matlib, sort, sqrt, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize, special
from scipy.stats import rv_discrete, rv_continuous
from scipy.integrate import quad

class pln_gen(rv_discrete):
    """Poisson lognormal distribution
    
    Method derived from Bulmer 1974 Biometrics 30:101-110    
    
    Bulmer equation 7 - approximation for large abundances
    Bulmer equation 2 - integral for small abundances    
    
    Adapted from Brian McGill's MATLAB function of the same name that was
    originally developed as part of the Palamedes software package by the
    National Center for Ecological Analysis and Synthesis working group on
    Tools and Fresh Approaches for Species Abundance Distributions
    (http://www.nceas.ucsb.edu/projects/11121)
    
    """    
    def _pmf(self, x, mu, sigma, lower_trunc, approx_cut = 10, full_output = 0):
        def untrunc_pmf(x_i, mu, sigma):
            pmf_i = 1e-120
            if sigma > 0:
                if x_i > approx_cut: 
                    #use approximation for large abundances    
                    #Bulmer equation 7
                    #tested vs. integral below - matches to about 6 significant digits for
                    #intermediate abundances (10-50) - integral fails for higher
                    #abundances, approx fails for lower abundances - 
                    #assume it gets better for abundance > 50
                    V = sigma ** 2
                    pmf_i = (1 / sqrt(2 * pi * V) / x_i *
                           exp(-(log(x_i) - mu) ** 2 / (2 * V)) *
                           (1 + 1 / (2 * x_i * V) * ((log(x_i) - mu) ** 2 / V +
                                                    log(x_i) - mu- 1)))
                else:
                    # Bulmer equation 2 -tested against Grundy Biometrika 38:427-434
                    # Table 1 & Table 2 and matched to the 4 decimals in the table except
                    # for very small mu (10^-2)
                    # having the /gamma(ab+1) inside the integrand is inefficient but
                    # avoids pseudo-singularities        
                    # split integral into two so the quad function finds the peak
                    # peak apppears to be just below ab - for very small ab (ab<10)
                    # works better to leave entire peak in one integral and integrate 
                    # the tail in the second integral
                    if x_i < 10:
                        ub = 10
                    else:
                        ub = x_i
                    term1 = ((2 * pi * sigma ** 2) ** -0.5)/ factorial(x_i)
                    #integrate low end where peak is so it finds peak
                    term2a = integrate.quad(lambda x: ((x ** (x_i - 1)) * 
                                                       (exp(-x)) * 
                                                       exp(-(log(x) - mu) ** 2 / 
                                                           (2 * sigma ** 2))), 0,
                                                   ub, full_output = full_output, limit = 100)
                    #integrate higher end for accuracy and in case peak moves
                    term2b = integrate.quad(lambda x: ((x ** (x_i - 1)) * 
                                                       (exp(-x)) * exp(-(log(x) - mu) ** 
                                                                       2/ (2 * sigma ** 
                                                                           2))), ub,
                                                   float('inf'), full_output = full_output, limit = 100)
                    Pr = term1 * term2a[0]
                    Pr_add = term1 * term2b[0]  
                    if Pr + Pr_add > 0: 
                    #likelihood shouldn't really be zero and causes problem taking 
                    #log of it
                        pmf_i = Pr + Pr_add  
            return pmf_i  
        
        pmf = []
        for i, x_i in enumerate(x):
            if lower_trunc[i]: # distribution lowered truncated at 1 (not accouting for zero-abundance species)
                if x_i == 0:
                    pmf_i = 0
                else:
                    pmf_i = untrunc_pmf(x_i, mu[i], sigma[i]) / (1 - untrunc_pmf(0, mu[i], sigma[i]))
            else:
                pmf_i = untrunc_pmf(x_i, mu[i], sigma[i])
            pmf.append(pmf_i)
        return np.array(pmf)
    
    def _cdf(self, x, mu, sigma, lower_trunc, approx_cut = 10, full_output = 0):
        x = np.array(x)
        cdf = []
        for x_i in x:
            cdf.append(sum(self.pmf(range(int(x_i) + 1), mu, sigma, lower_trunc)))
        return np.array(cdf)

    def _argcheck(self, *args):
        return 1
    
pln = pln_gen(name='pln', longname='Poisson lognormal', 
              shapes = 'mu, sigma, lower_trunc')

class trunc_logser_gen(rv_discrete):
    """Upper truncated logseries distribution
    
    Scipy based distribution class for the truncated logseries pmf, cdf and rvs
    
    Usage:
    PMF: trunc_logser.pmf(list_of_xvals, p, upper_bound)
    CDF: trunc_logser.cdf(list_of_xvals, p, upper_bound)
    Random Numbers: trunc_logser.rvs(p, upper_bound, size=1)
    
    """
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        if p < 1:
            return stats.logser.pmf(x, p) / stats.logser.cdf(upper_bound, p)
        else:
            x = np.array(x)
            ivals = np.arange(1, upper_bound + 1)
            normalization = sum(p ** ivals / ivals)
            pmf = (p ** x / x) / normalization
            return pmf

trunc_logser = trunc_logser_gen(a=1, name='trunc_logser',
                                longname='Upper truncated logseries',
                                shapes="upper_bound",
                                extradoc="""Truncated logseries
                                
                                Upper truncated logseries distribution
                                """
                                )

class trunc_expon_gen(rv_continuous):
    """Lower truncated exponential distribution
    
    Scipy based distribution class for the truncated exponential pdf, cdf and rvs
    
    Usage:
    PDF: trunc_exp.pdf(list_of_xvals, lambda, lower_bound)
    CDF: trunc_exp.cdf(list_of_xvals, lambda, lower_bound)
    Random Numbers: trunc_exp.rvs(lambda, lower_bound, size=1)
    
    """
    def _pdf(self, x, lmd, lower_bound):
        x = np.array(x)
        pdf = []
        for i, x_i in enumerate(x):
            if x_i < lower_bound[i]:
                pdf.append(0)
            else:
                pdf.append(lmd[i] * exp(-lmd[i] * (x_i - lower_bound[i])))
        return np.array(pdf)
    
    def _cdf(self, x, lmd, lower_bound):
        x = np.array(x)
        cdf = []
        for i, x_i in enumerate(x):
            cdf.append(1 - exp(-lmd[i] * (max(0, x_i - lower_bound[i]))))
        return np.array(cdf)
        
    def _argcheck(self, *args):
        return 1
    
# Currently the upper bound of searching xb is set arbitrarily to 10**10 for all distributions.
trunc_expon = trunc_expon_gen(xa = 0, xb = 10 ** 10, name = 'trunc_expon', longname = 'Lower truncated exponential',
                              shapes = 'lmd, lower_bound')

class trunc_pareto_gen(rv_continuous):
    """Lower truncated Pareto (power) distribution
    
    Scipy based distribution class for the truncated exponential pdf, cdf and rvs
    
    Usage:
    PDF: trunc_pareto.pdf(list_of_xvals, b, lower_bound)
    CDF: trunc_pareto.cdf(list_of_xvals, b, lower_bound)
    Random Numbers: trunc_exp.rvs(b, lower_bound, size=1)
    
    """
    def _pdf(self, x, b, lower_bound):
        x = np.array(x)
        pdf = []
        for i, x_i in enumerate(x):
            if x_i < lower_bound[i]:
                pdf.append(0)
            else:
                pdf.append(b[i] * lower_bound[i] ** b[i] / x_i ** (b[i] + 1))
        return np.array(pdf)
    
    def _cdf(self, x, b, lower_bound):
        x = np.array(x)
        cdf = []
        for i, x_i in enumerate(x):
            cdf.append(max(0, 1 - (lower_bound[i] / x_i) ** b[i]))
        return np.array(cdf)

trunc_pareto = trunc_pareto_gen(xa = 0, xb = 10 ** 10, name = 'trunc_pareto', longname = 'Lower truncated Pareto', 
                                shapes = 'b, lower_bound')
    
class trunc_weibull_gen(rv_continuous):
    """Lower truncated Weibull distribution"""
    def _pdf(self, x, k, lmd, lower_bound):
        self.a = lower_bound
        x = np.array(x)
        pdf = k / lmd * (x / lmd) ** (k - 1) * exp(-(x / lmd) ** k) / exp(-(lower_bound / lmd) ** k)
        return pdf
    
    def _argcheck(self, *args):
        return 1

trunc_weibull = trunc_weibull_gen(name = 'trunc_weibull', longname = 'Lower truncated Weibull', 
                                  shapes = 'k, lmd, lower_bound')
            
def pln_ll(x, mu, sigma, lower_trunc = True, full_output = 0):
    """Log-likelihood of a truncated Poisson lognormal distribution
    
    Method derived from Bulmer 1974 Biometrics 30:101-110    
    
    Bulmer equation A1
    
    Adapted from Brian McGill's MATLAB function of the same name that was
    originally developed as part of the Palamedes software package by the
    National Center for Ecological Analysis and Synthesis working group on
    Tools and Fresh Approaches for Species Abundance Distributions
    (http://www.nceas.ucsb.edu/projects/11121)    
    
    """
    #purify abundance vector
    x = np.array(x)
    x = x[x > 0]
    x.sort()
    cts = histogram(x, bins = range(1, max(x) + 2))
    observed_abund_vals = cts[1][cts[0] != 0]
    counts = cts[0][cts[0] != 0]
    plik = pln.logpmf(observed_abund_vals, mu, sigma, lower_trunc, full_output = full_output)
    lik_list = np.array([], dtype = float)
    for i, count in enumerate(counts):
        lik_list = np.append(lik_list, count * plik[i])
    ll = sum(lik_list)
    return ll   

def logser_ll(x, p, upper_trunc = False, upper_bound = None):
    """Log-likelihood of a logseries distribution
    
    x  -  quantiles
    p  -  lower or upper tail probability 
    upper_trunc - whether the distribution is upper truncated
    upper_bound - the upper bound of the distribution, if upper_trunc is True
    
    """
    if upper_trunc:
        return sum(trunc_logser.logpmf(x, p, upper_bound))
    else:
        return sum(stats.logser.logpmf(x, p))

def trunc_weibull_ll(x, k, lmd, lower_bound):
    """Log-likelihood of the Weibull distributed lower truncated at lower_bound"""
    return sum(trunc_weibull.logpdf(x, k, lmd, lower_bound))

def disunif_ll(ab, low, high):
    """Log-likelihood of a discrete uniform distribution with bounds [low, high]"""
    n = len(ab)
    return n * log(1 / (high - low + 1))

def geom_ll(ab, p):
    """Log-likelihood of a geomtric distribution"""
    return sum(stats.geom.logpmf(ab, p))

def negbin_ll(ab, n, p):
    """Log-likelihood of a negative binomial dstribution (truncated at 1)"""
    return sum(log(stats.nbinom.pmf(ab, n, p) / (1 - stats.nbinom.pmf(0, n, p))))

def dis_gamma_ll(ab, k, theta):
    """Log-likelihood of a discrete gamma distribution
    
    k - shape parameter
    theta - scale parameter
    Normalization constant is calculated based on a cuf-off (currently set at 10**5)
    
    """
    cutoff = 1e5
    gamma_sum = sum(stats.gamma.pdf(range(1, cutoff + 1), k, scale = theta))
    C = 1 / gamma_sum
    return sum(log(stats.gamma.pdf(ab, k, scale = theta) * C))

def gen_yule_ll(ab, a, b):
    """Log-likelihood of the Yule-Simon distribution.
    
    Follow the configuration of generalized Yule distribution in Nee 2003. 
    The configuration on wikipedia and in Simon 1955 is the species case 
    where a = 1 and b = rho. 
    
    """
    ll = 0
    for ab_i in ab: 
        ll += log(b * special.gamma(a + b) * special.gamma(a + ab_i - 1) / 
                  special.gamma(a) /  special.gamma(a + b + ab_i))
    return ll

def yule_ll(ab, rho):
    """Log-likelihood of the original Yule-Simon distribution."""
    return gen_yule_ll(ab, 1, rho)

def pln_solver(ab, lower_trunc = True):
    """Given abundance data, solve for MLE of pln parameters mu and sigma
    
    Adapted from MATLAB code by Brian McGill that was originally developed as
    part of the Palamedes software package by the National Center for Ecological
    Analysis and Synthesis working group on Tools and Fresh Approaches for
    Species Abundance Distributions (http://www.nceas.ucsb.edu/projects/11121)
    
    """
    ab = np.array(ab)
    mu0 = mean(log(ab))
    sig0 = std(log(ab))
    def pln_func(x): 
        return -pln_ll(ab, x[0], x[1], lower_trunc, full_output = 1)
    mu, sigma = optimize.fmin(pln_func, x0 = [mu0, sig0], disp = 0)
    return mu, sigma

def trunc_logser_solver(ab):
    """Given abundance data, solve for MLE of truncated logseries parameter p"""
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15
    S = len(ab)
    N = sum(ab)
    m = np.array(range (1, int(N) + 1)) 
    y = lambda x: sum(x ** m / N * S) - sum((x ** m) / m)
    p = optimize.bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                        min((sys.float_info[0] / S) ** (1 / N), 2), xtol = 1.490116e-08)
    return p

def trunc_expon_solver(x, lower_bound):
    """Given a random sample and lower bound, 
    
    solve for MLE of lower truncated exponential distribution lmd.
    
    """
    return 1 / (np.mean(np.array(x)) - lower_bound)

def trunc_pareto_solver(x, lower_bound):
    """Given a random sample and lower bound,
    
    solve for MLE of lower truncated Pareto distribution b. 
    
    """
    x = np.array(x)
    return len(x) / sum(log(x) - log(lower_bound))

def negbin_solver(ab):
    """Given abundance data, solve for MLE of negative binomial parameters n and p"""
    mu = np.mean(ab)
    var = np.var(ab, ddof = 1)
    p0 = 1 - mu / var
    n0 = mu * (1 - p0) / p0
    def negbin_func(x): 
        return -negbin_ll(ab, x[0], x[1])
    n, p = optimize.fmin(negbin_func, x0 = [n0, p0])
    return n, p

def dis_gamma_solver(ab):
    """Given abundance data, solve for MLE of discrete gamma parameters k and theta"""
    mu = np.mean(ab)
    var = np.var(ab, ddof = 1)
    theta0 = var / mu
    k0 = mu / theta0
    def dis_gamma_func(x):
        return -dis_gamma_ll(ab, x[0], x[1])
    k, theta = optimize.fmin(dis_gamma_func, x0 = [k0, theta0])
    return k, theta 

def gen_yule_solver(ab):
    """Given abundance data, solve for MLE of generalized Yule distribution parameters a and b"""
    p1 = ab.count(1) / len(ab)
    a0 = 1                    
    b0 = p1 / (1-p1) * a0  # Initial guess based on expected frequency of singletons. NOT STABLE. 
    def gen_yule_func(x):
        return -gen_yule_ll(ab, x[0], x[1])
    a, b = optimize.fmin(gen_yule_func, x0 = [a0, b0])
    return a, b

def yule_solver(ab):
    """Given abundance data, solve for MLE of original Yule distribution parameter rho"""
    rho0 = np.mean(ab) / (np.mean(ab) - 1)
    def yule_func(x):
        return -yule_ll(ab, x)
    rho = optimize.fmin(yule_func, x0 = rho0)
    return rho