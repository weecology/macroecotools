"""Probability and Likelihood Functions for Distribution Testing"""

from __future__ import division
from math import factorial, floor
from numpy import exp, histogram, log, matlib, sort, sqrt, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize, special
from scipy.stats import rv_discrete, rv_continuous

class poisson_lognormal(rv_discrete):
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
            if lower_trunc[i]:
                pmf_i = untrunc_pmf(x_i, mu[i], sigma[i]) / (1 - untrunc_pmf(0, mu[i], sigma[i]))
            else:
                pmf_i = untrunc_pmf(x_i, mu[i], sigma[i])
            pmf.append(pmf_i)
        return np.array(pmf)

    def _argcheck(self, *args):
        return 1
    
    def ll(self, x, full_output = 0):
        """Log-likelihood of a truncated Poisson lognormal distribution
        
        Method derived from Bulmer 1974 Biometrics 30:101-110    
        
        Bulmer equation A1
        
        Adapted from Brian McGill's MATLAB function of the same name that was
        originally developed as part of the Palamedes software package by the
        National Center for Ecological Analysis and Synthesis working group on
        Tools and Fresh Approaches for Species Abundance Distributions
        (http://www.nceas.ucsb.edu/projects/11121)    
        
        """
        mu = self.mu
        sigma = self.sigma
        #purify abundance vector
        x = np.array(x)
        x = x[x > 0]
        x.sort()
        cts = histogram(x, bins = range(1, max(x) + 2))
        observed_abund_vals = cts[1][cts[0] != 0]
        counts = cts[0][cts[0] != 0]
        plik = log(self.pmf(observed_abund_vals, full_output = full_output))
        term1 = array([], dtype = float)
        for i, count in enumerate(counts):
            term1 = np.append(term1, count * plik[i])
        
        #Renormalization for zero truncation
        term2 = len(x) * log(1 - self.pmf([0], full_output=full_output)[0])
    
        ll = sum(term1) - term2
        return ll

def pln_solver(ab):
    """Given abundance data, solve for MLE of pln parameters mu and sigma
    
    Adapted from MATLAB code by Brian McGill that was originally developed as
    part of the Palamedes software package by the National Center for Ecological
    Analysis and Synthesis working group on Tools and Fresh Approaches for
    Species Abundance Distributions (http://www.nceas.ucsb.edu/projects/11121)
    
    """

    mu0 = mean(log(ab))
    sig0 = std(log(ab))
    def pln_func(x): 
        return -pln_ll(x[0], x[1], ab, full_output=1)
    mu, sigma = optimize.fmin(pln_func, x0 = [mu0, sig0], disp=0)
    return mu, sigma

def logser_trunc_solver(ab):
    """Given abundance data, solve for MLE of truncated logseries parameter p"""
    S = len(ab)
    N = sum(ab)
    m = array(range (1, N+1)) 
    b = 1e-99
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15
    y = lambda x: S / N * sum(x ** m) - log(1 / (1 - x)) + special.betainc(N + 1, b, x) * special.beta(N + 1, b)
    p = optimize.bisect(y, BOUNDS[0] + DIST_FROM_BOUND, BOUNDS[1] - DIST_FROM_BOUND, 
                                        xtol = 1.490116e-08)
    return p

def logser_ll(x, p):
    """Log-likelihood of a logseries distribution
    
    x  -  quantiles
    p  -  lower or upper tail probability 
    
    """
    return sum(log(stats.logser.pmf(x, p)))

def trunc_logser_pmf(x, p, upper_bound):
    """Probability mass function for the upper truncated log-series"""
    if p < 1:
        return stats.logser.pmf(x, p) / stats.logser.cdf(upper_bound, p)
    else:
        x = np.array(x)
        ivals = np.arange(1, upper_bound + 1)
        normalization = sum(p ** ivals / ivals)
        pmf = (p ** x / x) / normalization
        return pmf

def trunc_logser_cdf(x_max, p, upper_bound):
    """Cumulative probability function for the upper truncated log-series"""
    if p < 1:
        return stats.logser.cdf(x_max, p) / stats.logser.cdf(upper_bound, p)
    else:
        x_list = range(1, int(x_max) + 1)
        cdf = sum(trunc_logser_pmf(x_list, p, upper_bound))
        return cdf

def disunif_ll(ab, low, high):
    """Log-likelihood of a discrete uniform distribution with bounds [low, high]"""
    n = len(ab)
    return n * log(1 / (high - low + 1))

def geom_ll(ab, p):
    """Log-likelihood of a geomtric distribution"""
    return sum(log(stats.geom.pmf(ab, p)))

def negbin_ll(ab, n, p):
    """Log-likelihood of a negative binomial dstribution (truncated at 1)"""
    return sum(log(stats.nbinom.pmf(ab, n, p) / (1 - stats.nbinom.pmf(0, n, p))))

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

def gen_yule_solver(ab):
    """Given abundance data, solve for MLE of generalized Yule distribution parameters a and b"""
    p1 = ab.count(1) / len(ab)
    a0 = 1                    
    b0 = p1 / (1-p1) * a0  # Initial guess based on expected frequency of singletons. NOT STABLE. 
    def gen_yule_func(x):
        return -gen_yule_ll(ab, x[0], x[1])
    a, b = optimize.fmin(gen_yule_func, x0 = [a0, b0])
    return a, b

def yule_ll(ab, rho):
    """Log-likelihood of the original Yule-Simon distribution."""
    return gen_yule_ll(ab, 1, rho)

def yule_solver(ab):
    """Given abundance data, solve for MLE of original Yule distribution parameter rho"""
    rho0 = np.mean(ab) / (np.mean(ab) - 1)
    def yule_func(x):
        return -yule_ll(ab, x)
    rho = optimize.fmin(yule_func, x0 = rho0)
    return rho