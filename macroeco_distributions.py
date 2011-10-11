"""Probability and Likelihood Functions for Distribution Testing"""
#TODO convert from importing from numpy to importing numpy as np

from __future__ import division
from math import factorial, floor
from numpy import array, exp, histogram, log, matlib, sort, sqrt, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize, special

def pln_lik(mu,sigma,abund_vect,approx_cut = 10):
    #TODO remove all use of matrices unless they are necessary for some
    #     unforseen reason
    """Probability function of the Poisson lognormal distribution
    
    method derived from Bulmer 1974 Biometrics 30:101-110    
    adapted from Brian McGill's MATLAB function of the same name
    
    Bulmer equation 7 - approximation for large abundances
    Bulmer equation 2 - integral for small abundances
    
    """
   
    L = matlib.repmat(None, len(abund_vect), 1)
    if sigma <= 0:
        L = matlib.repmat(1e-120, len(abund_vect), 1) #very unlikely to have negative params
    else:
        for i, ab in enumerate(abund_vect):
            if ab > approx_cut:
            #use approximation for large abundances    
            #Bulmer equation 7
            #tested vs. integral below - matches to about 6 significant digits for
            #intermediate abundances (10-50) - integral fails for higher
            #abundances, approx fails for lower abundances - 
            #assume it gets better for abundance > 50
                V = sigma ** 2
                L[i,] = (1 / sqrt(2 * pi * V) / ab *
                         exp(-(log(ab) - mu) ** 2 / (2 * V)) *
                         (1 + 1 / (2 * ab * V) * ((log(ab) - mu) ** 2 / V +
                                                  log(ab) - mu - 1)))
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
                if ab < 10:
                    ub = 10
                else: 
                    ub = ab       
                term1 = ((2 * pi * sigma ** 2) ** -0.5)/ factorial(ab)
            #integrate low end where peak is so it finds peak
                term2a = integrate.quad(lambda x: ((x ** (ab - 1)) * 
                                               (exp(-x)) * 
                                               exp(-(log(x) - mu) ** 2 / 
                                                (2 * sigma ** 2))), 0, ub)
            #integrate higher end for accuracy and in case peak moves
                term2b = integrate.quad(lambda x: ((x ** (ab - 1)) * 
                                               (exp(-x)) * exp(-(log(x) - mu) ** 
                                                               2/ (2 * sigma ** 
                                                                   2))), ub, float('inf'))
                Pr = term1 * term2a[0]
                Pr_add = term1 * term2b[0]                
                L[i,] = Pr + Pr_add            
            
                if L[i,] <= 0:
                #likelihood shouldn't really be zero and causes problem taking 
                #log of it
                    L[i,] = 1e-120
    return (L)

def pln_ll(mu,sigma,ab):
    """Log-likelihood of a truncated Poisson lognormal distribution
    
    method derived from Bulmer 1974 Biometrics 30:101-110    
    adapted from Brian McGill's MATLAB function of the same name
    
    Bulmer equation A1 - note 2nd term is renormalization for zero-truncation
    otherwise is just standard likelihood - relies on pln_lik function above to
    calculate probability for each r
    
    """
    #purify abundance vector
    ab = array(ab)
    ab.transpose()
    ab = ab[ab>0]
    ab.sort()
    
    cts = histogram(ab, bins = range(1, max(ab) + 2))
    observed_abund_vals = cts[1][cts[0] != 0]
    counts = cts[0][cts[0] != 0]
    plik = log(array(pln_lik(mu, sigma, observed_abund_vals), dtype = float))
    term1 = array([], dtype = float)
    for i, count in enumerate(counts):
        term1 = np.append(term1, count * plik[i])
    term2 = len(ab) * log(1 - array(pln_lik(mu, sigma, [0]), dtype = float))
    ll = sum(term1) - term2
    return ll[0]

def pln_solver(ab):
    """Given abundance data, solve for MLE of pln parameters mu and sigma"""
    mu0 = mean(log(ab))
    sig0 = std(log(ab))
    def pln_func(x): 
        return -pln_ll(x[0], x[1], ab)
    mu, sigma = optimize.fmin(pln_func, x0 = [mu0, sig0])
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
    x = np.array(x)
    ivals = np.arange(1, upper_bound + 1)
    normalization = sum(p ** ivals / ivals)
    pmf = (p ** x / x) / normalization
    return pmf

def trunc_logser_cdf(x_max, p, upper_bound):
    """Cumulative probability function for the upper truncated log-series"""
    x_list = range(1, floor(x_max) + 1)
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