"""Probability and Likelihood Functions for Distribution Testing"""
#TODO convert from importing from numpy to importing numpy as np

from __future__ import division
from math import factorial
from numpy import array, exp, histogram, log, matlib, sort, sqrt, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize

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

def logser_ll(x, p):
    """Log-likelihood of a logseries distribution
    
    x  -  quantiles
    p  -  lower or upper tail probability 
    
    """
    return sum(log(stats.logser.pmf(x, p)))

def disunif_ll(ab, low, high):
    """Log-likelihood of a discrete uniform distribution with bounds [low, high]"""
    n = len(ab)
    return n * log(1 / (high - low + 1))

def geom_ll(ab, p):
    """Log-likelihood of a geomtric distribution"""
    return sum(log(stats.geom.pmf(ab, p)))

def negbin_ll(ab, n, p):
    """Log-likelihood of a negative binomial dstirbution (truncated at 1)"""
    return sum(log(stats.nbinom.pmf(ab, n, p) / (1 - stats.nbinom.pmf(0, n, p))))

def negbin_solver(ab):
    """Given abundance data, solve for MLE of negative binomial parameters n and p"""
    mu = np.mean(ab)
    sig = np.var(ab, ddof = 1)
    p0 = 1 - mu / sig
    n0 = mu * (1 - p0) / p0
    def negbin_func(x): 
        return -negbin_ll(ab, x[0], x[1])
    n, p = optimize.fmin(negbin_func, x0 = [n0, p0])
    return n, p