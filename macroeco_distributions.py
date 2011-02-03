"""Probability and Likelihood Functions for Distribution Testing"""

from math import factorial
from numpy import array, exp, histogram, log, matlib, sort, sqrt, pi
from scipy import integrate, stats

def pln_lik(mu,sigma,ab,approx_cut = 10):
    """Probability function of the Poisson lognormal distribution
    
    method derived from Bulmer 1974 Biometrics 30:101-110    
    adapted from Brian McGill's MATLAB function of the same name
    
    Bulmer equation 7 - approximation for large abundances
    Bulmer equation 2 - integral for small abundances
    
    """
   
    L = matlib.repmat(None, len(ab), 1)
    if mu <= 0 or sigma <= 0:
        L = 1e-120 #very unlikely to have negative params
        L.transpose()
    for i in range (0, len(ab)):
        ab1 = ab[i]
        if ab1 > approx_cut:
            #use approximation for large abundances    
            #Bulmer equation 7
            #tested vs. integral below - matches to about 6 significant digits for
            #intermediate abundances (10-50) - integral fails for higher
            #abundances, approx fails for lower abundances - 
            #assume it gets better for abundance > 50
            L[i,] = (1 / sqrt(2 * pi * sigma * sigma) / ab[i] * 
                exp(-(log(ab[i]) - mu) ** 2 / (2 * sigma ** 2)) * 
                (1 + 1 / (2 * ab[i] * sigma * sigma) * 
                 ((log(ab[i]) - mu) ** 2 / (sigma * sigma) + log(ab[i]) - mu - 1)))
        else:
            # Bulmer equation 2 -tested against Grundy Biometrika 38:427-434
            # Table 1 & Table 2 and matched to the 4 decimals in the table except
            # for very small mu (10^-2)
            # having the /gamma(ab1+1) inside the integrand is inefficient but
            # avoids pseudo-singularities            
            # split integral into two so the quad function finds the peak
            # peak apppears to be just below ab1 - for very small ab1 (ab1<10)
            # works better to leave entire peak in one integral and integrate 
            # the tail in the second integral           
            if ab1 < 10:
                ub = 10
            else: 
                ub = ab1       
            term1 = ((2 * pi * sigma **2) ** -0.5)/ factorial(ab1)
            #integrate low end where peak is so it finds peak
            term2a = integrate.quad(lambda x: ((x ** (ab1 - 1)) * 
                                               (exp(-x)) * 
                                               exp(-(log(x) - mu) ** 2 / 
                                                (2 * sigma ** 2))), 0.00001, ub)
            #integrate higher end for accuracy and in case peak moves
            term2b = integrate.quad(lambda x: ((x ** (ab1 - 1)) * 
                                               (exp(-x)) * exp(-(log(x) - mu) ** 
                                                               2/ (2 * sigma ** 
                                                                   2))), ub, 1e5)
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
    counts = cts[0] 
    plik = log(array(pln_lik(mu, sigma, 
                             range(1, (len(cts[0]) + 1))), dtype = float))
    term1 = matlib.repmat(None, max(ab), 1)    
    for i in range(0, len(counts)):
        term1[i,] = counts[i] * plik[i]
    term2 = len(ab) * log(1 - array(pln_lik(mu, sigma, array(matlib.zeros((1,1)), 
                                                           dtype = int)), 
                                    dtype = float))
    ll = sum(term1)[0] - term2[0]
    return ll[0]

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