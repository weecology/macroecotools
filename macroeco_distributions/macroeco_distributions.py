"""Probability and Likelihood Functions for Distribution Testing

Probability distributions
    Poisson lognormal distribution
    Upper truncated logseries distribution
    Lower truncated exponential distribution
    Lower truncated Pareto (power) distribution
    Lower truncated Weibull distribution
    Upper truncated geometric distribution (without zeros)
    Upper truncated geometric distribution (with zeros)
    Generalized Yule distribution
    Original Yule distribution
    
    
Likelihood functions
    Log-likelihood truncated Poisson lognormal distribution
    Log-likelihood logseries distribution
    Log-likelihood lower truncated Weibull distribution
    Log-likelihood of a discrete uniform distribution with bounds [low, high]
    Log-likelihood of an untruncated geometric distribution
    Log-likelihood of an upper-truncated geometric distribution
    Log-likelihood of a negative binomial dstribution (truncated at 1)
    Log-likelihood of a discrete gamma distribution
    Log-likelihood of the generalized Yule distribution
    Log-likelihood of the original Yule-Simon distribution
    Log-likelihood of the Zipf distribution with x_min = 1

"""

from __future__ import division
import sys
from math import factorial, floor, sqrt
from numpy import exp, histogram, log, matlib, sort, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize, special
from scipy.stats import rv_discrete, rv_continuous, itemfreq
from scipy.optimize import bisect, fsolve
from scipy.special import logit, expit
from scipy.integrate import quad
import warnings

#._rvs method is not currently available for pln.
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
    def _pmf(self, x, mu, sigma, lower_trunc, approx_cut = 10):
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
                    pmf_i = 1 / sqrt(2 * pi * V) / x_i * \
                           exp(-(log(x_i) - mu) ** 2 / (2 * V)) * \
                           (1 + 1 / (2 * x_i * V) * ((log(x_i) - mu) ** 2 / V + \
                                                    log(x_i) - mu- 1))
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
                    eq = lambda t: np.exp(t * x_i - np.exp(t) - ((t - mu) / sigma) ** 2 / 2)
                    term2a = integrate.quad(eq, -np.inf, np.log(ub), full_output = 0, limit = 500)
                    #integrate higher end for accuracy and in case peak moves
                    term2b = integrate.quad(eq, np.log(ub), np.inf, full_output = 0, limit = 500)
                    Pr = term1 * (term2a[0] + term2b[0])
                    if Pr > 0: 
                    #likelihood shouldn't really be zero and causes problem taking 
                    #log of it
                        pmf_i = Pr  
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
    
    def _cdf(self, x, mu, sigma, lower_trunc, approx_cut = 10):
        x = np.array(x)
        cdf = []
        for x_i in x:
            cdf.append(sum(self.pmf(range(int(x_i) + 1), mu[0], sigma[0], lower_trunc[0])))
        return np.array(cdf)
    
    def _ppf(self, cdf, mu, sigma, lower_trunc, approx_cut = 10):
        cdf = np.array(cdf)
        ppf = []
        for cdf_i in cdf:
            ppf_i = 1
            while self.cdf(ppf_i, mu, sigma, lower_trunc, approx_cut = approx_cut) < cdf_i:
                ppf_i += 1
            ppf.append(ppf_i)
        return np.array(ppf)
    
    def _rvs(self, mu, sigma, lower_trunc):
        if not lower_trunc:
            pois_par = np.exp(stats.norm.rvs(loc = mu, scale = sigma, size = self._size))
            ab = stats.poisson.rvs(pois_par, size = self._size)
        else:
            ab = []
            while len(ab) < self._size:
                pois_par = np.exp(stats.norm.rvs(loc = mu, scale = sigma))
                ab_single = stats.poisson.rvs(pois_par)
                if ab_single: ab.append(ab_single)
        return np.array(ab)

    def _argcheck(self, *args):
        if args[2] is True: self.a = 1
        else: self.a = 0
        cond = args[1] > 0
        return cond
    
pln = pln_gen(name='pln', longname='Poisson lognormal', 
              shapes = 'mu, sigma, lower_trunc')

class trunc_logser_gen(rv_discrete):
    """Upper truncated logseries distribution
    
    Scipy based distribution class for the truncated logseries pmf, cdf and rvs
    Note that because this function is upper-truncated, its parameter p could be larger than 1, 
    unlike the untruncated logseries where 0<p<1.
    
    Usage:
    PMF: trunc_logser.pmf(list_of_xvals, p, upper_bound)
    CDF: trunc_logser.cdf(list_of_xvals, p, upper_bound)
    Random Numbers: trunc_logser.rvs(p, upper_bound, size=1)
    
    """
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        if p[0] < 1:
            return stats.logser.pmf(x, p) / stats.logser.cdf(upper_bound, p)
        else:
            ivals = np.arange(1, upper_bound[0] + 1)
            normalization = sum(p[0] ** ivals / ivals)
            pmf = (p[0] ** x / x) / normalization
            return pmf
    
    def _logpmf(self, x, p, upper_bound):
        x = np.array(x)
        if p[0] < 1:
            return stats.logser.logpmf(x, p) - stats.logser.logcdf(upper_bound, p)
        else:
            ivals = np.arange(1, upper_bound[0] + 1)
            normalization = sum(p[0] ** ivals / ivals)
            logpmf = x * log(p[0]) - log(x) - log(normalization)
            return logpmf
        
    def _cdf(self, x, p, upper_bound):
        x = np.array(x)
        if p[0] < 1:
            return stats.logser.cdf(x, p) / stats.logser.cdf(upper_bound, p)
        else:
            cdf_list = [sum(self.pmf(range(1, int(x_i) + 1), p[0], upper_bound[0])) for x_i in x]
            return np.array(cdf_list)
    
    def _rvs(self, p, upper_bound):
        out = []
        if p < 1:
            for i in range(self._size):
                rand_logser = stats.logser.rvs(p)
                while rand_logser > upper_bound:
                    rand_logser = stats.logser.rvs(p)
                out.append(rand_logser)
        else:
            rand_list = stats.uniform.rvs(size = self._size)
            for rand_num in rand_list:
                y = lambda x: self.cdf(x, p, upper_bound) - rand_num
                if y(1) > 0: out.append(1)
                else: out.append(int(round(bisect(y, 1, upper_bound))))
        return np.array(out)
    
    def _argcheck(self, *args):
        self.a = 1
        self.b = args[1]
        cond = (args[0] > 0) and (args[1] >= 1)
        return cond

trunc_logser = trunc_logser_gen(a=1, name='trunc_logser',
                                longname='Upper truncated logseries',
                                shapes="p, upper_bound",
                                extradoc="""Truncated logseries
                                
                                Upper truncated logseries distribution
                                """
                                )

class trunc_expon_gen(rv_continuous):
    """Lower truncated exponential distribution
    
    Scipy based distribution class for the truncated exponential pdf, cdf and rvs
    
    Usage:
    PDF: trunc_expon.pdf(list_of_xvals, lambda, lower_bound)
    CDF: trunc_expon.cdf(list_of_xvals, lambda, lower_bound)
    Random Numbers: trunc_expon.rvs(lambda, lower_bound, size=1)
    
    """
    def _pdf(self, x, lmd, lower_bound):
        return stats.expon.pdf(x, scale = 1/lmd, loc = lower_bound)
    
    def _logpdf(self, x, lmd, lower_bound):
        return stats.expon.logpdf(x, scale = 1/lmd, loc = lower_bound)
    
    def _cdf(self, x, lmd, lower_bound):
        return stats.expon.cdf(x, scale = 1/lmd, loc = lower_bound)
    
    def _rvs(self, lmd, lower_bound):
        return stats.expon.rvs(scale = 1/lmd, loc = lower_bound, size = self._size)

    def _argcheck(self, *args):
        self.a = args[1]
        self.xa = args[1]
        self.xb = 10 ** 10 # xb is arbitrarily set to a large number
        cond = (args[0] > 0) & (args[1] >= 0)
        return cond
    
# Currently the upper bound of searching xb is set arbitrarily to 10**10 for all distributions.
trunc_expon = trunc_expon_gen(name = 'trunc_expon', longname = 'Lower truncated exponential',
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
        return stats.pareto.pdf(x, b, scale = lower_bound)
    
    def _logpdf(self, x, b, lower_bound):
        return stats.pareto.logpdf(x, b, scale = lower_bound)
    
    def _cdf(self, x, b, lower_bound):
        return stats.pareto.cdf(x, b, scale = lower_bound)
    
    def _rvs(self, b, lower_bound):
        rand_num = stats.pareto.rvs(b, scale = lower_bound, size = self._size)
        return rand_num
    
    def _argcheck(self, *args):
        self.a = args[1]
        self.xa = args[1]
        self.xb = 10 ** 10
        cond = (args[0] > 0) & (args[1] > 0)
        return cond

trunc_pareto = trunc_pareto_gen(name = 'trunc_pareto', longname = 'Lower truncated Pareto', 
                                shapes = 'b, lower_bound')
    
class trunc_weibull_gen(rv_continuous):
    """Lower truncated Weibull distribution"""
    def _pdf(self, x, k, lmd, lower_bound):
        x = np.array(x)
        pdf = k / lmd * (x / lmd) ** (k - 1) * exp(-(x / lmd) ** k) / exp(-(lower_bound / lmd) ** k)
        #Alternative way of formulating pdf (same results, speed might differ):
        #pdf = stats.frechet_r.pdf(x, k, scale = lmd) / (1 - stats.frechet_r.cdf(lower_bound, k, scale = lmd))
        return pdf
    
    def _logpdf(self, x, k, lmd, lower_bound):
        x = np.array(x)
        logpdf = log(k / lmd) + (k - 1) * log(x / lmd) - (x / lmd) ** k + (lower_bound / lmd) ** k 
        return logpdf
    
    def _cdf(self, x, k, lmd, lower_bound):
        x = np.array(x)
        cdf = (stats.frechet_r.cdf(x, k, scale = lmd) 
               - stats.frechet_r.cdf(lower_bound, k, scale = lmd)) / (1 - stats.frechet_r.cdf(lower_bound, k, scale = lmd))
        return cdf
    
    def _rvs(self, k, lmd, lower_bound):
        rand_num = stats.frechet_r.rvs(k, scale = lmd, size = self._size)
        rand_num = rand_num[rand_num >= lower_bound]
        while (len(rand_num) < self._size):
            rand_new = stats.frechet_r.rvs(k, scale = lmd)
            if rand_new >= lower_bound:
                rand_num = np.append(rand_num, rand_new)
        return rand_num
    
    def _argcheck(self, *args):
        self.a = args[2]
        self.xa = args[2]
        self.xb = 10 ** 10
        cond = (args[0] > 0) & (args[1] > 0) & (args[2] >= 0)
        return cond

trunc_weibull = trunc_weibull_gen(name = 'trunc_weibull', longname = 'Lower truncated Weibull', 
                                  shapes = 'k, lmd, lower_bound')

class trunc_geom_gen(rv_discrete):
    """Upper truncated geometric distribution (without zeros)"""
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        pmf = (1 - p) ** (x - 1) * p / (1 - (1 - p) ** upper_bound)
        return pmf
    
    def _logpmf(self, x, p, upper_bound):
        x = np.array(x)
        logpmf = (x - 1) * log(1 - p) + log(p) - log(1 - (1 - p) ** upper_bound)
        return logpmf
    
    def _cdf(self, x, p, upper_bound):
        x = np.array(x)
        cdf = (1 - (1 - p) ** x) / (1 - (1 - p) ** upper_bound)
        return cdf
    
    def _ppf(self, cdf, p, upper_bound):
        cdf = np.array(cdf)
        x = np.log(1 - cdf * (1 - (1 - p) ** upper_bound)) / np.log(1 - p)
        return np.ceil(x)
    
    def _rvs(self, p, upper_bound):
        rand_num = stats.geom.rvs(p, size = self._size)
        rand_num = rand_num[rand_num <= upper_bound]
        while (len(rand_num) < self._size):
            rand_new = stats.geom.rvs(p)
            if rand_new <= upper_bound:
                rand_num = np.append(rand_num, rand_new)
        return rand_num
    
    def _argcheck(self, *args):
        self.a = 1
        self.b = args[1]
        cond = (args[0] > 0) & (args[1] >= 1) 
        return cond

trunc_geom = trunc_geom_gen(name = 'trunc_geom', longname = 'Upper truncated geometric', 
                                  shapes = 'p, upper_bound')

class trunc_geom_with_zeros_gen(rv_discrete):
    """Upper truncated geometric distribution (with zeros)"""
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        pmf = (1 - p) ** x * p / (1 - (1 - p) ** (upper_bound + 1))
        return pmf
    
    def _logpmf(self, x, p, upper_bound):
        x = np.array(x)
        logpmf = x * log(1 - p) + log(p) - log(1 - (1 - p) ** (upper_bound + 1))
        return logpmf
    
    def _cdf(self, x, p, upper_bound):
        x = np.array(x)
        cdf = (1 - (1 - p) ** (x + 1)) / (1 - (1 - p) ** (upper_bound + 1))
        return cdf
    
    def _ppf(self, cdf, p, upper_bound):
        cdf = np.array(cdf)
        x = np.log(1 - cdf * (1 - (1 - p) ** (upper_bound + 1))) /\
            np.log(1 - p) - 1
        return np.ceil(x)
    
    def _argcheck(self, *args):
        self.a = 0
        self.b = args[1]
        cond = (args[0] > 0) & (args[1] >= 0) 
        return cond

trunc_geom_with_zeros = trunc_geom_with_zeros_gen(name = 'trunc_geom_with_zeros', 
                                                  longname = 'Upper truncated geometric with zeros', 
                                                  shapes = 'p, upper_bound')

class nbinom_lower_trunc_gen(rv_discrete):
    """Negative binomial distribution lower-truncated at 1"""
    def _pmf(self, x, n, p):
        pmf = stats.nbinom.pmf(x, n, p) / (1 - stats.nbinom.pmf(0, n, p))
        return pmf
        
    def _logpmf(self, x, n, p):
        logpmf = stats.nbinom.logpmf(x, n, p) - stats.nbinom.logsf(0, n, p)
        return logpmf
    
    def _cdf(self, x, n, p):
        x = np.array(x)
        cdf = (stats.nbinom.cdf(x, n, p) - stats.nbinom.pmf(0, n, p)) / (1 - stats.nbinom.pmf(0, n, p))
        return np.array(cdf)
    
    def _ppf(self, cdf, n, p):
        cdf = np.array(cdf)
        if len(cdf) > 1: n, p = n[0], p[0]
        ppf = []
        for cdf_i in cdf:
            ppf_i = 1
            while self.cdf(ppf_i, n, p) < cdf_i:
                ppf_i += 1
            ppf.append(ppf_i)
        return np.array(ppf)
    
    def _rvs(self, n, p):
        cdf_list = stats.uniform.rvs(size = self._size)
        return self.ppf(cdf_list, n, p)
                        
    def _argcheck(self, *args):
        self.a = 1
        cond = (args[0] > 0) & (args[1] > 0) & (args[1] < 1) 
        return cond

nbinom_lower_trunc = nbinom_lower_trunc_gen(name = 'nbinom_lower_trunc', 
                                                  longname = 'Negative binomial truncated at 1', 
                                                  shapes = 'n, p')

def pln_ll(x, mu, sigma, lower_trunc = True):
    """Log-likelihood of a truncated Poisson lognormal distribution
    
    Method derived from Bulmer 1974 Biometrics 30:101-110    
    
    Bulmer equation A1
    
    Adapted from Brian McGill's MATLAB function of the same name that was
    originally developed as part of the Palamedes software package by the
    National Center for Ecological Analysis and Synthesis working group on
    Tools and Fresh Approaches for Species Abundance Distributions
    (http://www.nceas.ucsb.edu/projects/11121)    
    
    """
    x = np.array(x)
    uniq_counts = itemfreq(x)
    unique_vals, counts = zip(*uniq_counts)
    plik = pln.logpmf(unique_vals, mu, sigma, lower_trunc)
    ll = 0
    for i, count in enumerate(counts):
        ll += count * plik[i]
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
    """Log-likelihood of the Weibull distribution lower truncated at lower_bound"""
    return sum(trunc_weibull.logpdf(x, k, lmd, lower_bound))

def disunif_ll(ab, low, high):
    """Log-likelihood of a discrete uniform distribution with bounds [low, high]"""
    n = len(ab)
    return n * log(1 / (high - low + 1))

def geom_ll(ab, p):
    """Log-likelihood of a geometric distribution"""
    return sum(stats.geom.logpmf(ab, p))

def trunc_geom_ll(ab, p, upper_bound):
    """Log-likelhood of an upper-truncated geometric distribution"""
    return geom_ll(ab, p) - len(ab) * stats.geom.logcdf(upper_bound, p)

def nbinom_lower_trunc_ll(ab, n, p):
    """Log-likelihood of a negative binomial dstribution (truncated at 1)"""
    return sum(nbinom_lower_trunc.logpmf(ab, n, p))

def dis_gamma_ll(ab, k, theta):
    """Log-likelihood of a discrete gamma distribution
    
    k - shape parameter
    theta - scale parameter
    Normalization constant is calculated based on a cuf-off (currently set at 10**5)
    
    """
    cutoff = 1e5
    gamma_sum = sum(stats.gamma.pdf(range(1, int(cutoff) + 1), k, scale = theta))
    C = 1 / gamma_sum
    return sum(stats.gamma.logpdf(ab, k, scale = theta) + log(C))

def gen_yule_ll(ab, a, rho):
    """Log-likelihood of the Yule-Simon distribution.
    
    Follow the configuration of generalized Yule distribution in Nee 2003. 
    The configuration on wikipedia and in Simon 1955 is the special case 
    where a = 1. 
    
    """
    ll = len(ab) * (log(rho) + special.gammaln(a + rho) - special.gammaln(a))
    for ab_i in ab: 
        ll += special.gammaln(a + ab_i -1) - special.gammaln(a + rho + ab_i)
    return ll

def yule_ll(ab, rho):
    """Log-likelihood of the original Yule-Simon distribution."""
    return gen_yule_ll(ab, 1, rho)

def zipf_ll(ab, a):
    """Log-likelihood of the Zipf distribution with x_min = 1."""
    return sum(stats.zipf.logpmf(ab, a))

def pln_solver(ab, lower_trunc = True):
    """Given abundance data, solve for MLE of pln parameters mu and sigma
    
    Adapted from MATLAB code by Brian McGill that was originally developed as
    part of the Palamedes software package by the National Center for Ecological
    Analysis and Synthesis working group on Tools and Fresh Approaches for
    Species Abundance Distributions (http://www.nceas.ucsb.edu/projects/11121)
    
    """
    ab = np.array(ab)
    if lower_trunc is True:
        ab = check_for_support(ab, lower = 1)
    else: ab = check_for_support(ab, lower = 0)
    mu0 = mean(log(ab[ab > 0]))
    sig0 = std(log(ab[ab > 0]))
    def pln_func(x): 
        return -pln_ll(ab, x[0], exp(x[1]), lower_trunc)
    mu, logsigma = optimize.fmin_bfgs(pln_func, x0 = [mu0, log(sig0)], disp = 0)
    sigma = exp(logsigma)
    return mu, sigma

def logser_solver(ab):
    """Given abundance data, solve for MLE of logseries parameter p."""
    ab = check_for_support(ab, lower = 1)
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15    
    y = lambda x: 1 / log(1 / (1 - expit(x))) * expit(x) / (1 - expit(x)) - sum(ab) / len(ab)
    x = bisect(y, logit(BOUNDS[0] + DIST_FROM_BOUND), logit(BOUNDS[1] - DIST_FROM_BOUND),
                xtol = 1.490116e-08)
    return expit(x)

def trunc_logser_solver(ab, upper_bound = None):
    """Given abundance data, solve for MLE of truncated logseries parameter p. 
    
    Note that because the distribution is truncated, p can be larger than 1.
    If an upper bound is not given, it takes the default value sum(ab).
    
    """
    if upper_bound is None: upper_bound = sum(ab)
    ab = check_for_support(ab, lower = 1, upper = upper_bound)
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15
    S = len(ab)
    N = sum(ab)
    m = np.array(range (1, int(upper_bound) + 1)) 
    y = lambda x: sum(x ** m / N * S) - sum((x ** m) / m)
    x0 = logser_solver(ab)
    p = optimize.fsolve(y, x0, xtol = 1.490116e-08)[0]
    return p

def trunc_geom_solver(ab, upper_bound):
    """Given abundance data, solve for MLE of upper-truncated geometric distribution parameter p"""
    ab = check_for_support(ab, lower = 1, upper = upper_bound)
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -10
    S = len(ab)
    N = sum(ab)
    y = lambda x: ((S * upper_bound - N) * (1 - x) + S) * x ** upper_bound - S + N * (1 - x)
    one_minus_p = optimize.bisect(y, 1 - S/N - DIST_FROM_BOUND, BOUNDS[1] - DIST_FROM_BOUND, 
                                  xtol = 1.490116e-16)
    return 1 - one_minus_p

def trunc_expon_solver(x, lower_bound):
    """Given a random sample and lower bound, 
    
    solve for MLE of lower truncated exponential distribution lmd.
    
    """
    x = check_for_support(x, lower = lower_bound)
    return 1 / (np.mean(np.array(x)) - lower_bound)

def trunc_pareto_solver(x, lower_bound):
    """Given a random sample and lower bound,
    
    solve for MLE of lower truncated Pareto distribution b. 
    
    """
    x = check_for_support(x, lower = lower_bound)
    return len(x) / sum(log(x) - log(lower_bound))

def nbinom_lower_trunc_solver(ab):
    """Given abundance data, solve for MLE of negative binomial (lower-truncated at 1) parameters n and p"""
    ab = check_for_support(ab, lower = 1)
    mu = np.mean(ab)
    var = np.var(ab, ddof = 1)
    p0 = 1 - mu / var
    if p0 < 0: p0 = 10**-5
    elif p0 > 1: p0 = 1 - 10**-5
    logit_p0 = logit(p0)
    log_n0 = log(mu * (1 - p0) / p0)
    def negbin_func(x):
        return -nbinom_lower_trunc_ll(ab, exp(x[0]), expit(x[1]))
    log_n, logit_p = optimize.fmin(negbin_func, x0 = [log_n0, logit_p0])
    return exp(log_n), expit(logit_p)

def dis_gamma_solver(ab):
    """Given abundance data, solve for MLE of discrete gamma parameters k and theta"""
    ab = check_for_support(ab, lower = 0)
    mu = np.mean(ab)
    var = np.var(ab, ddof = 1)
    theta0 = var / mu
    k0 = mu / theta0
    def dis_gamma_func(x):
        return -dis_gamma_ll(ab, x[0], x[1])
    k, theta = optimize.fmin(dis_gamma_func, x0 = [k0, theta0])
    return k, theta 

def gen_yule_solver(ab):
    """Given abundance data, solve for MLE of generalized Yule distribution parameters a and b(rho)
    
    Algorithm extended from Garcia 2011.
    
    
    """
    ab = check_for_support(ab, lower = 1)
    a0 = 1
    rho0 = np.mean(ab) / (np.mean(ab) - 1)
    tol = 1.490116e-08
    loop_end = False
    count_one = len([k for k in ab if k == 1])
    ab_not_one = [k for k in ab if k != 1]
    max_iter = 1000 # maximum number of iterations 
    i = 0
    while (not loop_end) and (i < max_iter):
        rho1 = len(ab) / sum([sum([1 / (rho0 + j + a0) for j in range(0, k)]) for k in ab])
        func_a = lambda a: 1 / (a + rho1) * count_one + sum([1 / (a+rho1+k-1) - \
                                                             sum([rho1/(a+rho1+m)/(a+m) for m in range(0, k - 1)])\
                                                             for k in ab_not_one])
        try:
            a1 = optimize.newton(func_a, a0, maxiter = 500)
        except RuntimeError:
            i = max_iter
            break
        loop_end = (abs(rho1 - rho0) < tol) * (abs(a1 - a0) < tol)
        a0, rho0 = a1, rho1
        i += 1
    if i < max_iter: return a1, rho1
    else: 
        print "Failed to converge."
        return None, None
    
def yule_solver(ab):
    """Given abundance data, solve for MLE of original Yule distribution parameter rho
    
    Algorithm from Garcia 2011.
    
    """
    ab = check_for_support(ab, lower = 1)
    rho0 = np.mean(ab) / (np.mean(ab) - 1)
    tol = 1.490116e-08
    loop_end = False
    while not loop_end:
        rho1 = len(ab) / sum([sum([1 / (rho0 + j) for j in range(1, k+1)]) for k in ab])
        loop_end = (abs(rho1 - rho0) < tol)
        rho0 = rho1
    return rho1

def zipf_solver(ab):
    """Obtain the MLE parameter for a Zipf distribution with x_min = 1."""
    ab = check_for_support(ab, lower = 1)
    par0 = 1 + len(ab) / (sum(np.log(2 * np.array(ab))))
    def zipf_func(x):
        return -zipf_ll(ab, x)
    par = optimize.fmin(zipf_func, x0 = par0)[0]
    return par
    
def xsquare_pdf(x, dist, *pars):
    """Calculates the pdf for x, given the distribution of variable Y = sqrt(X) 
    
    and a given value x. 
    
    """
    x = np.array(x)
    return 1 / x ** 0.5 * dist.pdf(x ** 0.5, *pars) / 2 

def ysquareroot_pdf(y, dist, *pars):
    """Calculates the pdf for y, given the distribution of variable X = Y^2 and a given value y."""
    y = np.array(y)
    return 2 * dist.pdf(y ** 2, *pars) * y

def check_for_support(x, lower = 0, upper = np.inf, warning = True):
    """Check if x (list or array) contains values out of support [lower, upper]
    
    If it does, remove the values and optionally print out a warning.
    This function is used for solvers of distributions with support smaller than (-inf, inf).
    
    """
    if (min(x) < lower) or (max(x) > upper):
        if warning:
            print "Warning: Out-of-support values in the input are removed."
    x = np.array([element for element in x if lower <= element <= upper])
    return x
