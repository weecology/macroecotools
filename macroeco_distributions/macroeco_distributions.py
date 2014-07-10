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
    Log-likelihood of a geomtric distribution
    Log-likelihood of a negative binomial dstribution (truncated at 1)
    Log-likelihood of a discrete gamma distribution
    Log-likelihood of the generalized Yule distribution
    Log-likelihood of the original Yule-Simon distribution
    

"""

from __future__ import division
import sys
from math import factorial, floor, sqrt
from numpy import exp, histogram, log, matlib, sort, pi, std, mean
import numpy as np
from scipy import integrate, stats, optimize, special
from scipy.stats import rv_discrete, rv_continuous
from scipy.optimize import bisect
from scipy.integrate import quad

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
                    eq = lambda t: np.exp(t * x_i - np.exp(t) - (t - mu) ** 2 / 2 / (sigma ** 2))
                    term2a = integrate.quad(eq, -float('inf'), np.log(ub), full_output = 0, limit = 500)
                    #integrate higher end for accuracy and in case peak moves
                    term2b = integrate.quad(eq, np.log(ub), float('inf'), full_output = 0, limit = 500)
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
    
    def _cdf(self, x, mu, sigma, lower_trunc, approx_cut = 10):
        x = np.array(x)
        cdf = []
        for x_i in x:
            cdf.append(sum(self.pmf(range(int(x_i) + 1), mu[0], sigma[0], lower_trunc[0])))
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
        #return stats.logser.pmf(x, p) / stats.logser.cdf(upper_bound, p)
        if p[0] < 1:
            return stats.logser.pmf(x, p) / stats.logser.cdf(upper_bound, p)
        else:
            ivals = np.arange(1, upper_bound[0] + 1)
            normalization = sum(p[0] ** ivals / ivals)
            pmf = (p[0] ** x / x) / normalization
            return pmf

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
    
    def _argcheck(self, p, upper_bound):
        cond = (p > 0) & (upper_bound >= 1) 
        return cond

trunc_geom = trunc_geom_gen(name = 'trunc_geom', longname = 'Upper truncated geometric', 
                                  shapes = 'p, upper_bound')

class trunc_geom_with_zeros_gen(rv_discrete):
    """Upper truncated geometric distribution (with zeros)"""
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        pmf = (1 - p) ** x * p / (1 - (1 - p) ** (upper_bound + 1))
        return pmf
    
    def _cdf(self, x, p, upper_bound):
        x = np.array(x)
        cdf = (1 - (1 - p) ** (x + 1)) / (1 - (1 - p) ** (upper_bound + 1))
        return cdf
    
    def _ppf(self, cdf, p, upper_bound):
        cdf = np.array(cdf)
        x = np.log(1 - cdf * (1 - (1 - p) ** (upper_bound + 1))) /\
            np.log(1 - p) - 1
        return np.ceil(x)
    
    def _argcheck(self, p, upper_bound):
        cond = (p > 0) & (upper_bound >= 1) 
        return cond

trunc_geom_with_zeros = trunc_geom_with_zeros_gen(name = 'trunc_geom_with_zeros', 
                                                  longname = 'Upper truncated geometric with zeros', 
                                                  shapes = 'p, upper_bound')

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
    #purify abundance vector
    x = np.array(x)
    x = x[x > 0]
    x.sort()
    cts = histogram(x, bins = range(1, max(x) + 2))
    observed_abund_vals = cts[1][cts[0] != 0]
    counts = cts[0][cts[0] != 0]
    plik = pln.logpmf(observed_abund_vals, mu, sigma, lower_trunc)
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
    """Log-likelihood of the Weibull distribution lower truncated at lower_bound"""
    return sum(trunc_weibull.logpdf(x, k, lmd, lower_bound))

def disunif_ll(ab, low, high):
    """Log-likelihood of a discrete uniform distribution with bounds [low, high]"""
    n = len(ab)
    return n * log(1 / (high - low + 1))

def geom_ll(ab, p):
    """Log-likelihood of a geometric distribution"""
    return sum(stats.geom.logpmf(ab, p))

def negbin_ll(ab, n, p):
    """Log-likelihood of a negative binomial dstribution (truncated at 1)"""
    return sum(stats.nbinom.logpmf(ab, n, p)) - len(ab) * log(1 - stats.nbinom.pmf(0, n, p))

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
        return -pln_ll(ab, x[0], x[1], lower_trunc)
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

def trunc_geom_solver(ab, upper_bound):
    """Given abundance data, solve for MLE of upper-truncated geometric distribution parameter p"""
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -10
    S = len(ab)
    N = sum(ab)
    y = lambda x: (N * (S-1) * (1-x) + S) * x ** upper_bound + N * (1-x) - S
    one_minus_p = optimize.bisect(y, 1 - S/N - DIST_FROM_BOUND, BOUNDS[1] - DIST_FROM_BOUND, 
                                  xtol = 1.490116e-16)
    return 1 - one_minus_p

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
    """Given abundance data, solve for MLE of generalized Yule distribution parameters a and b(rho)
    
    Algorithm extended from Garcia 2011.
    
    
    """
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
        a1 = optimize.newton(func_a, a0, maxiter = 500)
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
    rho0 = np.mean(ab) / (np.mean(ab) - 1)
    tol = 1.490116e-08
    loop_end = False
    while not loop_end:
        rho1 = len(ab) / sum([sum([1 / (rho0 + j) for j in range(1, k+1)]) for k in ab])
        loop_end = (abs(rho1 - rho0) < tol)
        rho0 = rho1
    return rho1

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
