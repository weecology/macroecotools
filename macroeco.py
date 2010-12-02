"""Module for conducting standard macroecological plots and analyses"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def get_rad_from_cdf(cdf, S):
    """Return a predicted rank-abundance distribution from a theoretical CDF
    
    Keyword arguments:
    cdf -- a function characterizing the theoretical cdf (likely from pylab)
    S -- the number of species for which the RAD should be predicted. Should
    match the number of species in the community if comparing to empirical data.
    
    Finds the predicted rank-abundance distribution that results from a
    theoretical cumulative distribution function, by rounding the value of the
    cdf evaluated at 1 / S * (Rank - 0.5) to the nearest integer
    
    """

def plot_rad(Ns):
    """Plot a rank-abundance distribution based on a vector of abundances"""
    Ns.sort(reverse=True)
    rank = range(1, len(Ns) + 1)
    plt.plot(rank, Ns, 'bo-')
    plt.xlabel('Rank')
    plt.ylabel('Abundance')
    plt.show()

def get_rad_data(Ns):
    """Provide ranks and relative abundances for a vector of abundances"""
    Ns = np.array(Ns)
    Ns_sorted = -1 * np.sort(-1 * Ns)
    relab_sorted = Ns_sorted / sum(Ns_sorted)
    rank = range(1, len(Ns) + 1)
    return (rank, relab_sorted)
    
def plot_multiple_rads(list_of_abund_vectors):
    """Plots multiple rank-abundance distributions on a single plot"""
    #TO DO:
    #  Allow this function to handle a single abundance vector
    #     Currently would treat each value as a full abundance vector
    #     Could then change this to plot_rads and get rid of plot_rad
    plt.figure()
    line_styles = ['bo-', 'ro-', 'ko-', 'go-', 'bx--', 'rx--', 'kx--', 'gx--']
    num_styles = len(line_styles)
    plt.hold(True)
    for (style, Ns) in enumerate(list_of_abund_vectors):
        (rank, relab) = get_rad_data(Ns)
        #Plot line rotating through line_styles and starting at the beginning
        #of line_styles again when all values have been used
        plt.semilogy(rank, relab, line_styles[style % num_styles])
    plt.hold(False)
    plt.xlabel('Rank')
    plt.ylabel('Abundance')
    plt.show()
    
def plot_SARs(list_of_A_and_S):
    """Plot multiple SARs on a single plot. 
    
    Input: a list of lists, each sublist contains one vector for S and one vector for A.
    Output: a graph with SARs plotted on log-log scale, with colors spanning the spectrum.
    
    """
    N = len(list_of_A_and_S)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    for i in range(len(list_of_A_and_S)):
        sublist = list_of_A_and_S[i]
        plt.loglog(sublist[0], sublist[1], color = RGB_tuples[i])
    plt.hold(False)
    plt.xlabel('Area')
    plt.ylabel('Richness')
    plt.show()
