"""Module for conducting standard macroecological plots and analyses"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from numpy import log10

def AICc(k, L, n):
    """Computes the corrected Akaike Information Criterion. 
    
    Keyword arguments:
    L  --  log likelihood value of given distribution.
    k  --  number of fitted parameters.
    n  --  number of observations.
       
    """
    AICc = 2 * k - 2 * L + 2 * k * (k + 1) / (n - k - 1)
    return AICc

def aic_weight(AICc_dist1, AICc_dist2, n, cutoff = 4):
    """Computes Akaike weight for one model relative to another
    
    Based on information from Burnham and Anderson (2002).
    
    Keyword arguments:
    AICc_dist1  --  AICc for primary model of interest
    AICc_dist2  --  AICc for alternative model
    n           --  number of observations.
    cutoff      --  minimum number of observations required to generate a weight.
    
    """
    if n < cutoff:
        weight = None
        
    else:
        AICc_min = min(AICc_dist1, AICc_dist2)
        weight_dist1 = np.exp(-(AICc_dist1 - AICc_min) / 2)
        weight_dist2 = np.exp(-(AICc_dist2 - AICc_min) / 2)
        weight = weight_dist1 / (weight_dist1 + weight_dist2)  
      
    return(weight)   

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
def get_emp_cdf(dat):
    """Compute the empirical cdf given a list or an array"""
    dat = np.array(dat)
    emp_cdf = []
    for point in dat:
        point_cdf = len(dat[dat < point]) / len(dat)
        emp_cdf.append(point_cdf)
    return np.array(emp_cdf)

def plot_rad(Ns):
    """Plot a rank-abundance distribution based on a vector of abundances"""
    Ns.sort(reverse=True)
    rank = range(1, len(Ns) + 1)
    plt.plot(rank, Ns, 'bo-')
    plt.xlabel('Rank')
    plt.ylabel('Abundance')

def get_rad_data(Ns):
    """Provide ranks and relative abundances for a vector of abundances"""
    Ns = np.array(Ns)
    Ns_sorted = -1 * np.sort(-1 * Ns)
    relab_sorted = Ns_sorted / sum(Ns_sorted)
    rank = range(1, len(Ns) + 1)
    return (rank, relab_sorted)

def preston_sad(abund_vector, b=None, normalized = 'no'):
    """Plot histogram of species abundances on a log2 scale"""
    if b == None:
        q = np.exp2(list(range(0, 25)))    
        b = q [(q <= max(abund_vector)*2)]
    
    if normalized == 'no':
        hist_ab = np.histogram(abund_vector, bins = b)
    if normalized == 'yes':
        hist_ab_norm = np.histogram(abund_vector, bins = b)
        hist_ab_norm1 = hist_ab_norm[0]/(b[0:len(hist_ab_norm[0])])
        hist_ab_norm2 = hist_ab_norm[1][0:len(hist_ab_norm[0])]
        hist_ab = (hist_ab_norm1, hist_ab_norm2)
    return hist_ab
    
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
    
def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    raw_data = np.array([x, y])
    x = np.array(x)
    y = np.array(y)
    raw_data = raw_data.transpose()
    
    # Get unique data points by adding each pair of points to a set
    unique_points = set()
    for xval, yval in raw_data:
        unique_points.add((xval, yval))
    
    count_data = []
    for a, b in unique_points:
        if logscale == 1:
            num_neighbors = len(x[((log10(x) - log10(a)) ** 2 +
                                   (log10(y) - log10(b)) ** 2) <= log10(radius) ** 2])
        else:        
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data

def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density
    
    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)
    
    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.

    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))
    
    if plot_obj == None:
        plot_obj = plt.axes()
        
    if loglog == 1:
        plot_obj.set_xscale('log')
        plot_obj.set_yscale('log')
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                         c = np.sqrt(sorted_plot_data[:, 2]), edgecolors='none')
        plot_obj.set_xlim(min(x) * 0.5, max(x) * 2)
        plot_obj.set_ylim(min(y) * 0.5, max(y) * 2)
    else:
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                    c = log10(sorted_plot_data[:, 2]), edgecolors='none')
    return plot_obj

def e_var(abundance_data):
    """Calculate Smith and Wilson's (1996; Oikos 76:70-82) evenness index (Evar)
    
    Input:
    abundance_data = list of abundance fo all species in a community
    
    """
    S = len(abundance_data)
    ln_nj_over_S=[]
    for i in range(0, S):
        v1 = (np.log(abundance_data[i]))/S
        ln_nj_over_S.append(v1)     
    
    ln_ni_minus_above=[]
    for i in range(0, S):
        v2 = ((np.log(abundance_data[i])) - sum(ln_nj_over_S)) ** 2
        ln_ni_minus_above.append(v2)
        
    return(1 - ((2 / np.pi) * np.arctan(sum(ln_ni_minus_above) / S)))

def obs_pred_rsquare(obs, pred):
    """Determines the prop of variability in a data set accounted for by a model
    
    In other words, this determines the proportion of variation explained by
    the 1:1 line in an observed-predicted plot.
    
    """
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

def obs_pred_mse(obs, pred):
    """Calculate the mean squared error of the prediction given observation."""
    
    return sum((obs - pred) ** 2) / len(obs)

def comp_ed (spdata1,abdata1,spdata2,abdata2):
    """Calculate the compositional Euclidean Distance between two sites
    
    Ref: Thibault KM, White EP, Ernest SKM. 2004. Temporal dynamics in the
    structure and composition of a desert rodent community. Ecology. 85:2649-2655.
    
    """     
    abdata1 = (abdata1 * 1.0) / sum(abdata1)
    abdata2 = (abdata2 * 1.0) / sum(abdata2)
    intersect12 = set(spdata1).intersection(spdata2)
    setdiff12 = np.setdiff1d(spdata1,spdata2)
    setdiff21 = np.setdiff1d(spdata2,spdata1)
    relab1 = np.concatenate(((abdata1[np.setmember1d(spdata1,list(intersect12)) == 1]),
                             abdata1[np.setmember1d(spdata1,setdiff12)], 
                             np.zeros(len(setdiff21))))
    relab2 = np.concatenate((abdata2[np.setmember1d(spdata2,list(intersect12)) == 1],
                              np.zeros(len(setdiff12)),
                              abdata2[np.setmember1d(spdata2,setdiff21)]))
    return np.sqrt(sum((relab1 - relab2) ** 2))

def calc_comp_eds(ifile, fout):
    """Calculate Euclidean distances in species composition across sites.
    
    Determines the Euclidean distances among all possible pairs of sites and
    saves the results to a file

    Inputs:
    ifile -- ifile = np.genfromtxt(input_filename, dtype = "S15,S15,i8", 
                   names = ['site','species','ab'], delimiter = ",")
    fout -- fout = csv.writer(open(output_filename,'ab'))
    
    """
    #TODO - Remove reliance on on names of columns in input
    #       Possibly move to 3 1-D arrays for input rather than the 2-D with 3 columns
    #       Return result rather than writing to file
    
    usites = np.sort(list(set(ifile["site"]))) 

    for i in range (0, len(usites)-1):       
        spdata1 = ifile["species"][ifile["site"] == usites[i]]
        abdata1 = ifile["ab"][ifile["site"] == usites[i]]
        
        for a in range (i+1,len(usites)):  
            spdata2 = ifile["species"][ifile["site"] == usites[a]]
            abdata2 = ifile["ab"][ifile["site"] == usites[a]]   
            
            if len(spdata1) > cutoff and len(spdata2) > cutoff:
                ed = comp_ed (spdata1,abdata1,spdata2,abdata2)
                results = np.column_stack((usites[i], usites[a], ed))
                fout.writerows(results)