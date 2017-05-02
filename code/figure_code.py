#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt

from sys import argv

import numpy as np
import pandas as pd
import scipy.stats as ss
import pymc

from collections import defaultdict
from scipy.signal import savgol_filter
from scipy.optimize import brute
from scipy.stats import beta
from sklearn.metrics import auc as auc_calc

from snp_method_simulation import simulate_snp_sample

#Define input/output directories in relation to code directory
ANALYSES_DIR = "../simulation_output/"
FIGURES_DIR = "../manuscript_figure_panels/"

FIG_DPI = 300 #Set DPI on PNGs produced by code.

#If run as figure_code.py -e, reproduce the manuscript figs exactly.
if argv[-1] == "-e":
    np.random.seed(1000)

def main():
    """
    Plot the individual panels of each of the figures in the main manuscript.
    """

    plot_figure_1_panels()

    plot_figure_2_panels()

    plot_figure_3_panels()

    plot_figure_4_panels()

def plot_figure_1_panels():
    """
    Plot panels used in Figure 1.
    """

    ###
    # Top - Overview of the WGS method.
    ###

    reads_per_sample = 16e6
    ff = 0.2
    genome_size = 3e9
    bin_size = 50e3
    depth_per_bin = reads_per_sample / (genome_size / bin_size)
    chroms = range(1, 23) + ["X", "Y"]

    chrom_lengths = {1:249250621,
                     2:243199373,
                     3:198022430,
                     4:191154276,
                     5:180915260,
                     6:171115067,
                     7:159138663,
                     8:146364022,
                     9:141213431,
                     10:135534747,
                     11:135006516,
                     12:133851895,
                     13:115169878,
                     14:107349540,
                     15:102531392,
                     16:90354753,
                     17:81195210,
                     18:78077248,
                     19:59128983,
                     20:63025520,
                     21:48129895,
                     22:51304566,
                     "X":155270560,
                     "Y":59373566
                    }


    chrom_bins = defaultdict(list)
    for x in chrom_lengths:
        num_bins = int(chrom_lengths[x]/bin_size * 0.9)
        if x == 21:
            chrom_bins[x] = np.random.poisson((1 + ff/2.) * depth_per_bin, num_bins)
        elif x != "X" and x != "Y":
            chrom_bins[x] = np.random.poisson(depth_per_bin, num_bins)
        elif x == "X":
            chrom_bins[x] = np.random.poisson((1 - ff/2.) * depth_per_bin, num_bins)
        else:
            chrom_bins[x] = np.random.poisson(ff/2. * depth_per_bin, num_bins)

    y_axis = []
    cols = []
    flip = True
    tick_pos = []
    last_pos = 0
    for chrom in chroms:
        y_axis.extend(chrom_bins[chrom])
        tick_pos.append(len(chrom_bins[chrom])/2 + last_pos)
        y_axis.extend([-20] * 1000)
        last_pos = len(y_axis)

        if chrom == 21:
            cols.extend(["red"] * (len(chrom_bins[chrom]) + 1000))
            if flip:
                flip = False
            else:
                flip = True
        elif flip:
            cols.extend(["darkgray"] * (len(chrom_bins[chrom]) + 1000))
            flip = False
        else:
            cols.extend(["lightgray"] * (len(chrom_bins[chrom]) + 1000))
            flip = True

    x_axis = range(len(y_axis))

    figure_options = {'figsize':(31, 8)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 2)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 40
    label_font_size = 55
    box_width = 4
    y_min = 1
    y_max = 400

    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])

    ax[0].scatter(x_axis, y_axis, color=cols, s=1, alpha=1)
    ax[0].set_xlim([-1500, 80500])
    x_ticks = tick_pos
    ax[0].set_xticks(x_ticks)
    ax[0].xaxis.set_tick_params(width=box_width, length=12)
    chroms2 = [chroms[x] if x % 2 == 0 else "" for x in range(len(chroms))]
    chroms2[-1] = "Y"
    ax[0].xaxis.tick_top()
    ax[0].set_xticklabels(chroms2, fontsize=tick_font_size)
    ax[0].set_xlabel('Chromosomes', fontsize=label_font_size)
    ax[0].xaxis.set_label_position('top')

    y_ticks = []
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(["" for x in range(len(y_ticks))], fontsize=tick_font_size)
    ax[0].yaxis.set_tick_params(width=box_width, length=12)
    ax[0].set_ylabel("Counts per bin", fontsize=label_font_size)
    [i.set_linewidth(box_width) for i in ax[0].spines.itervalues()]
    ax[0].set_ylim([y_min, y_max])

    y_axis = range(y_min, y_max)
    normal = [ss.poisson.pmf(x, depth_per_bin) for x in y_axis]
    aneuploid = [ss.poisson.pmf(x, (1+ff/2.) * depth_per_bin) for x in y_axis]

    normal[0], normal[-1] = 0, 0
    aneuploid[0], aneuploid[-1] = 0, 0

    ax[1].fill_between(normal, y_axis, color="darkgrey", alpha=0.5)
    ax[1].fill_between(aneuploid, y_axis, color="red", alpha=0.5)
    ax[1].set_ylim([y_min, y_max])
    ax[1].set_xlim([0, 0.05])
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    [i.set_linewidth(3) for i in ax[1].spines.itervalues()]
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(FIGURES_DIR + "Fig_1_top.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    ###
    # Bottom left - Overview of the SNP method on disomic sample.
    ###

    ff = 0.2               #Fetal fraction
    num_snps = 3348
    mean_counts = 859      #Mean counts per SNP
    bbdisp = 1000          #Beta dispersion factor
    nbdisp = 0.0005        #Negative binomial dispersion parameter
    H = (1, 1, None)         #Disomic sample

    #Simulate points to plot on figure.
    df_points = simulate_snp_sample(ff=ff,
                                    H=H,
                                    num_snps=num_snps,
                                    mean_counts=mean_counts,
                                    bbdisp=bbdisp,
                                    nbdisp=nbdisp)

    #Repeat simulation on much larger number of points to calculate joint
    #probability distributions
    df_dist = simulate_snp_sample(ff=ff,
                                  H=H,
                                  num_snps=100000,
                                  mean_counts=mean_counts,
                                  bbdisp=bbdisp,
                                  nbdisp=nbdisp)

    generate_snp_method_fig1_plot(points=df_points,
                                  distros=df_dist,
                                  outfile="Fig_1_bot_left.png",
                                  ylab=True)

    ###
    # Bottom right - Overview of the SNP method on paternal M2 trisomic sample.
    ###


    H = (1, 2, 2)

    df_points = simulate_snp_sample(ff=ff,
                                    H=H,
                                    num_snps=num_snps,
                                    mean_counts=mean_counts,
                                    bbdisp=bbdisp,
                                    nbdisp=nbdisp)

    df_dist = simulate_snp_sample(ff=ff,
                                  H=H,
                                  num_snps=100000,
                                  mean_counts=mean_counts,
                                  bbdisp=bbdisp,
                                  nbdisp=nbdisp)

    generate_snp_method_fig1_plot(points=df_points,
                                  distros=df_dist,
                                  outfile="Fig_1_bot_right.png")

def generate_snp_method_fig1_plot(points, distros, outfile, bbdisp=1000, nbdisp=0.0005, ylab=False):
    """
    Generate PNG format plot of fraction A allele reads illustrating SNP
    method simulated sample as formatted in Figure 1.

    Args:
        points (dataframe): Dataframe of SNPs to plot from simulate_snp_sample()
        distros (dataframe): Dataframe of SNPs to fit distributions, should
                             contain >25,000 SNPs to fit smooth distributions
        outfile (str): Location of the output PNG file
        ylab (bool): If True, add label to y-axis of plot

    Returns:
        None

    Raises:
        AssertionError: If points is not a pandas dataframe
        AssertionError: If distros is not a pandas dataframe
    """

    assert isinstance(points, pd.core.frame.DataFrame), "points must be pandas dataframe"
    assert isinstance(distros, pd.core.frame.DataFrame), "distros must be pandas dataframe"

    #Generate distributions of values
    c_dict = defaultdict(list)
    c_fit = {}

    for i,r in distros.iterrows():
        if r["TOT_COUNTS"] > 0:
            c_dict[(r["MAT_GENO"], r["FET_GENO"])].append(r["A_COUNT"]/float(r["TOT_COUNTS"]))

    for x in c_dict:
        if x[0] == ("BA"):
            key = (("AB"), x[1])
        else:
            key = x

        if (key != ("AA","AA") and
            key != ("BB","BB") and
            key != ("AA","AAA") and
            key != ("BB","BBB")):

            p = np.mean(c_dict[x])
            t = generate_bb_pdf(p=p, bbdisp=bbdisp, nbdisp=nbdisp)
            c_fit[key] = savgol_filter(t, 11, 2)

    for x in c_fit:
        c_fit[x][0] = 0
        c_fit[x][-1] = 0

    figure_options = {'figsize':(8, 8)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 2)
    fig, ax = plt.subplots(panels[0], panels[1])

    #Need to remove SNPs with zero counts
    points = points[points["TOT_COUNTS"] > 0]

    x_vals = range(len(points))
    cols = get_colors(points["MAT_GENO"])

    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])

    ax[0].scatter(x_vals, np.true_divide(points["A_COUNT"], points["TOT_COUNTS"]), color=cols, s=50, alpha=0.5)
    ax[0].set_xlim([0, len(x_vals)])
    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].set_ylim([0, 1])
    y_ticks = np.arange(0, 1.01, 0.1)
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels([y_ticks[x] if x % 2 == 0 else "" for x in range(len(y_ticks))], fontsize=36)
    if ylab:
        ax[0].set_ylabel("Fraction of A allele reads", fontsize=40)
    else:
        ax[0].set_ylabel("Fraction of A allele reads", fontsize=40, color="white")
    ax[0].yaxis.grid(b=False, which='major', color='grey', linestyle='-', lw=2)
    ax[0].set_axisbelow(True)

    y_axis = np.arange(0, 1.00001, 0.01)
    for x in c_fit:
        if (x != ("AA","AA") and
            x != ("BB","BB") and
            x != ("AA","AAA") and
            x != ("BB","BBB")):

            if x[0] == "AA":
                col = "red"
            elif x[0] == "AB":
                col = "green"
            elif x[0] == "BA":
                col = "green"
            elif x[0] == "BB":
                col = "blue"
            ax[1].fill_between(c_fit[x], -10, y_axis, color=col, alpha=0.5)
    ax[1].set_ylim([0.01, 0.99])
    ax[1].set_xlim(left=0)
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(FIGURES_DIR + outfile,
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

def generate_bb_pdf(p, bbdisp=1000, nbdisp=0.0005, depth=859):
    """
    Generate the joint negative binomial/Poisson depth sampling +
    betabinomial/binomial allele sampling pdf for SNP method plotting in 0.01%
    allele fraction increments.

    Args:
        p (float): Proportion A allele reads
        bbdisp (float): beta-binomial dispersion parameter in allele counts
        nbdisp (float): negative-binomial dispersion parameter in read counts
        depth (int): Mean counts per SNP

    Returns:
        List containing joint pdf in 0.01% allele fraction increments from 0
        to 1

    Raises:
        AssertionError: if p <= 0 or >= 1
        AssertionError: if bbdisp <= 1 or not 'binomial'
        AssertionError: if nbdisp < 0 or > 1
    """

    assert 0 < p < 1, "Value of p must be > 0 and < 1"
    assert bbdisp >= 1 or bbdisp == "binomial", ("Value of bbdisp must be > 1 "
                                                 "or 'binomial'")
    assert 0 < nbdisp <= 1, "Value of nbdisp must be > 0 and < 1"

    flip = False
    if bbdisp != "binomial" and p < 0.5:
        flip = True
        p = 1 - p

    #Enumerate allele fraction possibilities
    afs = defaultdict(list)
    max_d = depth * 2
    for x, y in ((x, y) for x in range(1, max_d + 1) for y in range(x + 1)):
        af = np.round(y / float(x), 2)
        afs[af].append((y, x))


    #Calculate probability of observing read depth given p
    if nbdisp == 1: #If nbdisp == 1 collapse to Poisson
        probs = ss.poisson.pmf(range(1, max_d + 1), int(depth * p + 0.5))
        d_prob = {x:y for x, y in zip(range(1, max_d + 1), probs)}
    else:
        a_nb = depth * nbdisp /float(1 - nbdisp)
        b_nb = nbdisp
        probs = ss.nbinom.pmf(range(1, max_d + 1), a_nb, b_nb)
        d_prob = {x:y for x, y in zip(range(1, max_d + 1), probs)}


    #Calculate probability of observing each allele fraction
    #Given the beta-binomial or binomial parameters
    pdf = []
    rng = np.arange(0, 1.01, 0.01)

    if bbdisp == "binomial":
        for x in rng:
            x = np.round(x, 2)
            a_vals = [val[0] for val in afs[x]]
            depth_vals = [val[1] for val in afs[x]]
            paf = ss.binom.pmf(a_vals, depth_vals, p)
            d_probs = [d_prob[d] for d in depth_vals]
            pdf.append(np.dot(d_probs, paf))
    else:
        a_bb = bbdisp
        b_bb = (bbdisp / p) - bbdisp
        for x in rng:
            x = np.round(x, 2)
            paf = []
            for a, depth in afs[x]:
                 paf.append(np.exp(pymc.betabin_like(x=a,
                                                     alpha=a_bb,
                                                     beta=b_bb,
                                                     n=depth)))
            d_probs = [d_prob[val[1]] for val in afs[x]]
            pdf.append(np.dot(d_probs, paf))

    if flip:
        pdf = pdf[::-1]

    return pdf

def get_colors(maternal_haplotypes):
    """Color maternal haplotypes AA, red; AB, green; BB, blue.
    """
    cols = []
    col_dict = {"AA":"red",
                "AB":"green",
                "BA":"green",
                "BB":"blue"}

    for m in maternal_haplotypes:
        cols.append(col_dict[m])

    return cols

def plot_figure_2_panels():
    """
    Plot panels used in Figure 2.
    """

    ###
    # Panel B - Distribution of SNP allele counts under different origins of
    #           nondisjunction.
    ###

    #Diploid
    ff = 0.1               # Fetal fraction
    num_snps=3348          # Number of SNPs per chromosome
    mean_counts = 859      # Mean counts per SNP
    bbdisp = 1000          # Beta dispersion factor
    nbdisp = 0.0005        # Negative binomial dispersion parameter

    ploidies = {
        "01_disomy":(1, 1, None),
        "02_maternal_m1":(2, 1, 1),
        "03_maternal_m2":(2, 1, 2),
        "04_paternal_m1":(1, 2, 1),
        "05_paternal_m2":(1, 2, 2)
    }

    for key, val in ploidies.iteritems():

        ylab = False
        if key == "01_disomy":
            ylab = True

        df_points = simulate_snp_sample(ff=ff,
                                        H=val,
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=nbdisp)

        df_dist = simulate_snp_sample(ff=ff,
                                      H=val,
                                      num_snps=100000,
                                      mean_counts=mean_counts,
                                      bbdisp=bbdisp,
                                      nbdisp=nbdisp)

        generate_snp_method_fig2_plot(points=df_points,
                                      distros=df_dist,
                                      outfile="Fig_2B_{}.png".format(key),
                                      ylab=ylab)

    ###
    # Panel C - Distribution of WGS bin counts under different origins of
    #           nondisjunction.
    ###

    #Disomic
    generate_fig2_wgs_bin_count_plot(outfile="Fig_2C_01_disomic.png",
                                     trisomic=False,
                                     ylab=True)

    #All trisomies
    generate_fig2_wgs_bin_count_plot(outfile="Fig_2C_others_trisomic.png",
                                     trisomic=True)

def generate_snp_method_fig2_plot(points, distros, outfile, bbdisp=1000, nbdisp=0.0005, ylab=False):
    """
    Generate PNG format plot of fraction A allele reads illustrating SNP
    method simulated sample as formatted in Figure 2.

    Args:
        points (dataframe): Dataframe of SNPs to plot from simulate_snp_sample()
        distros (dataframe): Dataframe of SNPs to fit distributions, should
                             contain >25,000 SNPs to fit smooth distributions
        outfile (str): Location of the output PNG file
        ylab (bool): If True, add label to y-axis of plot

    Returns:
        None

    Raises:
        AssertionError: If points is not a pandas dataframe
        AssertionError: If distros is not a pandas dataframe
    """

    assert isinstance(points, pd.core.frame.DataFrame), "points must be pandas dataframe"
    assert isinstance(distros, pd.core.frame.DataFrame), "distros must be pandas dataframe"

    #Generate distributions of values
    c_dict = defaultdict(list)
    c_fit = {}

    for i,r in distros.iterrows():
        if r["TOT_COUNTS"] > 0:
            c_dict[(r["MAT_GENO"], r["FET_GENO"])].append(r["A_COUNT"]/float(r["TOT_COUNTS"]))

    for x in c_dict:
        if x[0] == ("BA"):
            key = (("AB"),x[1])
        else:
            key = x

        if (key != ("AA","AA") and
            key != ("BB","BB") and
            key != ("AA","AAA") and
            key != ("BB","BBB")):

            p = np.mean(c_dict[x])
            t = generate_bb_pdf(p=p, bbdisp=bbdisp, nbdisp=nbdisp)
            c_fit[key] = savgol_filter(t, 11, 2)

    for x in c_fit:
        c_fit[x][0] = 0
        c_fit[x][-1] = 0

    figure_options = {'figsize':(5, 8)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 2)
    fig, ax = plt.subplots(panels[0], panels[1])

    #Need to remove SNPs with zero counts
    points = points[points["TOT_COUNTS"] > 0]

    x_vals = range(len(points))
    cols = get_colors(points["MAT_GENO"])

    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])

    ax[0].scatter(x_vals, np.true_divide(points["A_COUNT"],points["TOT_COUNTS"]), color=cols, s=50, alpha=0.5)
    ax[0].set_xlim([0, len(x_vals)])
    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].set_ylim([0, 1])
    y_ticks = np.arange(0, 1.01, 0.1)
    ax[0].set_yticks(y_ticks)
    if ylab:
        ax[0].set_ylabel("Fraction of A allele reads", fontsize=40)
        ax[0].set_yticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1], fontsize=36)
    else:
        ax[0].set_ylabel("Fraction of A allele reads", fontsize=40, color="white")
        ax[0].set_yticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1], fontsize=36, color="white")
    ax[0].yaxis.grid(b=False, which='major', color='grey', linestyle='-')
    [i.set_linewidth(2) for i in ax[0].spines.itervalues()]
    for y in ax[0].get_yaxis().majorTicks:
        y.set_pad(10)

    y_axis = np.arange(0, 1.00001, 0.01)
    for x in c_fit:
        if (x != ("AA","AA") and
            x != ("BB","BB") and
            x != ("AA","AAA") and
            x != ("BB","BBB")):

            if x[0] == "AA":
                col = "red"
            elif x[0] == "AB":
                col = "green"
            elif x[0] == "BA":
                col = "green"
            elif x[0] == "BB":
                col = "blue"
            ax[1].fill_between(c_fit[x], -10, y_axis, color=col, alpha=0.5)
    ax[1].set_ylim([0.01, 0.99])
    ax[1].set_xlim([0, 0.15])
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(FIGURES_DIR + outfile,
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

def generate_fig2_wgs_bin_count_plot(outfile, ylab=False, trisomic=False):
    """
    Generate plot of simulated WGS method bins under diploid or triploid
    scenarios for chromosome 21 along with the distribution from which those
    plots were drawn as formatted in Figure 2.

    Args:
        outfile (str): The location of the output PNG file
        ylab (bool): If true, add a label to the y axis of the plot
        trisomic (bool): If True, plot trisomic bins in red instead of disomic
                         bins in grey

    Returns:
        None
    """

    reads_per_sample = 16e6
    ff = 0.1
    genome_size = 3e9
    bin_size = 50e3
    depth_per_bin = reads_per_sample / (genome_size / bin_size)
    chroms = ["chr21"]

    chrom_lengths = {"chr21":48129895}

    chrom_bins = defaultdict(list)
    for x in chrom_lengths:
        num_bins = int(chrom_lengths[x]/bin_size * 0.9)
        if trisomic:
            chrom_bins[x] = np.random.poisson((1+ff/2.) * depth_per_bin, num_bins)
        else:
            chrom_bins[x] = np.random.poisson(depth_per_bin, num_bins)

    #Munge into plot
    y_axis = []
    cols = []
    tick_pos = []
    last_pos = 0
    for chrom in chroms:
        y_axis.extend(chrom_bins[chrom])
        tick_pos.append(len(chrom_bins[chrom])/2 + last_pos)
        y_axis.extend([-20] * 1000)
        last_pos = len(y_axis)

        if trisomic:
            cols.extend(["red"] * (len(chrom_bins[chrom]) + 1000))
        else:
            cols.extend(["darkgray"] * (len(chrom_bins[chrom]) + 1000))

    x_axis = range(len(y_axis))

    figure_options = {'figsize':(5,8)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1,2)
    fig,ax = plt.subplots(panels[0],panels[1])

    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 2])
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])

    y_max = 360
    y_min = 190
    box_width = 4

    ax[0].scatter(x_axis, y_axis, color=cols, s=5, alpha=1)
    ax[0].set_xlim([-500,1350])
    ax[0].set_ylim([y_min,y_max])
    ax[0].yaxis.set_major_locator(plt.NullLocator())
    x_ticks = tick_pos
    ax[0].set_xticks(x_ticks)
    ax[0].xaxis.set_tick_params(width=box_width, length=12)

    ax[0].set_xticklabels([""], fontsize=40, rotation=45)
    if ylab:
        ax[0].set_ylabel("Counts per bin", fontsize=40)
    else:
        ax[0].set_ylabel("Counts per bin", fontsize=40, color="white")
    [i.set_linewidth(box_width) for i in ax[0].spines.itervalues()]
    ax[0].get_xaxis().tick_bottom()

    y_axis = range(y_min,y_max)
    if trisomic:
        distro = [ss.poisson.pmf(x,(1+ff/2.)*depth_per_bin) for x in y_axis]
        distro[0],distro[-1] = 0,0
        ax[1].fill_between(distro, y_axis, color="red", alpha=0.5)
    else:
        distro = [ss.poisson.pmf(x,depth_per_bin) for x in y_axis]
        distro[0],distro[-1] = 0,0
        ax[1].fill_between(distro, y_axis, color="darkgrey", alpha=0.5)

    ax[1].set_ylim([y_min,y_max])
    ax[1].set_xlim([0,0.05])
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    [i.set_linewidth(box_width) for i in ax[1].spines.itervalues()]
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(FIGURES_DIR + outfile,
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

def plot_figure_3_panels():
    """
    Plot panels used in Figure 3.
    """

    ###
    # Panel A - AUC as a function of fetal fraction for all origins of trisomy
    #           under the SNP method aggregated as a function of their
    #           prevalence across fetal fractions 0.1 - 4%.
    ###

    perms = 1000

    auc_df = pd.DataFrame(columns=["FF", "MEAN_AUC", "LOW", "HIGH"])

    for ff in np.arange(0.001, 0.041, 0.001):

        auc = [] #Store permutation-specific AUCs

        #Create df of LORs
        df_all = pd.DataFrame()

        for genotype in ["d", "m1", "m2", "p1", "p2"]:

            df = pd.read_csv(ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005/snp_{g}_ff{ff:.03f}_10000.csv".format(g=genotype, ff=ff), header=0)
            df_all[genotype] = df["LOR"]

        for p in range(perms):

            d = np.sort(np.random.choice(df_all["d"], size=1000, replace=False))
            a = np.sort(np.concatenate([np.random.choice(df_all["m1"], size=700, replace=False),
                                        np.random.choice(df_all["m2"], size=200, replace=False),
                                        np.random.choice(df_all["p1"], size=30, replace=False),
                                        np.random.choice(df_all["p2"], size=70, replace=False)]))

            tp, fp = [], []
            for t in np.sort(np.concatenate([a, d])):
                tp.append(np.sum(a <= t)/1000.)
                fp.append(np.sum(d < t)/1000.)
            tp[0], tp[-1] = 0, 1
            fp[0], fp[-1] = 0, 1
            auc.append(auc_calc(fp, tp))


        auc_df.loc[len(auc_df)] = [ff,
                                   np.mean(auc),
                                   np.percentile(auc, 2.5),
                                   np.percentile(auc, 97.5)]


    plt.rc('figure', figsize=(10, 10))  #Change plot defaults
    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 40
    label_font_size = 40
    box_width = 4

    auc_snp_df = auc_df[auc_df["FF"] <= 0.04]

    ax.plot(auc_snp_df["FF"], savgol_filter(auc_snp_df["MEAN_AUC"], 7, 2), lw=5, color="red", alpha=1, label="Aggregate of\nnondisjunctions")
    ax.fill_between(auc_snp_df["FF"], savgol_filter(auc_snp_df['LOW'],7,2), savgol_filter(auc_snp_df['HIGH'], 7, 2), color="red", alpha=0.2)

    ax.set_ylabel('Area under the curve (AUC)', fontsize=label_font_size)
    ax.set_xlabel('Fetal fraction', fontsize=label_font_size)
    ax.set_xticks(np.arange(0, 0.041, 0.01))
    ax.set_xticklabels(["0", "1%", "2%", "3%", "4%"])
    ax.set_xlim(-0.001, 0.042)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylim(0.47, 1.03)

    legend = ax.legend(loc=4, fontsize=32, frameon=False)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(7)


    for tl in ax.get_xticklabels():
        tl.set_fontsize(tick_font_size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)
    [i.set_linewidth(3) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major')
    ax.tick_params('x', length=10, width=box_width, which='major')

    for x in ax.get_xaxis().majorTicks:
        x.set_pad(15)
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(15)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]

    fig.savefig(FIGURES_DIR + "Fig_3A_SNP_AUCs_vs_FF.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    ###
    # Panel B - AUC as a function of fetal fraction for T21, T18, and T13 for
    #           the WGS method across fetal fractions 0.1 - 4%.
    ###

    perms = 1000
    auc_wgs_df = pd.DataFrame(columns=["FF",
                                       "CHR13_MEAN_AUC",
                                       "CHR13_LOW",
                                       "CHR13_HIGH",
                                       "CHR18_MEAN_AUC",
                                       "CHR18_LOW",
                                       "CHR18_HIGH",
                                       "CHR21_MEAN_AUC",
                                       "CHR21_LOW",
                                       "CHR21_HIGH"])

    df = pd.read_csv(ANALYSES_DIR + "wgs_method_10000affected_for_AUCs.csv", header=0)

    ff = 0
    for i in range(0, len(df.columns), 6):

        ff += 0.001

        auc_13, auc_18, auc_21 = [], [], []

        for p in range(perms):
            d13 =  np.random.choice(df.iloc[:,i], size=1000, replace=False)
            a13 =  np.random.choice(df.iloc[:,i + 1], size=1000, replace=False)

            tp13, fp13 = [], []
            for t in np.sort(np.concatenate([a13, d13]))[::-1]:
                tp13.append(np.sum(a13 >= t)/1000.)
                fp13.append(np.sum(d13 >= t)/1000.)
            tp13[0], tp13[-1] = 0, 1
            fp13[0], fp13[-1] = 0, 1
            auc_13.append(auc_calc(fp13, tp13))

            d18 =  np.random.choice(df.iloc[:,i + 2], size=1000, replace=False)
            a18 =  np.random.choice(df.iloc[:,i + 3], size=1000, replace=False)

            tp18, fp18 = [], []
            for t in np.sort(np.concatenate([a18,d18]))[::-1]:
                tp18.append(np.sum(a18 >= t)/1000.)
                fp18.append(np.sum(d18 >= t)/1000.)
            tp18[0], tp18[-1] = 0, 1
            fp18[0], fp18[-1] = 0, 1
            auc_18.append(auc_calc(fp18, tp18))

            d21 =  np.random.choice(df.iloc[:,i + 4], size=1000, replace=False)
            a21 =  np.random.choice(df.iloc[:,i + 5], size=1000, replace=False)

            tp21, fp21 = [], []
            for t in np.sort(np.concatenate([a21, d21]))[::-1]:
                tp21.append(np.sum(a21 >= t)/1000.)
                fp21.append(np.sum(d21 >= t)/1000.)
            tp21[0], tp21[-1] = 0, 1
            fp21[0], fp21[-1] = 0, 1
            auc_21.append(auc_calc(fp21, tp21))

        auc_wgs_df.loc[len(auc_wgs_df)] = [ff,
                                       np.mean(auc_13),
                                       np.percentile(auc_13, 2.5),
                                       np.percentile(auc_13, 97.5),
                                       np.mean(auc_18),
                                       np.percentile(auc_18, 2.5),
                                       np.percentile(auc_18, 97.5),
                                       np.mean(auc_21),
                                       np.percentile(auc_21, 2.5),
                                       np.percentile(auc_21, 97.5)]

    plt.rc('figure', figsize=(10,10))
    panels = (1,1)
    fig,ax = plt.subplots(panels[0],panels[1])

    tick_font_size = 40
    label_font_size = 40
    box_width = 4

    ax.plot(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR13_MEAN_AUC'], 7, 2), lw=5, color="#111111", alpha=1, label="T13")
    ax.fill_between(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR13_LOW'], 7, 2), savgol_filter(auc_wgs_df['CHR13_HIGH'],7,2), color="#111111", alpha=0.2)
    ax.plot(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR18_MEAN_AUC'], 7, 2), lw=5, color="#666666", alpha=1, label="T18")
    ax.fill_between(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR18_LOW'], 7, 2), savgol_filter(auc_wgs_df['CHR18_HIGH'],7,2), color="#666666", alpha=0.2)
    ax.plot(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR21_MEAN_AUC'], 7, 2), lw=5, color="#AAAAAA", alpha=1, label="T21")
    ax.fill_between(auc_wgs_df["FF"], savgol_filter(auc_wgs_df['CHR21_LOW'], 7, 2), savgol_filter(auc_wgs_df['CHR21_HIGH'],7,2), color="#AAAAAA", alpha=0.2)

    ax.set_ylabel('Area under the curve (AUC)', fontsize=label_font_size)
    ax.set_xlabel('Fetal fraction', fontsize=label_font_size)
    ax.set_xticks(np.arange(0, 0.041, 0.01))
    ax.set_xticklabels(["0", "1%", "2%", "3%", "4%"])
    ax.set_xlim(-0.001, 0.042)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylim(0.47, 1.03)

    legend = ax.legend(loc=4, fontsize=40, frameon=False)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(7)

    for tl in ax.get_xticklabels():
        tl.set_fontsize(tick_font_size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)
    [i.set_linewidth(3) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major')
    ax.tick_params('x', length=10, width=box_width, which='major')

    for x in ax.get_xaxis().majorTicks:
        x.set_pad(15)
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(15)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]

    fig.savefig(FIGURES_DIR + "Fig_3B_WGS_AUCs_vs_FF.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    ###
    # Panel C - Sensitivity of SNP method for detection of maternal m1,
    #           maternal m2, and paternal origins of trisomy for fetal
    #           fractions 1, 2, 3, and 4%.
    ###

    specificity = 0.9987 #Equivalent to z-score threshold of 3

    samples = ["{}_{}".format(x, y) for x in ["D", "M1", "M2", "P"] for y in range(1, 5)]
    folder = ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005/"
    files = [folder + "snp_{}_ff0.0{}0_10000.csv".format(x, y) for x in ["d", "m1", "m2", "p1"] for y in range(1, 5)]

    df = pd.DataFrame()

    for s, f in zip(samples, files):
        df_t = pd.read_csv(f)
        df[s] = df_t.LOR

    N = 3
    prev = [0.7, 0.2, 0.1] #Prevalence
    ff1 = [calc_sens_at_spec(df[d], df[a], specificity) for d, a in zip(["D_1"] * 3, ["M1_1", "M2_1", "P_1"])]
    ff2 = [calc_sens_at_spec(df[d], df[a], specificity) for d, a in zip(["D_2"] * 3, ["M1_2", "M2_2", "P_2"])]
    ff3 = [calc_sens_at_spec(df[d], df[a], specificity) for d, a in zip(["D_3"] * 3, ["M1_3", "M2_3", "P_3"])]
    ff4 = [calc_sens_at_spec(df[d], df[a], specificity) for d, a in zip(["D_4"] * 3, ["M1_4", "M2_4", "P_4"])]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.10        # the width of the bars
    space = 0.05

    plt.rc('figure', figsize=(10, 10))  #Change plot defaults
    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 40
    label_font_size = 45
    box_width = 4

    offset = 0.31
    rects0 = ax.bar(ind, prev, width * 2.5, color="white", lw=3, zorder=3)
    rects1 = ax.bar(ind + offset, ff1, width, color=["red", "orange", "blue"], alpha=0.25, lw=2)
    rects2 = ax.bar(ind + offset + width + space, ff2, width, color=["red", "orange", "blue"], alpha=0.5, lw=2)
    rects3 = ax.bar(ind + offset + space * 2 + width * 2, ff3, width, color=["red", "orange", "blue"], alpha=0.75, lw=2)
    rects4 = ax.bar(ind + offset + space * 3 + width * 3, ff4, width, color=["red", "orange", "blue"], alpha=1, lw=2)

    ax.plot((0.93, 0.93), (0, 1.1), lw=3, ls="--", color="#AAAAAA")
    ax.plot((1.93, 1.93), (0, 1.1), lw=3, ls="--", color="#AAAAAA")

    ax.set_xlim(-width * 2, 3.1)
    ax.set_xticks(ind + 0.16 + space * 2 + width * 2)
    ax.set_xticklabels(('Maternal\nM1', 'Maternal\nM2', 'Paternal'))
    for tl in ax.get_xticklabels():
        tl.set_fontsize(35)

    ax.set_ylim(-0.01, 1.05)
    ax.set_ylabel('SNP method sensitivity', fontsize=label_font_size)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major')
    ax.tick_params('x', length=0, width=box_width, which='major')

    for x in ax.get_xaxis().majorTicks:
        x.set_pad(15)
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(15)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]

    fig.savefig(FIGURES_DIR + "Fig_3C_SNP_sens_by_trisomy_origin.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)


    ###
    # Panel D - Comparison of the sensitivity of the WGS method for the three
    #           chromosomes of interest to the aggregate sensitivity of the
    #           SNP method for all parental origins of trisomy for fetal
    #           fractions 1, 2, 3, and 4%.
    ###

    wgs_file = ANALYSES_DIR + "wgs_method_10000affected.csv"
    snp_dir = ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005/"
    outfile = FIGURES_DIR + "Fig_3D_comp_WGS_and_SNP_sens.png"
    comp_sensitivity_plot(wgs_file=wgs_file,
                          snp_dir=snp_dir,
                          outfile=outfile,
                          ylab=True)

def comp_sensitivity_plot(wgs_file, snp_dir, outfile, ylab=False, snp_no_call=True):

    #Read in WGS sensitivity values
    df = pd.read_csv(ANALYSES_DIR + wgs_file)

    wgs21 = [df.iloc[9, 5], df.iloc[19, 5], df.iloc[29, 5], df.iloc[39, 5]]
    wgs13 = [df.iloc[9, 1], df.iloc[19, 1], df.iloc[29, 1], df.iloc[39, 1]]
    wgs18 = [df.iloc[9, 3], df.iloc[19, 3], df.iloc[29, 3], df.iloc[39, 3]]

    #Calculate aggregate SNP method sensitivity
    samples = ["{}_{}".format(x, y) for x in ["D", "M1", "M2", "P"] for y in range(1, 5)]
    folder = snp_dir + "/"
    files = [folder + "snp_{x}_ff{y:0.3f}_10000.csv".format(x=x, y=y) for x in ["d", "m1", "m2", "p1"] for y in np.arange(0.01, 0.05, 0.01)]

    df_S1 = pd.DataFrame()

    for s, f in zip(samples, files):
        df = pd.read_csv(f)
        df_S1[s] = df.LOR

    prev = [0.7,0.2,0.1]
    ff1_snp = [calc_sens_at_spec(df_S1[d], df_S1[a], 0.99) for d, a in zip(["D_1"] * 3,["M1_1", "M2_1", "P_1"])]
    ff2_snp = [calc_sens_at_spec(df_S1[d], df_S1[a], 0.99) for d, a in zip(["D_2"] * 3,["M1_2", "M2_2", "P_2"])]
    ff3_snp = [calc_sens_at_spec(df_S1[d], df_S1[a], 0.99) for d, a in zip(["D_3"] * 3,["M1_3", "M2_3", "P_3"])]
    ff4_snp = [calc_sens_at_spec(df_S1[d], df_S1[a], 0.99) for d, a in zip(["D_4"] * 3,["M1_4", "M2_4", "P_4"])]


    snp_agg = [np.dot(ff1_snp, prev), np.dot(ff2_snp, prev), np.dot(ff3_snp, prev), np.dot(ff4_snp, prev)]
    snp_agg1 = [0, 0, np.dot(ff3_snp, prev), np.dot(ff4_snp, prev)]
    snp_agg2 = [np.dot(ff1_snp, prev), np.dot(ff2_snp, prev), 0, 0]

    plt.rc('figure', figsize=(10, 10))  #Change plot defaults
    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 50
    label_font_size = 60
    box_width = 4

    N=4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    space = 0.05
    lw = 3

    rects1 = ax.bar(ind, wgs21, width, color="#AAAAAA", lw=lw, zorder=3)
    rects2 = ax.bar(ind + width + space, wgs18, width, color="#666666", lw=lw)
    rects3 = ax.bar(ind + space * 2 + width * 2, wgs13, width, color="#111111", linewidth=lw)
    if snp_no_call:
        rects4 = ax.bar(ind + space * 3 + width * 3, snp_agg2, width, color="white", edgecolor="red", lw=lw)
        rects4 = ax.bar(ind + space * 3 + width * 3, snp_agg1, width, color="red", lw=lw)
    else:
        rects4 = ax.bar(ind + space * 3 + width * 3, snp_agg, width, color="red", lw=lw)


    ax.plot((0.86, 0.86),(0, 1.1), lw=3, ls="--", color="#AAAAAA")
    ax.plot((1.86, 1.86),(0, 1.1), lw=3, ls="--", color="#AAAAAA")
    ax.plot((2.86, 2.86),(0, 1.1), lw=3, ls="--", color="#AAAAAA")

    if ylab:
        ax.set_ylabel('Sensitivity', fontsize=label_font_size)
    else:
        ax.set_ylabel('Sensitivity', fontsize=label_font_size, color="white")
    ax.set_xlabel('Fetal fraction', fontsize=label_font_size)
    ax.set_xlim(-width, 3.9)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xticks(ind + space * 2 + width * 2)
    ax.set_xticklabels(("1%", "2%", "3%", "4%"))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    for tl in ax.get_xticklabels():
        tl.set_fontsize(tick_font_size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)
    [i.set_linewidth(3) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major')
    ax.tick_params('x', length=0, width=box_width, which='major')

    for x in ax.get_xaxis().majorTicks:
        x.set_pad(15)
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(15)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]

    fig.savefig(outfile,
                dpi=200,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

def plot_figure_4_panels():
    """
    Plot panels used in Figure 4.
    """

    ###
    # Panel B - Comparison of the senstivity of WGS and SNP methods at
    #           detecting T21 as a function of inv. proc rate.
    ###

    #Fit beta to FF distribution parameters given in Nicolaides et al. 2012
    a, b = opt_beta(0.105, 0.078, 0.13)

    df = pd.read_csv(ANALYSES_DIR + "wgs_method_10000affected.csv", header=0)

    #Calculate aggregate sensitivity of T21 over the < 2.8% FF range
    sens = df.CHR21_SENSITIVITY[:27] #Sensitivities in 0.1% increments

    #Get normalized frequencies of samples in the < 2.8% FF range
    freq = ss.beta.pdf(np.arange(0.001, 0.028, 0.001), a=a, b=b, loc=0, scale=1)
    freq /= sum(freq)

    wgs_aggregate_sensitivity = np.dot(sens, freq)

    #Perform over range of invasive procedure rates

    df = pd.DataFrame(columns=["A_RATE", "DET_RATE"])

    #SNP rate of detection
    cases = 1000
    t21_rate = 3.3 / 100. #T21 rate from Taylor-Phillips et al. 2016.
    rd_submit = 0.565 #Proportion of people that submit a redraw from Dar et al. 2014
    rd_success = 0.74 #Generous from Ryan et al. 2016 (assume that all redraws are successful)

    for i in np.arange(0, 1.02, 0.02):

        inv_rate = i #Proportion of women who consent to inv. procedure when POSITIVE (Dar et al. 2014)

        t21s = int(np.round(cases * t21_rate,0)) #Total T21s

        #Number of cases that are ultimately no-called
        no_calls = (cases * rd_submit * (1 - rd_success)) + (cases * (1 - rd_submit))

        #Total invasive procedures
        inv_procs = no_calls * inv_rate

        #Detected T21s
        det_rd = int(np.round((cases - no_calls) * t21_rate, 0)) #By redraw
        det_inv = int(np.round(inv_procs * t21_rate, 0))

        df.loc[len(df)] = [i,np.round((det_inv + det_rd)/float(t21s), 4)]

    figure_options = {'figsize':(10, 10)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 40
    label_font_size = 45
    box_width = 4

    x = (df.A_RATE[0], df.A_RATE[50])
    y = (df.DET_RATE[0], df.DET_RATE[50])

    ax.plot([0.55, 0.55], [0, 1], color="#777777", lw=4, ls="-")
    ax.plot(x, y, color="red", lw=7, label="T21 detection rate")
    ax.plot(x, [wgs_aggregate_sensitivity, wgs_aggregate_sensitivity], color="black", lw=7, ls="-", label="WGS detection rate")

    for tl in ax.get_xticklabels():
        tl.set_fontsize(tick_font_size)
    ax.set_xlabel("Patients receiving\ninvasive procedures", fontsize=label_font_size)
    ax.set_ylabel("T21 sensitivity\n at <2.8% FF", fontsize=label_font_size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major', pad=10)
    ax.tick_params('x', length=10, width=box_width, which='major', direction='in', pad=15)

    fig.savefig(FIGURES_DIR + "Fig_4B_WGS_vs_SNP_det_rate.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    ###
    # Panel C - Illustration of 10,000 clinical outcomes.
    ###

    #SNP rate of detection
    cases = 10000
    inv_rate = 0.55 #Proportion of women who consent to invasive procedure when POSITIVE (Dar et al. 2014)
    loss_rate = 0.002 #Rate of aneuploidy loss from Yaron et al. 2016

    t21s = int(np.round(cases * t21_rate, 0)) #Total T21s

    #Number of cases that are ultimately no-called
    no_calls = (cases * rd_submit * (1 - rd_success)) + (cases * (1 - rd_submit))

    #Total invasive procedures
    inv_procs = int(np.round(no_calls * inv_rate, 0))

    #Detected T21s
    det_rd = int(np.round((cases - no_calls) * t21_rate, 0)) #By redraw
    det_inv = int(np.round(inv_procs * t21_rate, 0))

    #Procedural loss
    loss = int(np.round(inv_procs * loss_rate, 0))

    wgs_detected = int(np.round(t21s * wgs_aggregate_sensitivity, 0))
    wgs_fn = t21s - wgs_detected

    snp_dt_rd = det_rd
    snp_dt_inv = det_inv
    snp_fn = t21s - det_rd - det_inv
    snp_loss = loss

    wgs_det = [wgs_detected, 0]
    rd_det = [0, snp_dt_rd]
    inv_det = [0, snp_dt_inv]
    fn = [wgs_fn, snp_fn]
    inv_loss = [0, snp_loss]
    inv_p = [0, inv_procs]

    figure_options = {'figsize':(13, 30)}
    font = {'weight' : 'normal',
            'size'   : 16}
    plt.rc('font', **font)
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 1)
    fig,ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 55
    label_font_size = 80
    box_width = 4
    bar_width = 0

    N = 2
    ind = np.arange(N)    # the x locations for the groups
    width = 0.50       # the width of the bars: can also be len(x) sequence

    #Figure out z-axis
    alpha = 1
    ax.bar(ind+width, wgs_det, width, color="lightgreen", alpha=alpha, label='WGS detected', lw=bar_width)
    ax.bar(ind+width, rd_det, width, color="lightgreen", alpha=alpha,bottom=[wgs_det[x] for x in range(len(wgs_det))], label='SNP detected by redraw', lw=bar_width, zorder=1)
    ax.bar(ind+width, inv_det, width, color="skyblue", alpha=alpha, bottom=[wgs_det[x] + rd_det[x] for x in range(len(wgs_det))], label='SNP detected by inv. proc.', lw=bar_width, zorder=1)
    ax.bar(ind+width, fn, width, color="salmon",alpha=alpha, bottom=[wgs_det[x] + rd_det[x] + inv_det[x] for x in range(len(wgs_det))], label='False-negative', lw=bar_width, zorder=1)
    ax.bar(ind+width, inv_loss, width, color="yellow",alpha=alpha, bottom=[wgs_det[x] + rd_det[x] + inv_det[x] + fn[x] for x in range(len(wgs_det))], label='Pregnancy-loss', lw=bar_width, zorder=1)
    ax.bar(ind+width, inv_p, width, color="#999999", alpha=alpha, bottom=[wgs_det[x] + rd_det[x] + inv_det[x] + fn[x] + inv_loss[x] for x in range(len(wgs_det))], label='Invasive procedures',
           lw=bar_width, zorder=1)

    line_col="#666666"
    offset = 0.01
    ax.plot([width/2.,1 - 2 * width/2 - offset], [wgs_det,wgs_det], lw=3, ls="--", color=line_col, zorder=2)
    ax.plot([0.5 + 2 * width/2 + offset,2 - 2 * width/2 - offset], [wgs_det,wgs_det], lw=3, ls="--", color=line_col, zorder=2)
    ax.plot([1.5 + 2 * width/2 + offset,3], [wgs_det,wgs_det], lw=3, ls="--", color=line_col, zorder=2)

    ax.set_xticks(ind + width * 1.5)
    ax.set_xticklabels(('WGS', 'SNP'), fontsize=label_font_size)

    ax.set_ylim([0, 3600])
    ax.set_xlim([width/2.,2 + width/2])
    ax.set_yticks(np.arange(0, 3800, 400))
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)
    ax.set_ylabel('Pregnancies', fontsize=label_font_size)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major', pad=10)
    ax.tick_params('x', length=10, width=box_width, which='major',direction='in', pad=10)

    fig.savefig(FIGURES_DIR + "Fig_4C_clinical_outcomes.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    #Uncomment to print values displayed by the bars.
    # print "WGS_DET:", wgs_detected
    # print "WGS_FN:", wgs_fn
    # print "SNP_REDRAW_DET:", snp_dt_rd
    # print "SNP_INV_DET:", snp_dt_inv
    # print "SNP_FN:", snp_fn
    # print "INV_LOSS:", snp_loss
    # print "TOTAL INV:", inv_procs

def opt_beta(median, quart1, quart3):
    """
    Find the beta distribution that most closely fits most closely to the
    specified median, first, and third quartiles by minimizing the sum of
    squares deviation from the 25th and 75th percentiles (i.e., first and
    third quartiles).

    Args:
        median (float): median of the beta distribution
        quart1 (float): value corresponding to the first quartile
        quart3 (float): value corresponding to the third quartile

    Returns:
        a,b: alpha and beta parameters of the beta distribution

    Raises:
        AssertionError: If median <= 0 or >= 1
        AssertionError: If quart1 <= 0 or >= 1
        AssertionError: If quart3 <= 0 or >= 1
        AssertionError: If not quart1 < median < quart3

    """

    assert 0 < median < 1, "median must be > 0 and < 1"
    assert 0 < quart1 < 1, "quart1 must be > 0 and < 1"
    assert 0 < quart3 < 1, "quart3 must be > 0 and < 1"
    assert quart1 < median < quart3, "Values must be: quart1 < median < quart3"

    prior = [0, 0]

    def _beta(popsize):
        a = popsize * median + prior[0]
        b = popsize - a + prior[1]
        dist = beta(a, b)
        inf_lo = dist.ppf(0.25)
        inf_hi = dist.ppf(0.75)
        obj = (quart1 - inf_lo) ** 2 + (quart3 - inf_hi) ** 2
        if np.isnan(obj):
            return 2
        return obj

    res = brute(_beta, ranges=[(1, 1e6)], Ns=100, full_output=1)
    popsize = res[0]
    a = popsize * median
    b = popsize - a

    return a, b


def calc_sens_at_spec(disomy, trisomy, specificity):
    """
    Calculate the maximum sensitivity achievable at a pre-specified specificity
    given the distributions of LORs of disomic and and trisomic samples.

    Args:
        disomy (list): list of LORs for the disomic samples
        trisomy (list): list of LORs for the trisomic samples
        specificity (float): specificity af a fraction

    Returns:
        sensitivity (fraction)

    Raises:
        AssertionError: If spec <= 0 or >= 1
    """

    assert 0 < specificity < 1, "specificity must be > 0 and < 1"

    specificity *= 100

    #Convert infinities to numbers
    trisomy[trisomy == np.inf] = 1000000
    trisomy[trisomy == -np.inf] = -1000000
    disomy[disomy == np.inf] = 1000000
    disomy[disomy == -np.inf] = -1000000

    thresh = np.percentile(disomy, 100-specificity)
    sensitivity = np.sum(trisomy < thresh) / float(len(trisomy))

    return sensitivity

if __name__ == '__main__':
    main()
