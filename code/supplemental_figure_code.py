#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt

from sys import argv

import numpy as np
import pandas as pd
import scipy.stats as ss

from collections import defaultdict
from scipy.signal import savgol_filter

from snp_method_simulation import simulate_snp_sample
from figure_code import (comp_sensitivity_plot,
                         generate_snp_method_fig1_plot,
                         generate_snp_method_fig2_plot,
                         get_colors,
                         opt_beta)

#Define input/output directories in relation to code directory
ANALYSES_DIR = "../simulation_output/"
FIGURES_DIR = "../manuscript_figure_panels/"

#Set DPI on PNGs produced by code.
FIG_DPI = 300

#If run as figure_code.py -e, reproduce the manuscript figs exactly.
if argv[-1] == "-e":
    np.random.seed(1000)

def main():
    """
    Plot the individual panels of each of the figures in the manuscript
    supplement. Note that Supplemental Figure S6 uses the same panels as
    Figure 2 from the main manuscript.
    """

    plot_figure_S1_panels()

    plot_figure_S2_panels()

    plot_figure_S3_panels()

    plot_figure_S4_panels()

    plot_figure_S5_panels()

    plot_figure_S6_panels()

    plot_figure_S7_panels()

def plot_figure_S1_panels():
    """
    Plot panels used in Figure S1.
    """


    ###
    # A Panels - plot of dispersion of bin counts across simulated chromosome
    #            21
    ###

    nbdisps = [0.1, 0.333, 0.666, 1] #Note that nbdisp = 1 collapses to Poisson
    outfiles = ["Fig_S1A_wgs_neg_binom_0.1.png",
                "Fig_S1A_wgs_neg_binom_0.333.png",
                "Fig_S1A_wgs_neg_binom_0.666.png",
                "Fig_S1A_wgs_neg_binom_Poisson.png"]

    for nbdisp, outfile in zip(nbdisps, outfiles):
        if nbdisp == 0.1:
            plot_wgs_chr21_dip_tri(nbdisp=nbdisp, outfile=outfile, ylab=True)
        else:
            plot_wgs_chr21_dip_tri(nbdisp=nbdisp, outfile=outfile)


    ###
    # B Panels - Reproduction of manuscript Figure 3D under different values of
    #            nbdisp for the WGS method.
    ###

    wgs_files = ["wgs_method_10000affected_nb0.1.csv",
                 "wgs_method_10000affected_nb0.333.csv",
                 "wgs_method_10000affected_nb0.666.csv",
                 "wgs_method_10000affected.csv"]

    outfiles = ["Fig_S1B_wgs_neg_binom_0.1.png",
                "Fig_S1B_wgs_neg_binom_0.333.png",
                "Fig_S1B_wgs_neg_binom_0.666.png",
                "Fig_S1B_wgs_neg_binom_Poisson.png"]

    snp_dir = ANALYSES_DIR +"snp_method_bbdisp1000_nbdisp0.0005"

    for wgs_file, outfile in zip(wgs_files, outfiles):

        ylab = False
        if wgs_file == "wgs_method_10000affected_nb0.1.csv":
            ylab = True
        comp_sensitivity_plot(ANALYSES_DIR + wgs_file,
                              snp_dir,
                              FIGURES_DIR + outfile,
                              ylab=ylab)

def plot_wgs_chr21_dip_tri(outfile, nbdisp=1, ff=0.1, ylab=False):
    """
    Generate plot of simulated WGS method bins under diploid and triploid
    scenarios for chromosome 21 along with the distribution from which those
    plots were drawn.

    Args:
        outfile (str): The location of the output PNG file.
        ff (float): Fetal fraction
        nbdisp (float): The negative-binomial dispersion coefficient.
        ylab (bool): If true, add a label to the y axis of the plot.

    Returns:
        None

    Raises:
        AssertionError: If nbdisp <= 0 or nbdisp > 1
    """

    assert 0 < nbdisp <= 1, "Value of nbdisp must be 0 < nbdisp <= 1"

    #Generate data
    reads_per_sample = 16e6
    genome_size = 3e9
    bin_size = 50e3
    depth_per_bin = reads_per_sample / (genome_size / bin_size)
    chroms = ["chr21_dip", "chr21_an"]

    chrom_len = 48129895
    chrom_bins = defaultdict(list)

    for x in chroms:
        num_bins = int(chrom_len/bin_size * 0.9)

        #If nbdisp == 1, then collapse to Poisson distribution
        if nbdisp == 1:
            if x == "chr21_an":
                chrom_bins[x] = np.random.poisson((1+ff/2.) * depth_per_bin, num_bins)
            else:
                chrom_bins[x] = np.random.poisson(depth_per_bin, num_bins)

        #Otherwise sample from negative binomial with defined dispersion
        else:
            if x == "chr21_an":
                a = (1+ff/2.) * depth_per_bin * nbdisp / float(1-nbdisp)
            else:
                a = depth_per_bin * nbdisp / float(1-nbdisp)
            chrom_bins[x] = np.random.negative_binomial(a, nbdisp, num_bins)

    y_axis = []
    cols = []
    flip = True
    tick_pos = []
    last_pos = 0
    for chrom in chroms:
        y_axis.extend(chrom_bins[chrom])
        tick_pos.append(len(chrom_bins[chrom])/2 + last_pos)
        y_axis.extend([-20] * 300)
        last_pos = len(y_axis)

        if chrom == "chr21_an":
            cols.extend(["red"] * (len(chrom_bins[chrom]) + 300))
            if flip:
                flip = False
            else:
                flip = True
        elif flip:
            cols.extend(["darkgray"] * (len(chrom_bins[chrom]) + 300))
            flip = False
        else:
            cols.extend(["lightgray"] * (len(chrom_bins[chrom]) + 300))
            flip = True

    x_axis = range(len(y_axis))

    figure_options = {'figsize':(10, 10)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 2)
    fig, ax = plt.subplots(panels[0], panels[1])

    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])

    y_max = 500
    y_min = 100
    box_width = 4

    ax[0].scatter(x_axis, y_axis, color=cols, s=20, alpha=1)
    ax[0].set_xlim([-200, 2250])
    ax[0].set_ylim([y_min, y_max])
    ax[0].yaxis.set_major_locator(plt.NullLocator())
    x_ticks = tick_pos
    ax[0].set_xticks(x_ticks)
    ax[0].xaxis.set_tick_params(width=box_width, length=12)

    ax[0].set_xticklabels(["Dis","Tri"], fontsize=50, rotation=45)
    if ylab:
        ax[0].set_ylabel("Counts per bin", fontsize=60)
    else:
        ax[0].set_ylabel("Counts per bin", fontsize=60, color="white")
    [i.set_linewidth(box_width) for i in ax[0].spines.itervalues()]
    ax[0].get_xaxis().tick_bottom()

    y_axis = range(y_min,y_max)
    if nbdisp == 1:
        normal = [ss.poisson.pmf(x,depth_per_bin) for x in y_axis]
        aneuploid = [ss.poisson.pmf(x, (1+ff/2.)*depth_per_bin) for x in y_axis]
        normal[0], normal[-1] = 0,0
        aneuploid[0],aneuploid[-1] = 0,0
        ax[1].fill_between(normal, y_axis, color="darkgrey", alpha=0.5)
        ax[1].fill_between(aneuploid, y_axis, color="red", alpha=0.5)
    else:
        a = depth_per_bin * nbdisp / float(1-nbdisp)
        normal = [ss.nbinom.pmf(x, a, nbdisp) for x in y_axis]
        a = (1+ff/2.) * depth_per_bin * nbdisp / float(1-nbdisp)
        aneuploid = [ss.nbinom.pmf(x, a, nbdisp) for x in y_axis]
        normal[0], normal[-1] = 0, 0
        aneuploid[0],aneuploid[-1] = 0, 0
        ax[1].fill_between(normal, y_axis, color="darkgrey", alpha=0.5)
        ax[1].fill_between(aneuploid, y_axis, color="red", alpha=0.5)

    ax[1].set_ylim([y_min, y_max])
    ax[1].set_xlim([0, 0.025])
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

def plot_figure_S2_panels():
    """
    Plot panels used in Figure S2.
    """

    ###
    # Reproduction of Figure 3 from Hall MP, Hill M, Zimmermann B, Sigurjonsson
    # S, Westemeyer M et al. 2014. Non-invasive prenatal detection of trisomy
    # 13 using a single nucleotide polymorphism- and informatics-based
    # approach. PLoS One. 9(5):e96677. - applying the SNP method simulation
    # parameters used in the manuscript.
    ###

    ff = 0.192             #Fetal fraction
    num_snps=3348          #Number of SNPs per chromosome
    mean_counts = 859      #Mean counts per SNP
    bbdisp = 1000          #Beta dispersion factor
    nbdisp = 0.0005        #Negative binomial dispersion parameter
    H = (1, 2 ,2)          #Paternal M2 trisomy

    df_points_tri = simulate_snp_sample(ff=ff,
                                        H=H,
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=nbdisp)

    H = (1, 1, None)         #Disomy

    df_points_di1 = simulate_snp_sample(ff=ff,
                                        H=H,
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=nbdisp)
    df_points_di2 = simulate_snp_sample(ff=ff,
                                        H=H,
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=nbdisp)
    df_points_di3 = simulate_snp_sample(ff=ff,
                                        H=H,
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=nbdisp)

    df_points_tri = df_points_tri[df_points_tri["TOT_COUNTS"] > 0]
    bar1 = len(df_points_tri)
    df_points_di1 = df_points_di1[df_points_di1["TOT_COUNTS"] > 0]
    bar2 = len(df_points_di1) + bar1
    df_points_di2 = df_points_di2[df_points_di2["TOT_COUNTS"] > 0]
    bar3 = len(df_points_di2) + bar2
    df_points_di3 = df_points_di3[df_points_di3["TOT_COUNTS"] > 0]

    df = df_points_tri.append(df_points_di1)
    df = df.append(df_points_di2)
    df = df.append(df_points_di3)

    bar4 = len(df)

    figure_options = {'figsize':(12, 10)}
    plt.rc('figure', **figure_options)  #Change plot defaults

    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    x_vals = range(len(df))
    cols = get_colors(df["MAT_GENO"])

    ax.scatter(x_vals, np.true_divide(df["A_COUNT"], df["TOT_COUNTS"]), color=cols, s=30, alpha=0.5)
    ax.plot([10, 10], [0, 1], color="grey", ls="--", lw=3)
    ax.plot([bar1, bar1], [0, 1], color="grey", ls="--", lw=3)
    ax.plot([bar2, bar2], [0, 1], color="grey", ls="--", lw=3)
    ax.plot([bar3, bar3], [0, 1], color="grey", ls="--", lw=3)
    ax.plot([bar4 - 10, bar4 - 10], [0, 1], color="grey", ls="--", lw=3)
    ax.set_xlim([0, len(x_vals)])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.set_ylim([0, 1])
    y_ticks = np.arange(0, 1.01, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=20)
    ax.set_ylabel("Fraction of A allele reads", fontsize=20)
    ax.set_xlabel("Chromosomal Position", fontsize=20)
    ax.yaxis.grid(b=False, which='major', color='grey', linestyle='-')
    [i.set_linewidth(0) for i in ax.spines.itervalues()]
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(10)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(FIGURES_DIR + "Fig_S2_Rep_Hall_et_al_2014_Fig3.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

def plot_figure_S3_panels():
    """
    Plot panels used in Figure S3.
    """

    ###
    # A Panels - Allele fractions and corresponding distributions for samples
    #            simulated under the SNP method with varying values of bbdisp
    ###

    ff = 0.1               # Fetal fraction
    num_snps=3348          # Number of SNPs per chromosome
    mean_counts = 859      # Mean counts per SNP
    nbdisp = 0.0005        # Negative binomial dispersion parameter

    panels = {
        "01_disomy_bbdisp100":[(1, 1, None), 100],
        "02_disomy_bbdisp1000":[(1, 1, None), 1000],
        "03_disomy_bbdisp10000":[(1, 1, None), 10000],
        "04_disomy_binomial":[(1, 1, None), 'binomial'],
        "05_maternal_m1_bbdisp100":[(2, 1, 1), 100],
        "06_maternal_m1_bbdisp1000":[(2, 1, 1), 1000],
        "07_maternal_m1_bbdisp10000":[(2, 1, 1), 10000],
        "08_maternal_m1_binomial":[(2, 1, 1), 'binomial'],
    }

    for key, val in panels.iteritems():

        ylab = False
        if (key == "01_disomy_bbdisp100" or
            key == "05_maternal_m1_bbdisp100"):
            ylab = True

        df_points = simulate_snp_sample(ff=ff,
                                        H=val[0],
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=val[1],
                                        nbdisp=nbdisp)

        df_dist = simulate_snp_sample(ff=ff,
                                      H=val[0],
                                      num_snps=100000,
                                      mean_counts=mean_counts,
                                      bbdisp=val[1],
                                      nbdisp=nbdisp)

        #For distribution plotting purposes, setting bbdisp to extremely high
        #value is indistiguishable from the binomial.
        generate_snp_method_fig1_plot(points=df_points,
                                      distros=df_dist,
                                      outfile="Fig_S3A_{}.png".format(key),
                                      bbdisp=val[1],
                                      nbdisp=nbdisp,
                                      ylab=ylab)

    ###
    # B Panels - Reproduction of manuscript Figure 3D under different values of
    #            bbdisp under the SNP method.
    ###

    wgs_file = "wgs_method_10000affected.csv"

    snp_dirs = [ANALYSES_DIR + "snp_method_bbdisp100_nbdisp0.0005",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005",
                ANALYSES_DIR + "snp_method_bbdisp10000_nbdisp0.0005",
                ANALYSES_DIR + "snp_method_binomial_nbdisp0.0005"]

    outfiles = ["Fig_S3B_01_snp_bbdisp100.png",
                "Fig_S3B_02_snp_bbdisp1000.png",
                "Fig_S3B_03_snp_bbdisp10000.png",
                "Fig_S3B_04_snp_binomial.png"]

    for snp_dir, outfile in zip(snp_dirs, outfiles):

        ylab = False
        if snp_dir == snp_dirs[0]:
            ylab = True
        comp_sensitivity_plot(ANALYSES_DIR + wgs_file,
                              snp_dir,
                              FIGURES_DIR + outfile,
                              ylab=ylab)

def plot_figure_S4_panels():
    """
    Plot panels used in Figure S4.
    """

    ###
    # A Panels - Allele fractions and corresponding distributions for samples
    #            simulated under the SNP method with varying values of nbdisp
    ###

    ff = 0.1               # Fetal fraction
    num_snps=3348          # Number of SNPs per chromosome
    mean_counts = 859      # Mean counts per SNP
    bbdisp = 1000          # Beta binomial dispersion parameter

    panels = {
        "01_disomy_nbdisp0.0005":[(1, 1, None), 0.0005],
        "02_disomy_nbdisp0.005":[(1, 1, None), 0.005],
        "03_disomy_nbdisp0.05":[(1, 1, None), 0.05],
        "04_disomy_nbdisp1":[(1, 1, None), 1],
        "05_maternal_m1_nbdisp0.0005":[(2, 1, 1), 0.0005],
        "06_maternal_m1_nbdisp0.005":[(2, 1, 1), 0.005],
        "07_maternal_m1_nbdisp0.05":[(2, 1, 1), 0.05],
        "08_maternal_m1_nbdisp1":[(2, 1, 1), 1],
    }

    for key, val in panels.iteritems():

        ylab = False
        if (key == "01_disomy_nbdisp0.0005" or
            key == "05_maternal_m1_nbdisp0.0005"):
            ylab = True

        df_points = simulate_snp_sample(ff=ff,
                                        H=val[0],
                                        num_snps=num_snps,
                                        mean_counts=mean_counts,
                                        bbdisp=bbdisp,
                                        nbdisp=val[1])

        df_dist = simulate_snp_sample(ff=ff,
                                      H=val[0],
                                      num_snps=100000,
                                      mean_counts=mean_counts,
                                      bbdisp=bbdisp,
                                      nbdisp=val[1])

        generate_snp_method_fig1_plot(points=df_points,
                                      distros=df_dist,
                                      outfile="Fig_S4A_{}.png".format(key),
                                      bbdisp=bbdisp,
                                      nbdisp=val[1],
                                      ylab=ylab)

    ###
    # B Panels - Reproduction of manuscript Figure 3D under different values of
    #            nbdisp under the SNP method.
    ###

    wgs_file = "wgs_method_10000affected.csv"

    snp_dirs = [ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.005",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.05",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp1"]

    outfiles = ["Fig_S4B_01_snp_nbdisp0.0005.png",
                "Fig_S4B_02_snp_nbdisp0.005.png",
                "Fig_S4B_03_snp_nbdisp0.05.png",
                "Fig_S4B_04_snp_nbdisp1.png"]

    for snp_dir, outfile in zip(snp_dirs, outfiles):

        ylab = False
        if snp_dir == snp_dirs[0]:
            ylab = True
        comp_sensitivity_plot(ANALYSES_DIR + wgs_file,
                              snp_dir,
                              FIGURES_DIR + outfile,
                              ylab=ylab)

def plot_figure_S5_panels():
    """
    Plot panels used in Figure S5.
    """

    ###
    # Reproduction of manuscript Figure 3D varying the number of SNPs
    # interrogated on the chromosome of interest.
    ###

    wgs_file = "wgs_method_10000affected.csv"

    snp_dirs = [ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005_snps1000",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005_snps6000",
                ANALYSES_DIR + "snp_method_bbdisp1000_nbdisp0.0005_snps10000"]

    outfiles = ["Fig_S5_01_snp_1000snps.png",
                "Fig_S5_02_snp_3348snps.png",
                "Fig_S5_03_snp_6000snps.png",
                "Fig_S5_04_snp_10000snps.png"]

    for snp_dir, outfile in zip(snp_dirs, outfiles):

        ylab = False
        if snp_dir == snp_dirs[0]:
            ylab = True
        comp_sensitivity_plot(ANALYSES_DIR + wgs_file,
                              snp_dir,
                              FIGURES_DIR + outfile,
                              ylab=ylab)

def plot_figure_S6_panels():
    """
    Plot panels used in Figure S6.
    """

        #Diploid
    ff = 0.1               # Fetal fraction
    num_snps=3348          # Number of SNPs per chromosome
    mean_counts = 859      # Mean counts per SNP
    bbdisp = 1000          # Beta dispersion factor
    nbdisp = 0.0005        # Negative binomial dispersion parameter

    ploidies = {
        "01_maternal_m1":(2, 1, 1),
        "02_maternal_m2":(2, 1, 2),
        "03_paternal_m1":(1, 2, 1),
        "04_paternal_m2":(1, 2, 2)
    }

    for key, val in ploidies.iteritems():

        ylab = False
        if key == "01_maternal_m1" or key == "03_paternal_m1":
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
                                      outfile="Fig_S6_{}.png".format(key),
                                      ylab=ylab)

def plot_figure_S7_panels():
    """
    Plot panels used in Figure S7.
    """

    ###
    # Distribution of fetal fractions based on fitting a beta distribution to
    # the parameters reported in Nicolaides et al. 2012.
    ###

    #Fit beta to FF distribution parameters given in Nicolaides et al. 2012
    a,b = opt_beta(median=0.105,
                   quart1=0.078,
                   quart3=0.13)

    df = pd.read_csv(ANALYSES_DIR + "wgs_method_10000affected.csv", header=0)

    #Get normalized frequencies of samples in the < 2.8% FF range
    freq = ss.beta.pdf(np.arange(0.001, 0.028, 0.001), a=a, b=b, loc=0, scale=1)
    freq /= sum(freq)

    plt.rc('figure', figsize=(10, 10))  #Change plot defaults
    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 30
    label_font_size = 45
    box_width = 4

    xs = np.arange(0.001, 1, 0.001)
    ax.plot(xs, ss.beta.pdf(xs, a, b)/float(np.sum(ss.beta.pdf(xs, a, b))), lw=5)
    xs2 = np.arange(0.001, 0.028, 0.0001)
    ax.fill_between(xs2, ss.beta.pdf(xs2, a, b)/float(np.sum(ss.beta.pdf(xs, a, b))), color="red")
    ax.plot([0, 0.04],[0.002, 0.002], lw=5, ls="--", color="grey")
    ax.plot([0.04, 0.04],[0, 0.002], lw=5, ls="--", color="grey")
    ax.set_ylabel('Fraction of samples', fontsize=label_font_size)
    ax.set_xlabel('Fetal fraction', fontsize=label_font_size)
    ax.set_xticks(np.arange(0, 0.36, 0.05))
    ax.set_xticklabels(np.arange(0, 0.36, 0.05))
    ax.set_yticks(np.arange(0, 0.013, 0.002))
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.012])
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

    fig.savefig(FIGURES_DIR + "Fig_S7A_whole_ff_distro.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

    plt.rc('figure', figsize=(10, 10))  #Change plot defaults
    panels = (1, 1)
    fig, ax = plt.subplots(panels[0], panels[1])

    tick_font_size = 30
    label_font_size = 45
    box_width = 4

    xs = np.arange(0.001, 1, 0.001)
    ax.plot(xs, ss.beta.pdf(xs, a, b)/float(np.sum(ss.beta.pdf(xs, a, b))), lw=5)
    xs2 = np.arange(0.001, 0.028, 0.0001)
    ax.fill_between(xs2, ss.beta.pdf(xs2, a, b)/float(np.sum(ss.beta.pdf(xs, a, b))), color="red")

    ax2 = ax.twinx()
    ax2.plot(df.FETAL_FRAC, savgol_filter(df.CHR21_SENSITIVITY, 7, 2), ls="--", color="orange", lw=5)

    ax.set_ylabel('Fraction of samples', fontsize=label_font_size)
    ax.set_xlabel('Fetal fraction', fontsize=label_font_size)
    ax.set_xticks(np.arange(0, 0.041, 0.01))
    ax.set_xticklabels(np.arange(0, 0.041, 0.01))
    ax.set_yticks(np.arange(0, 0.0031, 0.001))
    ax.set_xlim([0, 0.04])
    ax.set_ylim([0, 0.002])
    for tl in ax.get_xticklabels():
        tl.set_fontsize(tick_font_size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(tick_font_size)

    ax2.set_ylabel('WGS method\n T21 sensitivity', fontsize=label_font_size)
    ax2.set_ylim([0, 1])
    ax2.set_yticks(np.arange(0, 1.01, 0.2))
    for tl in ax2.get_yticklabels():
        tl.set_fontsize(tick_font_size)

    [i.set_linewidth(3) for i in ax.spines.itervalues()]
    ax.tick_params('y', length=10, width=box_width, which='major')
    ax2.tick_params('y', length=10, width=box_width, which='major')
    ax.tick_params('x', length=10, width=box_width, which='major')

    for x in ax.get_xaxis().majorTicks:
        x.set_pad(15)
    for x in ax2.get_yaxis().majorTicks:
        x.set_pad(15)
    for y in ax.get_yaxis().majorTicks:
        y.set_pad(15)

    [i.set_linewidth(box_width) for i in ax.spines.itervalues()]
    [i.set_linewidth(box_width) for i in ax2.spines.itervalues()]

    fig.savefig(FIGURES_DIR + "Fig_S7B_low_ff_distro.png",
                dpi=FIG_DPI,
                bbox_inches="tight",
                format="png")
    plt.close(fig)

if __name__ == '__main__':
    main()