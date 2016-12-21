#!/usr/bin/env python
import argparse
import hashlib

import numpy as np
import pandas as pd

def parse_args():

    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    epilog = """\
Calculate sensitivity and specificity for the WGS method for chromosomes 13,18,
and 21, over a range of fetal fractions (0.1% to 4% in 0.1% increments).
Results are output to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Simulate WGS method",
                                     add_help=False,
                                     epilog=epilog,
                                     formatter_class=CustomFormatter)
    req = parser.add_argument_group('Required arguments')
    req.add_argument('-o', '--out',
                     dest='outfile',
                     action='store',
                     help='Output CSV',
                     type=str,
                     metavar='STR')
    opt = parser.add_argument_group('Optional arguments')
    opt.add_argument('-a', '--affected',
                     dest='affected',
                     action='store',
                     help='Number number of affected samples to simulate \
                          (default: 10,000)',
                     default=10000,
                     type=int,
                     metavar='INT')
    opt.add_argument('-n', '--nbdisp',
                     dest='nbdisp',
                     action='store',
                     help='Instead of using the Poisson distribution (Fan and \
                           Quake 2010) sample chromosomal bins from a negative\
                           -binomial distribution, with dispersion --nbdisp',
                     default=1,
                     type=float,
                     metavar='INT')
    opt.add_argument('-z', '--zscores',
                     dest='zscores',
                     action='store_true',
                     help='Rather than calculating performance parameters, \
                          output a CSV of z-scores.')
    opt.add_argument('-e', '--reproducible',
                     dest='reproducible',
                     action='store_true',
                     help='Seed random numbers to reproduce manuscript output \
                           and figs exactly.')
    opt.add_argument('-h', '--help',
                     action='help',
                     help='show this help message and exit')
    return parser.parse_args()

def main():
    """
    Parameters used in this analysis were obtained from the following sources
    as indicated in comments below:

    Jensen TJ, Zwiefelhofer T, Tim RC, Dzakula Z, Kim SK, Mazloom AR, et al.
    High-throughput massively parallel sequencing for fetal aneuploidy
    detection from maternal plasma. PLoS One. 2013 Mar 6;8(3):e57381.

    Taylor-Phillips S, Freeman K, Geppert J, Agbebiyi A, Uthman OA, Madan J,
    et al. Accuracy of non-invasive prenatal testing using cell-free DNA for
    detection of Down, Edwards and Patau syndromes: a systematic review and
    meta-analysis. BMJ Open. 2016 Jan 18;6(1):e010002.

    """

    args = parse_args()

    #If --reproducible/-e option is set, seed the random number generator in
    #order to reproduce the same output for the same parameter set.
    if args.reproducible:
        seed = get_wgs_seed(args)
        np.random.seed(seed)

    nbdisp = args.nbdisp

    # PARAMETERS
    READS_PER_SAMPLE = 16e6        #From Jensen et al. (2013)
    Z_CUTOFF_4_POS = 3             #Cutoff for positive call.
    SAMPLES_PER_BATCH = 100        #Samples per batch (i.e., flowcell)
    GENOME_SIZE = 3e9              #Size of genome
    BIN_SIZE = 50e3                #Bin size according to Jensen et al. (2013)

    #Rates obtained from Taylor-Phillips et al. 2016.
    ANEUPLOIDY_RATES = {"chr13":0.5/100.,
                        "chr18":1.5/100.,
                        "chr21":3.3/100.}

    #Chromosome sizes taken from hg19. 10% of bins discarded to account for
    #unmappable or high-GC bins (Jensen et al. 2013).
    CHROM_BINS = {'chr13': int(115169878/BIN_SIZE * 0.9),
                  'chr18': int(78077248/BIN_SIZE * 0.9),
                  'chr21': int(48129895/BIN_SIZE * 0.9)}

    counts_per_bin = READS_PER_SAMPLE * BIN_SIZE / GENOME_SIZE #Mean counts-per-bin

    #Output data as csv
    df = pd.DataFrame(columns=["FETAL_FRAC",
                               "CHR13_SENSITIVITY",
                               "CHR13_SPECIFICITY",
                               "CHR18_SENSITIVITY",
                               "CHR18_SPECIFICITY",
                               "CHR21_SENSITIVITY",
                               "CHR21_SPECIFICITY"])

    #Simulate over a range of fetal fractions (0.1 - 4% in 0.1% increments)
    low = 0.001
    high = 0.04
    increm = 0.001

    df_zscores = pd.DataFrame()

    for FF in np.arange(low, high+increm, increm):

        affected = {"chr13":0, #Keep simulating until -a affected samples for
                    "chr18":0, #each chromosome.
                    "chr21":0}

        row = [FF]
        for chrom in ["chr13", "chr18", "chr21"]:

            zs_d, zs_a = [], []

            num_bins = CHROM_BINS[chrom] #Number of bins on chromosome

            while affected[chrom] < args.affected:

                #Create the batch
                samples = np.random.choice([0, 1],
                                           size=SAMPLES_PER_BATCH,
                                           p=[1-ANEUPLOIDY_RATES[chrom],
                                              ANEUPLOIDY_RATES[chrom]])

                za, zd = [], []

                #Calculate mean counts-per-bin
                means = []
                for i in samples:
                    ff = 0
                    if i == 1:
                        ff = FF / 2.

                    mu = (1 + ff) * counts_per_bin
                    #By default, sample bins from the Poisson distribution,
                    #which is equivalent to setting nbdisp to 1.
                    if nbdisp == 1:
                        counts = np.random.poisson(lam=mu, size=num_bins)
                    else: #If 0 < nbdisp < 1, sample from the negative-binmial
                        a = mu * nbdisp / float(1-nbdisp)
                        counts = np.random.negative_binomial(a, nbdisp, num_bins)

                    means.append(np.mean(counts))

                #Calculate sample-specific z-score
                means = np.asarray(means)
                for i, j in zip(samples, range(SAMPLES_PER_BATCH)):
                    z = (means[j] - np.mean(means[np.arange(len(means)) != j]))/\
                        np.std(means[np.arange(len(means)) != j])

                    if i == 0:
                        zd.append(z)
                    else:
                        za.append(z)

                if zd:
                    if len(zs_d) < args.affected:
                        zs_d.append(np.random.choice(zd))
                if za:
                    if len(zs_a) < args.affected:
                        zs_a.append(np.random.choice(za))

                affected[chrom] = min(len(zs_a), len(zs_d))

            #Calculate sensitivity and specificity parameters
            zs_d = np.asarray(zs_d)
            zs_a = np.asarray(zs_a)
            sens = np.sum(zs_a >= Z_CUTOFF_4_POS) / float(len(zs_a))
            spec = np.sum(zs_d < Z_CUTOFF_4_POS) / float(len(zs_d))

            df_zscores["DIPLOID_{chrom}_{ff}".format(chrom=chrom, ff=FF)] = zs_d
            df_zscores["ANEUPLOID_{chrom}_{ff}".format(chrom=chrom, ff=FF)] = zs_a

            row.extend([sens, spec])

        df.loc[len(df)] = row

    #Either output a table of z-scores or a table of performance parameters.
    if args.zscores:
        df_zscores.to_csv(args.outfile, index=False)
    else:
        df.to_csv(args.outfile, index=False)

def get_wgs_seed(args):
    """
    Concatenate the parameters specified in the command-line arguments into a
    string and use to generate a sha1 hash that will form the basis for a seed
    value for numpy's random number generator. Ensures that a particular choice
    of parameters always produces the same output.

    Args:
        args(argparse.Namespace): The output of parse_args()

    Returns:
        32 bit int for use as numpy.random seed
    """

    a_list = [args.affected,
              args.nbdisp,
              args.zscores]

    args_string = "".join([str(x) for x in a_list])

    # numpy RNG seed must be a 32-bit integer
    return int(hashlib.sha1(args_string).hexdigest(),16) & (2**32 -1)

if __name__ == '__main__':
    main()
