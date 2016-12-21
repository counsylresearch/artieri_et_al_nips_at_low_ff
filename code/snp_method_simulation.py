#!/usr/bin/env python
import argparse
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
import pymc
from scipy.misc import logsumexp
import scipy.stats as ss

def parse_args():

    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    epilog = """\
Possible ploidy states are:

d, disomy
m1, maternal meiosis I trisomy
m2, maternal meiosis II trisomy
p1, paternal meiosis I trisomy
p2, paternal meiosis II trisomy

If the beta-binomial dispersion parameter (-b/--bbdisp) is set to 'binomial',
then data are simulated and LORs calculated assuming binomial sampling of
allelic reads.

If the negative-binomial dispersion parameter (-n/--nbdisp) is set to 1, the
negative-binomial distribution collapses to the Poisson distribution and
counts-per-SNP are sampled from the Poisson distribution.

"""
    parser = argparse.ArgumentParser(description=("Repeatedly simulate and " +
                                                  "call SNP method samples " +
                                                  "to calculate performance " +
                                                  "stats."),
                                     add_help=False,
                                     epilog=epilog,
                                     formatter_class=CustomFormatter)
    req = parser.add_argument_group('Required arguments')
    req.add_argument('-p', '--ploidy',
                     dest='ploidy',
                     action='store',
                     help='The chromosomal ploidy hypothesis to be tested (d, \
                           m1, m2, p1, or p2)',
                     choices=["d", "m1", "m2", "p1", "p2"],
                     type=str,
                     metavar='string')
    req.add_argument('-f', '--fetalfraction',
                     dest='ff',
                     action='store',
                     help='The fetal fraction at which to simulate data',
                     type=float,
                     metavar='float')
    req.add_argument('-o', '--outfile',
                     dest='outfile',
                     action='store',
                     help='Output CSV to store permuted LORs',
                     type=str,
                     metavar='CSV')
    opt = parser.add_argument_group('Optional arguments')
    opt.add_argument('-r', '--perms',
                     dest='perms',
                     action='store',
                     help='Number of samples to simulate and call LORs \
                     (default: 10000)',
                     default=10000,
                     type=int,
                     metavar='int')
    opt.add_argument('-b', '--bbdisp',
                     dest='bbdisp',
                     action='store',
                     help='beta-binomial dispersion parameter used to \
                     simulate/call data (default: 1000)',
                     default=1000,
                     metavar='float or \'binomial\'')
    opt.add_argument('-n', '--nbdisp',
                     dest='nbdisp',
                     action='store',
                     help='Negative-binomial dispersion parameter used to \
                           simulate data (default: 0.0005)',
                     default=0.0005,
                     type=float,
                     metavar='float')
    opt.add_argument('-s', '--snps',
                     dest='numsnps',
                     action='store',
                     help='The number of SNPs on the chromosome (default: \
                           3,348)',
                     default=3348,
                     type=int,
                     metavar='INT')
    opt.add_argument('-c', '--counts',
                     dest='mcounts',
                     action='store',
                     help='Mean counts-per-SNP (default: 859)',
                     default=859,
                     type=int,
                     metavar='INT')
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

    args = parse_args()

    #If --reproducible/-e option is set, seed the random number generator in
    #order to reproduce the same output for the same parameter set.
    if args.reproducible:
        seed = get_snp_seed(args)
        np.random.seed(seed)

    if args.bbdisp != "binomial":
        assert args.bbdisp > 1, "-b/--bbdisp must be > 1 or 'binomial'"
    assert 0 < args.nbdisp <= 1, "-n/--nbdisp must be 0 > and <= 1"
    assert 0 < args.ff < 1, "-f/--fetalfraction must be 0 > and < 1"

    ploidy_dict = {"d":(1, 1, None),
                   "m1":(2, 1, 1),
                   "m2":(2, 1, 2),
                   "p1":(1, 2, 1),
                   "p2":(1, 2, 2)}

    df_out = pd.DataFrame(columns=["LOR"])

    for _ in xrange(args.perms):

        df = simulate_snp_sample(ff=args.ff,
                                 H=ploidy_dict[args.ploidy],
                                 num_snps=args.numsnps,
                                 mean_counts=args.mcounts,
                                 bbdisp=args.bbdisp,
                                 nbdisp=args.nbdisp)

        lor = calculate_snp_method_lor(df=df,
                                       ff=args.ff,
                                       bbdisp=args.bbdisp)

        df_out.loc[len(df_out)] = [lor]

    df_out.to_csv(args.outfile, index=False)

def get_snp_seed(args):
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

    a_list = [args.ploidy,
              args.ff,
              args.perms,
              args.bbdisp,
              args.nbdisp,
              args.numsnps,
              args.mcounts]

    args_string = "".join([str(x) for x in a_list])

    # numpy RNG seed must be a 32-bit integer
    return int(hashlib.sha1(args_string).hexdigest(),16) & (2**32 -1)

def simulate_snp_sample(ff, H, num_snps, mean_counts, bbdisp, nbdisp):
    """
    Generate maternal, paternal, and fetal haplotypes for a chromosome, then
    sample reads according to these genotypes and the specified fetal fraction.

    Args:
        ff (float): Fetal fraction
        H (tuple): Fetal chromsosomal state hypothesis
        num_snps (int): Number of SNPs interrogated on the chromosome
        mean_counts (int): Mean sequencing counts (depth) per SNP
        bbdisp (float or 'binomial'): beta-binomial dispersion coefficient.
        nbdisp (float): The negative-binomial dispersion coefficient.

    Returns:
        A pandas dataframe containing the following columns:

        MAT_GENO: The maternal genotype (AA,AB,BB)
        A_COUNT: Counts of the A allele
        TOT_COUNTS: The total counts at the site
        POP_A_FRAC: The population frequency of the A allele

    Note:
        Possible values of H are (# mat chroms, # pat chroms, M(1) or M(2) nondisjunction):

            (1,1,None) - disomy
            (2,1,1) - maternal m1 nondisjunction
            (2,1,2) - maternal m2 nondisjunction
            (1,2,1) - paternal m1 nondisjunction
            (1,2,2) - paternal m2 nondisjunction

        If bbdisp = 'binomial', sample alleles from a binomial distribution

        If nbdisp = 1, the negative-binomial distribution collapses to the
        Poisson distribution. Therefore, sample counts-per-SNP from the
        Poisson distribution.

    Raises:
        AssertionError: If num_snps is not an integer
        AssertionError: If mean_counts is not an integer
        AssertionError: If ff <= 0 or ff >= 1
        AssertionError: If nbdisp <= 0 or nbdisp > 1
        AssertionError: If bbdisp <= 1 and bbdisp != 'binomial'
    """

    assert isinstance(num_snps, int), "num_snps must be an integer"
    assert isinstance(mean_counts, int), "mean_counts must be an integer"
    assert 0 < ff < 1, "Value of ff must be 0 < nbdisp < 1"
    assert 0 < nbdisp <= 1, "Value of nbdisp must be 0 < nbdisp <= 1"

    if bbdisp == 'binomial':
        bbdisp = np.inf
    else:
        assert bbdisp >= 1, "Value of bbdisp must be > 1 or 'binomial'"
        bbdisp = float(bbdisp)

    #Create an array of allele frequencies of length num_snps sampled from a
    #beta distribution with parameters mu=0.5, stdev=0.1.
    mu = 0.5
    sd = 0.1
    a, b = fit_beta_to_mu_sd(mu, sd)
    afs = np.random.beta(a=a, b=b, size=num_snps)

    #Generate parental haplotypes
    maternal_haplotypes = list(sample_genotype(afs))
    paternal_haplotypes = list(sample_genotype(afs))

    #Segregate haplotypes according to the chromosomal ploidy state, H.
    if (H[0] + H[1]) == 2: #Disomy
        m_hap = [np.random.choice(2)]
        p_hap = [np.random.choice(2)]
    elif H[2] == 1: #M1 nondisjunction - both homologous chromosomes
        if H[0] == 2: #Maternal
            m_hap = [0, 1]
            p_hap = [np.random.choice(2)]
        else:          #Paternal
            m_hap = [np.random.choice(2)]
            p_hap = [0, 1]
    elif H[2] == 2: #M2 nondisjunction - two copies of one homologous chrom.
        matched = np.random.choice(2)
        if H[0] == 2: #Maternal
            m_hap = [matched, matched]
            p_hap = [np.random.choice(2)]
        else:          #Paternal
            m_hap = [np.random.choice(2)]
            p_hap = [matched, matched]

    #Generate fetal haplotype
    fetal_haplotypes = []
    for egg, sperm in zip(maternal_haplotypes, paternal_haplotypes):
        fet_hap = [egg[x] for x in m_hap]
        fet_hap.extend([sperm[x] for x in p_hap])
        fetal_haplotypes.append("".join(fet_hap))

    #Sample counts of the 'A' allele based on the maternal and fetal genotypes, the
    #fetal fraction and the chromosomal ploidy.
    a_counts, tot_counts = sample_cfdna_reads(ff=ff,
                                              gml=maternal_haplotypes,
                                              gfl=fetal_haplotypes,
                                              H=H,
                                              mean_counts=mean_counts,
                                              bbdisp=bbdisp,
                                              nbdisp=nbdisp)

    #Returnt data as a pandas dataframe
    df = pd.DataFrame()
    df["MAT_GENO"] = maternal_haplotypes
    df["A_COUNT"] = a_counts
    df["TOT_COUNTS"] = tot_counts
    df["POP_A_FRAC"] = afs
    df["FET_GENO"] = fetal_haplotypes

    return df


def fit_beta_to_mu_sd(mu, sd):
    """
    Return the alpha and beta parameters that produce a beta distribution
    with specified mean and stdev.

    Args:
        mu (float): Mean value of the distribution
        sd (float): Standard deviation of the distribution

    Returns:
        alpha,beta

    Note:
        If sd > mu/2, returns values that produce bimodal distributions.

    Raises:
        AssertionError: If mu <= 0 or mu >= 1
        AssertionError: If sd <= 0 or sd >= 1
    """

    assert 0 < mu < 1, "The mean must be > 0 and < 1"
    assert 0 < sd < 1, "The sd must be > 0 and < 1"

    a = (((1-mu)/sd**2)-(1/mu)) * mu**2
    b = a * (1/mu - 1)

    return (a, b)

def sample_genotype(afs):
    """
    Given a list of SNP population allele fractions (frequency of A allele),
    return a list of SNP genotypes conforming to sampling from Hardy-Weinberg
    expectations.

    Args:
        afs (list): A list of population allele fractions per snp.

    Returns:
        A list of of phased genotype strings: ["AA","AB","BA","BB", ...]

    Note:
        Genotypes are phased (i.e., "AB" and "BA" are not equivalent) in order
        to properly segregate chromosomes to determine the fetal genotype.

    Raises:
        AssertionError: If any allele fraction < 0 or > 1
    """

    chrom = []
    for af in afs:
        assert 0 <= af <= 1, "Allele fractions must be 0 - 1, inclusive"
        genotypes = ['AA', 'AB', 'BA', 'BB']
        p, q = (af, 1-af) #Where P is A and q is B
        probs = [(p**2), p*q, p*q, (q**2)]
        x = np.random.choice(genotypes, p=probs)
        chrom.append("{}{}".format(x[0], x[1]))
    return chrom

def sample_cfdna_reads(ff, gml, gfl, H, mean_counts, bbdisp, nbdisp):
    """
    Given a known fetal fraction, maternal and child genotypes, and
    chromosomal state, return the number of A allele reads and the total counts
    at each site.

    Args:
        ff (float): fetal-fraction
        gml (list): list of maternal genotypes e.g. ["AA","AB",...]
        gfl (list): list of fetal genotypes e.g. ["AA","AB",...]
        H (tuple): Fetal ploidy state (#mat chr,#pat chr, meiotic stage of NDJ)
        mean_counts (int): Mean depth-per-SNP
        bbdisp (float): beta-binomial dispersion parameter in allele counts
        nbdisp (float): negative-binomial dispersion parameter in read depth

    Returns:
        (list of a_counts at each site, list of total counts at each site)

    Note:
        If bbdisp == np.inf the beta-binomial distribution collapses to the
        simple binomial.

        If nbdisp == 1 the negative-binomial distribution collapses to the
        Poisson distribution.

    Raises:
        AssertionError: if ff <= 0 or ff >= 1
        AssertionError: if gml is not a list
        AssertionError: if gfl is not a list
        AssertionError: if mean_counts is not int
        AssertionError: if nbdisp <= 0 or nbdisp > 1
        AssertionError: if bbdisp < 1
    """

    assert isinstance(gml, list), "gm must be a list of genotypes"
    assert isinstance(gfl, list), "gf must be list of genotypes"
    assert isinstance(H, tuple) and len(H) == 3, "H must be a tuple of length 3"
    assert isinstance(mean_counts, int), "mean_counts must be an integer"
    assert 0 < ff < 1, "Value of ff must be 0 < nbdisp < 1"
    assert 0 < nbdisp <= 1, "Value of nbdisp must be 0 < nbdisp <= 1"
    assert bbdisp >= 1, "Value of bbdisp must be > 1"

    mu = mean_counts
    number = len(gml)
    #Sample tot_counts according to a negative-binomial distribution when
    #0 < nbdisp < 1
    if nbdisp < 1:
        a = mu * nbdisp/float(1-nbdisp)
        tot_counts = np.random.negative_binomial(a, nbdisp, number)
    else: #If nbdisp == 1, collapse to the Poisson distribution
        tot_counts = np.random.poisson(lam=mu, size=number)


    a_counts = []
    for m, f, tot_count in zip(gml, gfl, tot_counts):

        if tot_count != 0:
            #Calculate probability of sampling the A allele given the genotypes and
            #fetal fraction.
            pA = get_pA_for_ff(gm=m,
                               gf=f,
                               nm=2,
                               nf=(H[0] + H[1]),
                               ff=ff,
                               H=H)

            #Sample counts of the A allele based on the beta-binomial distribution.
            a_count = sample_cfdna_allele(p=pA,
                                          tot_counts=tot_count,
                                          bbdisp=bbdisp)
            a_counts.append(a_count)

        else:
            a_counts.append(0)

    return (a_counts, tot_counts)

def get_pA_for_ff(gm, gf, nm, nf, ff, H):
    """
    Return the probability of sampling allele 'A' from a mother/child cfDNA
    mixture given the maternal genotype, fetal genotype, maternal ploidy,
    fetal ploidy, fetal fraction, and the chromosomal ploidy hypothesis,
    p(A|gm,gf,nm,nf,ff,H).

    Equation obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 32 [0345]

    Args:
        gm (tuple): maternal genotype
        gf (tuple): fetal genotype
        nm (int): maternal somy
        nf (int): fetal somy
        ff (float): fetal fraction
        H (tuple): Fetal ploidy state (#mat chr,#pat chr, meiotic stage of NDJ)

    Returns:
        The probability of sampling allele 'A'.

    Note:
        The equation presented in Rabinowitz et al. (2014) p. 32, section 0345
        is incorrect and does not properly reflect the maternal child mixture.
        This has been corrected in the code below.

    Raises:
        AssertionError: if gm is not a string of length 2
        AssertionError: if gf is not a string of length 2
        AssertionError: if nm is not an integer
        AssertionError: if nf is not an integer
        AssertionError: if ff < 0 or > 1
        AssertionError: if H is not a tuple of length 3
    """

    assert isinstance(gm, str) and len(gm) == 2, "gm must be a string of length 2"
    assert isinstance(gf, str) and 2 <= len(gf) <= 3, "gf must be a string of length 2 or 3"
    assert isinstance(nm, int), "nm must be an integer"
    assert isinstance(nf, int), "nf must be an integer"
    assert 0 < ff < 1, "Value of ff must be > 0  and < 1"
    assert isinstance(H, tuple) and len(H) == 3, "H must be a tuple of lenght 3"

    #Rearrange mirror genotypes
    gm = tuple(sorted(gm))
    gf = tuple(sorted(gf))

    Am = gm.count('A') #Number of As in maternal genotype
    Af = gf.count('A') #Number of As in fetal genotype
    ffc = get_corrected_ff(ff, H) #Correct ff for ploidy

    #prob = p(A|m,c,cf)
    prob = ((Am/float(nm)) * (1-ffc)) + ((Af/float(nf)) * ffc)

    return prob

def sample_cfdna_allele(p, tot_counts, bbdisp):
    """
    Sample reads at site from a beta-binomial distribution, where the
    probability of sampling the A allele is based on the proportion of the A
    allele in the maternal-fetal mixture drawn from a betabinomial distribution
    with dispersion parameter bbdisp. If the beta-binomial parameter is set to
    np.inf, then the beta-binomial distribution collapses to the binomial.

    Args:
        p (float): proportion of A allele in maternal-fetal mix
        tot_counts (int): the number of reads at the site (depth).
        bbdisp (float): beta-binomial dispersion parameter in allele counts

    Returns:
        The number of A allele reads.

    Note:
        If bbdisp == np.inf the beta-binomial distribution collapses to the
        simple binomial.

    Raises:
        AssertionError: If p < 0 or > 1
        AssertionError: If bbdisp < 1
    """

    assert bbdisp >= 1, "Value of bbdisp must be > 1"
    assert 0 <= p <= 1, "Value of p must be > 0 and < 1"

    #At probs 0 or 1 the beta parameters are undefined, therefore default to
    #sampling from a binomial distribution with p= 0 or 1. Always apply the
    #sampling variance from the most abundant allele.
    flip = False
    if p != 0 and p != 1 and bbdisp != np.inf:
        if p < 0.5:
            p = 1-p
            flip = True
        a = bbdisp
        b = (bbdisp / p) - bbdisp
        if flip:
            return tot_counts - pymc.rbetabin(a, b, tot_counts)
        else:
            return pymc.rbetabin(a, b, tot_counts)
    else:
        return np.random.binomial(tot_counts, p)


def read_pcg_df(infile):
    """
    Read in table corresponding to p(gf|gm,gp,H), or the probability of the
    fetal genotype, gf, given the maternal and paternal genotypes (gm, gp) and
    the chromosomal copy number hypothesis, H, and return as a dict.

    The table was obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 32 [0343]

    Args:
        infile (str): location of CSV file containing table.

    Returns:
        Dictionary of fetal genotype probabilities given the specified ploidy
        hypothesis.
    """

    pcg_dict = {}

    df = pd.read_csv(infile, header=0, comment="#")
    for _, r in df.iterrows():

        key = ("{}{}".format(r["MAT"][0], r["MAT"][1]),
               "{}{}".format(r["PAT"][0], r["PAT"][1]),
               r["TYPE"])

        pcg_dict[(key)] = {}
        for g in ["AA", "AB", "BB", "AAA", "AAB", "ABB", "BBB"]:
            pcg_dict[(key)][g] = r[g]

    return pcg_dict

def approximate_father_genotype(gf, gm, H, pcg, af):
    """
    Estimate p(gf|gm,gp,H), or the probability of the fetal genotype, gf,
    given the maternal and paternal genotypes (gm, gp) and the chromosomal
    copy number hypothesis, H, as the sum of of the probabilities of all
    potential paternal genotypes based on the population frequency of the A
    allele under Hardy-Weinberg equilibrium.

    The equation was obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 33 [0351]

    Args:
        gf (tuple): fetal genotype
        gm (tuple): maternal genotype
        H (tuple): Fetal ploidy state (#mat chr,#pat chr, meiotic stage of NDJ)
        pcg (dict): Probability of child genotype dict from read_pcg_df()
        af (float): The population allele frequency for the allele in question.

    Returns:
        The probability of the specified fetal genotype given the individual
        probabilities of the possible paternal genotypes.

    Raises:
        AssertionError: if gf is not tuple of length 2
        AssertionError: if gm is not tuple of length 2
        AssertionError: if H is not tuple of length 3
        AssertionError: if pcg is not dict
        AssertionError: if af is < 0 or > 1
    """

    assert isinstance(gm, str) and len(gm) == 2, "gm must be string of length 2"
    assert isinstance(gf, str) and 2 <= len(gf) <= 3, "gf must be string of length 2 or 3"
    assert isinstance(H, tuple) and len(H) == 3, "H must be a tuple of length 3"
    assert isinstance(pcg, dict), "pcg must be a dict"
    assert 0 < af < 1, "Value of af must be > 0 and < 1"

    genotypes = ("AA", "AB", "BB")
    p, q = (af, 1-af) #Generate AF priors, where P is A and q is B
    probs = [(p**2), 2*p*q, (q**2)]
    p_sum = 0
    H_to_nondisjunction_stage = {
        # (H[0], H[1], H[2])
        (1, 1, None): 'd',
        (2, 1, 1): 'm1',
        (2, 1, 2): 'm2',
        (1, 2, 1): 'p1',
        (1, 2, 2): 'p2',
    }

    for gp, prob in zip(genotypes, probs):
        p_sum += pcg[gm, gp, H_to_nondisjunction_stage[H]][gf] * prob

    return p_sum


def lA_for_genotype(a_count, depth, gm, gf, ff, H, bbdisp):
    """
    Calculate the log-likelihood of observing the read counts of individual
    alleles given the maternal and fetal genotypes (gm,gf), the fetal fraction,
    ff, the ploidy hypothesis, H, and the betabinomial dispersion parameter,
    bbdisp.

    The equation was obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 32 [0344,0345]

    Args:
        a_count (int): number of A allele reads
        depth (int): total number of reads at site
        gm (tuple): maternal genotype
        gf (tuple): fetal genotype
        ff (float): fetal fraction
        H (tuple): Fetal ploidy state (#mat chr,#pat chr, meiotic stage of NDJ)
        bbdisp (float): beta-binomial dispersion parameter in allele counts

    Returns:
        The log-likelihood of observing the specified number of A alleles given
        the total counts.

    Note:
        If bbdisp == np.inf the beta-binomial distribution collapses to the
        simple binomial.

    Raises:
        AssertionError: if a_count and depth are not INT or a_count > depth
        AssertionError: if gf is not a tuple of length 2
        AssertionError: if gm is not a tuple of length 2
        AssertionError: if ff is < 0 or > 1
        AssertionError: if H is not a tuple of length 3
        AssertionError: if bbdisp <= 1
    """

    assert isinstance(a_count, int), "a_count must be an integer"
    assert isinstance(depth, int), "depth must be an integer"
    assert a_count <= depth, "a_count must be <= depth"
    assert isinstance(gf, str) and 2 <= len(gf) <= 3, "gf must be string of length 2 or 3"
    assert isinstance(gm, str) and len(gm) == 2, "gm must be string of length 2"
    assert isinstance(H, tuple) and len(H) == 3, "H must be a tuple of length 3"
    assert 0 < ff < 1, "Value of ff must be > 0  and < 1"
    assert bbdisp >= 1, "Value of bbdisp must be > 1"

    #Get probability of sampling 'A' given the genotype
    pA = get_pA_for_ff(gm=gm,
                       gf=gf,
                       nm=2,
                       nf=(H[0] + H[1]),
                       ff=ff,
                       H=H)

    #Must account for 0 and 100% probability edge cases
    if pA == 0:
        pA = 0.00001
    elif pA == 1:
        pA = 0.99999

    #Mirror data so that high and low A counts are sampled from the same
    #absolute degree of dispersion.
    if bbdisp != np.inf:
        if pA < 0.5:
            a = bbdisp
            b = (bbdisp / (1-pA)) - bbdisp
            t = depth - a_count
            return pymc.betabin_like(t, a, b, depth)
        else:
            a = bbdisp
            b = (bbdisp / pA) - bbdisp
            return pymc.betabin_like(a_count, a, b, depth)
    else: #If bbdisp == np.inf, call directly from the binomial pmf
        return np.log(ss.binom.pmf(a_count, depth, pA))

def calculate_snp_method_lor(df, ff, bbdisp):
    """
    Given a dataframe containing the output of the SNP method simulation,
    the fetal fraction, and the beta-binomial dispersion parameter,
    calculate the log-odds ratio of a disomic fetus:

    LOR = log( L(Hd) / (1-L(Hd)) )

    Method obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 31-33 [0327 - 0351]

    Args:
        df (dataframe): pandas dataframe containing simulated sample
        ff (float): fetal fraction
        bbdisp (float or 'binomial'): beta-binomial dispersion parameter in allele counts

    Returns:
        The log-odds ratio of a disomic fetus.

    Note:
        The dataframe must contain the following columns:

        MAT_GENO: The maternal genotype (AA,AB,BB)
        A_COUNTS: Counts of the A allele
        TOT_COUNTS: The total depth at the site
        POP_A_FRAC: The population frequency of the A allele

    Raises:
        AssertionError: if df is not a pandas dataframe
        AssertionError: if ff is < 0 or > 1
        AssertionError: if bbdisp <= 1 or != "binomial"
    """

    assert isinstance(df, pd.core.frame.DataFrame), "df must be pandas dataframe"
    assert 0 < ff < 1, "Value of ff must be > 0  and < 1"

    if bbdisp == 'binomial':
        bbdisp = np.inf
    else:
        assert bbdisp >= 1, "Value of bbdisp must be > 1 or 'binomial'"
        bbdisp = float(bbdisp)

    #Read in the matrix of probabilities of observing the child genotype
    #given the parental genotypes and the hypothesis.
    #e.g p(c|mi,fi,H)
    pcg_dict = read_pcg_df("prob_child_genotype.csv")

    #Chromosomal state hypotheses (# of maternal copies,
    #                              # of paternal copies,
    #                              non-disjunction meiotic stage)
    hypotheses = [(1, 1, None), #Disomy
                  (2, 1, 1),
                  (2, 1, 2),
                  (1, 2, 1),
                  (1, 2, 2)]

    hyp_Lih = defaultdict(float) #Stores hypothesis log-likelihoods

    #Iterate through SNPs and determine the log-likelihood of it belonging to each
    #chromosomal hypothesis.
    for _, r in df.iterrows():

        gm = r["MAT_GENO"]
        a_count = r["A_COUNT"]
        depth = r["TOT_COUNTS"]
        af = r["POP_A_FRAC"]

        if depth == 0: #Ignore 0 depth sites
            continue

        #Sort genotypes as BA<->AB
        gm = "".join(sorted(list(gm)))

        for H in hypotheses:

            #Possible child genotypes are defined by ploidy state
            if (H[0] + H[1]) == 2: #Diploid
                fetal_genos = ("AA", "AB", "BB")

            else: #Trisomic
                fetal_genos = ("AAA", "AAB", "ABB", "BBB")

            #The SNP-wise log-likelihood is equal to:
            #log(sum( Pc_mfh * Lx_mccf ))

            #The SNP-level log-likelihood of each chromosomal hypothesis must
            #be integrated over all possible fetal genotypes
            cg_sum = 0
            for gf in fetal_genos:

                #Calculate p(gf|gm,gp,H), or the probability of the fetal
                #genotype, gf, given the maternal and paternal genotypes
                #(gm, gp) and the chromosomal copy number hypothesis, H.

                #As we do not know the paternal genotype a priori, we must
                #intergrate over each possible paternal genotype, weighed by
                #the population frequency of the A allele at the locus.
                Pgf_gmgpH = approximate_father_genotype(gf=gf,
                                                        gm=gm,
                                                        H=H,
                                                        pcg=pcg_dict,
                                                        af=af)

                #Calculate p(data|H,gf,gm,ff), or the log-likelihood of
                #observing the data given the hypothesis, the maternal and
                #fetal genotypes and the fetal fraction.
                Lx_gmgfff = lA_for_genotype(a_count=a_count,
                                            depth=depth,
                                            gm=gm,
                                            gf=gf,
                                            ff=ff,
                                            H=H,
                                            bbdisp=bbdisp)

                cg_sum += Pgf_gmgpH * np.exp(Lx_gmgfff)

            hyp_Lih[H] += np.log(cg_sum)

    norm = logsumexp(hyp_Lih.values())
    Ph = np.exp(hyp_Lih[(1, 1, None)] - norm)
    lor = np.log(Ph/float(1-Ph))

    return lor

def get_corrected_ff(ff, H):
    """
    Return the expected ff ('corrected fetal fraction') for a given chromosome
    based on the chromosomal ploidy hypothesis, H.

    Equation obtained from:

    Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al.
    Methods for non-invasive prenatal ploidy calling.
    US Patent Application 14/179,399, 2014. p. 32 [0345]

    Args:
        ff (float): fetal fraction
        H (tuple): Fetal ploidy state (#mat chr,#pat chr, meiotic stage of NDJ)

    Returns:
        The fetal fraction corrected for chromosomal ploidy.

    Note:
        In a normal disomy, the corrected fetal fraction = fetal fraction.

    Raises:
        AssertionError: if ff < 0 or > 1
        AssertionError: if H is not a tuple of length 3
    """
    assert 0 < ff < 1, "Value of ff must be > 0  and < 1"
    assert isinstance(H, tuple) and len(H) == 3, "H must be a tuple of lenght 3"

    nf = H[0] + H[1]  #Somy of child at chromosome.
    nm = 2              #Somy of mother fixed at 2
    ffc = ff * nf /(nm * (1-ff) + nf * ff)
    return ffc

if __name__ == '__main__':
    main()
