#!/bin/bash
#This shell-script contains the commands used to generate the data analyzed
#in the manuscript.

CODEDIR="./"
SIMDIR="../simulation_output/"

# Default SNP method simulation parameters
DEFAULT_BETABIN=1000
DEFAULT_NEGBIN=0.0005
DEFAULT_SNPS=3348
DEFAULT_FETAL_FRACTIONS=`seq 0.001 0.001 0.04`
SUPP_FETAL_FRACTIONS="0.010 0.020 0.030 0.040"


###
# WGS Method
###

emit_all_wgs_commands() {
    #Calculate sensitivity/specificity for the WGS method for T21, T18, and T13,
    #over fetal fractions 0.001 - 0.04 in 0.001 increments based on 10,000
    #permutations of the data and simulating variance in bin depths according to
    #Poisson expectations (Figure 3D).
    OUTFILE="${SIMDIR}wgs_method_10000affected.csv"
    echo "$OUTFILE: ${CODEDIR}wgs_method_simulation.py"
    echo -e "\tpython -O ${CODEDIR}wgs_method_simulation.py -o $OUTFILE -e"

    #As above but varying the degree of negative binomial dispersion (Figure S1).
    for nb in 0.1 0.333 0.666
    do
        OUTFILE="${SIMDIR}wgs_method_10000affected_nb${nb}.csv"
        echo "$OUTFILE: ${CODEDIR}wgs_method_simulation.py"
        echo -e "\tpython -O ${CODEDIR}wgs_method_simulation.py -o $OUTFILE -n $nb -e"
    done

    #Perform 10,000 permutations of the WGS method over fetal fractions 0.001 -
    #0.04 in 0.001 increments, however output a CSV file containing all permuted
    #z-scores rather than the calculated sensitivity and specifcity statistics for
    #the purpose of calculating ROC curves and AUC metrics (Figure 3A)
    OUTFILE="${SIMDIR}wgs_method_10000affected_for_AUCs.csv"
    echo "$OUTFILE: ${CODEDIR}wgs_method_simulation.py"
    echo -e "\tpython -O ${CODEDIR}wgs_method_simulation.py -o $OUTFILE -z -e"

}
###
# SNP Method
###

generate_snp_method_outdir() {
    local BETABIN=$1
    local NEGBIN=$2
    local SNPS=$3

    local OUTDIR="${SIMDIR}snp_method"
    if [ "$BETABIN" == "binomial" ]; then
        OUTDIR="${OUTDIR}_binomial"
    else
        OUTDIR="${OUTDIR}_bbdisp${BETABIN}"
    fi
    if [ "$NEGBIN" == "poisson" ]; then
        OUTDIR="${OUTDIR}_poisson"
    else
        OUTDIR="${OUTDIR}_nbdisp${NEGBIN}"
    fi
    if [ "$SNPS" == $DEFAULT_SNPS ]; then
        OUTDIR="${OUTDIR}"
    else
        OUTDIR="${OUTDIR}_snps${SNPS}"
    fi

    echo $OUTDIR
}
emit_directory_rule() {
    DIRNAME="$1"
    echo "$DIRNAME:"
    echo -e "\tmkdir -p $DIRNAME"
}
emit_snp_method_command() {
    local GENO=$1
    local FF=$2
    local BETABIN=$3
    local NEGBIN=$4
    local SNPS=$5

    local OUTDIR=`generate_snp_method_outdir $BETABIN $NEGBIN $SNPS`

    if [ "$SNPS" == $DEFAULT_SNPS ]; then
        local SNPARG=""
    else
        local SNPARG="-s ${SNPS}"
    fi

    local OUTFILE="snp_${GENO}_ff${FF}_10000.csv"
    echo "${OUTDIR}/${OUTFILE}: ${OUTDIR} ${CODEDIR}snp_method_simulation.py"
    echo -e "\tpython -O ${CODEDIR}snp_method_simulation.py -p $GENO -f $FF -o ${OUTDIR}/${OUTFILE} -b $BETABIN -n $NEGBIN -e $SNPARG"
}
generate_snp_commands() {
    local BETABIN=$1
    local NEGBIN=$2
    local SNPS=$3
    shift
    shift
    shift

    local OUTDIR=`generate_snp_method_outdir $BETABIN $NEGBIN $SNPS`
    emit_directory_rule $OUTDIR
    for FF in "$@"; do
        for GENO in d m1 m2 p1 p2; do
            emit_snp_method_command $GENO $FF $BETABIN $NEGBIN $SNPS
        done
    done
}

emit_all_snp_commands() {
    #Generate log-odds ratios for 10,000 SNP method samples for each nondisjunction
    #event origin over fetal fractions 0.001 - 0.04 in 0.001 increments using
    #default simulation parameters (Figure 3B,C,D).

    generate_snp_commands $DEFAULT_BETABIN $DEFAULT_NEGBIN $DEFAULT_SNPS $DEFAULT_FETAL_FRACTIONS

    # Supplemental figure data: LORs for 10000 SNP method samples for each
    # nondisjunction at fetal fractions of 1%, 2%, 3%, 4% with:

    #      bbdisp values 100, 10000, and using binomial sampling (Figure S3).
    for betabin in 100 10000 binomial; do
        generate_snp_commands $betabin $DEFAULT_NEGBIN $DEFAULT_SNPS $SUPP_FETAL_FRACTIONS
    done

    #      nbdisp values 0.005, 0.05, and 1 (Poisson) (Figure S4)
    for negbin in 1 0.05 0.005; do
        generate_snp_commands $DEFAULT_BETABIN $negbin $DEFAULT_SNPS $SUPP_FETAL_FRACTIONS
    done
    #      Number of SNPs on the interrogated chromosome as 1000, 6000, and 10000 (Figure S5).
    for snps in 1000 6000 10000; do
        generate_snp_commands $DEFAULT_BETABIN $DEFAULT_NEGBIN $snps $SUPP_FETAL_FRACTIONS
    done
}


# Get the targets generated
WGS_TARGETS=`emit_all_wgs_commands | grep : | cut -f1 -d: | tr '\n' ' '`
SNP_TARGETS=`emit_all_snp_commands | grep : | cut -f1 -d: | tr '\n' ' '`
echo "all: $WGS_TARGETS $SNP_TARGETS"
emit_all_wgs_commands
emit_all_snp_commands
