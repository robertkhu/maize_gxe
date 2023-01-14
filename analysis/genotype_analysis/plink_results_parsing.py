import argparse
import pandas as pd
import os, glob
import subprocess
import numpy as np

import matplotlib.pyplot as plt

import ipdb


def main(args):

    plink_results = pd.read_csv(args.input_file, sep='\t')

    # Remove variants that are essentially identical by various metrics (more likely to be in LD with each other)
    plink_results = plink_results.drop_duplicates(['A1_FREQ', 'OBS_CT', 'BETA', 'SE', 'T_STAT'])
    filt_results = plink_results[(plink_results['BETA'].abs() > args.beta_cutoff) & (plink_results['P'] < args.p_cutoff)]

    filt_results = filt_results[['ID', 'BETA', 'P']]

    # Output some files with data
    filt_results.to_csv('{}_filtered_results.csv'.format(args.output_prefix), sep='\t', index=False)
    filt_results['ID'].to_csv('{}_filtered_snplist.txt'.format(args.output_prefix), index=False, header=None)


    # Use the filtered file to run Plink and get the matrix of genotypes to variants (smaller)
    plink_cmd = ['plink', 
        '--bfile', args.input_bfile_prefix, 
        '--recode', 'A', 
        '--out', args.output_prefix, 
        '--extract', '{}_filtered_snplist.txt'.format(args.output_prefix)]

    subprocess.run(plink_cmd)
    #plink --bfile 5_Genotype_Data_All_Years --recode A --out maize_results_test --extract maize_trait_selected_filtered_snplist.txt

    # '{}.raw'.format(args.output_prefix) is where the matrix of samples to genotypes will be

    # All credits to Rasika for writing the below
    sample_geno_matrix = pd.read_csv('{}.raw'.format(args.output_prefix), sep='\t')
    geno_effect_matrix = filt_results[['ID, BETA']].set_index('ID')

    sample_geno_matrix = sample_geno_matrix.drop('PAT', 'MAT', 'SEX', 'PHENOTYPE', 'FID').fillna(0)

    sample_prs = np.matmul(sample_geno_matrix, geno_effect_matrix)
    sample_prs.to_csv('{}_prs_scores.txt'.format(args.output_prefix), header=None)

    print("Completed!")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Plink results to pull variants')

    parser.add_argument("--input-file", type=str, default=None,
            help="input file with the Plink summary statistics")

    parser.add_argument("--input-bfile-prefix", type=str, default=None,
            help="input file with the actual raw Plink data")

    parser.add_argument("--p-cutoff", type=float, default=1e-100,
            help="P-value cutoff for variant selection")

    parser.add_argument("--beta-cutoff", type=float, default=0,
            help="Beta-value cutoff for variant selection (absolute value, greater means less variants)")

    parser.add_argument("--output-prefix", type=str, default='maize_trait_selected',
            help="Output prefix for the resulting files (variant list alone, variant list with effect sizes, hybrid to score calculation)")


    args = parser.parse_args()

    # Example run:
    # python plink_results_parsing.py --input-file maize_results_test.Yield_Mg_ha.glm.linear --input-bfile-prefix 5_Genotype_Data_All_Years --p-cutoff 1e-100 --beta-cutoff 0 --output-prefix maize_trait_selected

    
    print(args)
    print()
    print("Parsing file now...")
    main(args)