import argparse
import pandas as pd
import os, glob
import subprocess
from pathlib import Path
import numpy as np

import ipdb


def main(args):

    plink_results = pd.read_csv(args.input_file, sep='\t')

    # Remove variants that are essentially identical by various metrics (more likely to be in LD with each other)
    plink_results = plink_results.drop_duplicates(['A1_FREQ', 'OBS_CT', 'BETA', 'SE', 'T_STAT'])
    filt_results = plink_results[(plink_results['BETA'].abs() > args.beta_cutoff) & (plink_results['P'] < args.p_cutoff)]

    filt_results = filt_results[['ID', 'BETA', 'P']]

    filt_results.to_csv('{}_filtered_results.csv'.format(args.output_prefix), sep='\t', index=False)

    filt_results['ID'].to_csv('{}_filtered_snplist.txt'.format(args.output_prefix), index=False, header=None)

    print("Completed!")

    ipdb.set_trace()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Plink results to pull variants')

    parser.add_argument("--input-file", type=str, default=None,
            help="input file with the Plink summary statistics")

    parser.add_argument("--p-cutoff", type=float, default=1e-100,
            help="P-value cutoff for variant selection")

    parser.add_argument("--beta-cutoff", type=float, default=0,
            help="Beta-value cutoff for variant selection (absolute value, greater means less variants)")

    parser.add_argument("--output-prefix", type=str, default='maize_trait_selected',
            help="Output prefix for the resulting files (variant list alone, variant list with effect sizes, hybrid to score calculation)")


    args = parser.parse_args()

    
    print(args)
    print()
    print("Parsing file now...")
    main(args)