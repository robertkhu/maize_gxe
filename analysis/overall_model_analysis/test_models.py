import argparse
import pandas as pd
import numpy as np
import os, glob, copy
import random

import ipdb

# Model imports
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNet, Lars, \
    LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, PoissonRegressor, GammaRegressor, \
    SGDRegressor, PassiveAggressiveRegressor

# Various imports for splitting, data processing, etc.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Various helper functions
from sklearn.base import clone
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from tqdm import tqdm




def get_various_metrics(y_true, y_test, print_full_report=True):
    metrics = {}

    # Using Sklearn to get a variety of metrics for multiclass problems
    metrics['mse'] = sklearn.metrics.mean_squared_error(y_true, y_test)
    metrics['rmse'] = np.sqrt(metrics['mse'])

    # if(print_full_report):
    #     print(sklearn.metrics.classification_report(y_true, y_test))

    return metrics

@ignore_warnings(category=ConvergenceWarning)
def train_and_analyze_model(model, X_train, y_train, X_test, y_test, seed, verbose=False):
    model.random_state=seed
    model.fit(X_train, y_train)

    model_metrics = get_various_metrics(y_test, model.predict(X_test), print_full_report=verbose)

    if(verbose):
        print("Acc: {:0.4f}".format(model_metrics['acc']))
        print("Bal_Acc: {:0.4f}".format(model_metrics['bal_acc']))
        
        # print("Confusion matrix: ")
        # print('\n'.join([''.join(['\t{}'.format(str(cell)) for cell in row]) for row in model_metrics['confusion']]))

    return model, model_metrics


def main(args):
    
    # Load data
    ml_data = pd.read_csv(args.input_file)

    # Get actual feature data from the array (TODO: which columns?)
    ml_data_features = ml_data.iloc[:, 7:38]

    # Select specific features from the data?
    ml_data_features = ml_data_features.iloc[:, np.r_[0:7, 16, 23:26, 28:29]]

    # Get the yield values for each hybrid
    yield_vals = ml_data['yield']



    # Model definitions and testing

    models = {'linear_regression': LinearRegression(),
                'ridge_cv': RidgeCV(),
                'lasso': Lasso(),
                'elastic': ElasticNet(),
                'lars': Lars(),
                'lasso_lars': LassoLars(),
                'orthogonal_mp': OrthogonalMatchingPursuit(),
                'bayesian_ridge': BayesianRidge(),
                'ard': ARDRegression(),
                'poisson': PoissonRegressor(),
                'gamma': GammaRegressor(),
                'sgd': SGDRegressor(),
                'passive_aggressive': PassiveAggressiveRegressor()}



    iter_metrics = {}
    model_iter = {}

    print("Starting model iterations (seeds {} to {}, inclusive)...".format(args.seed, args.seed+args.num_iters-1))
    print("")


    for seed_iter in tqdm(range(args.seed, args.seed+args.num_iters)):
        #print("Iteration {}".format(seed_iter+1), end='\r')

        # Get training and testing data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(ml_data_features.values, sevo_codes.values, test_size=0.2, random_state=seed_iter)

        # Scale the data and such
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        for mname, model in tqdm(models.items(), leave=False):
            if args.verbose:
                print("Training and fitting {} now...".format(mname))

            model_fitted, metrics = train_and_analyze_model(clone(model), X_train, y_train, X_test, y_test, seed=seed_iter, verbose=args.verbose)
            model_iter[mname] = {'model': model_fitted, 'metrics': metrics}
            
            if args.verbose:
                print('-------------------\n')


        iter_metrics[seed_iter] = copy.deepcopy(model_iter)


    # Aggregate iteration metrics using Pandas and dictionaries
    agg_df = pd.DataFrame.from_dict(iter_metrics, orient='index')

    agg_dict = {}
    raw_agg_dict = {}

    for model_name in agg_df.columns:
        raw_agg_dict[model_name] = {}
        agg_dict[model_name] = {}
        model_col = agg_df[model_name]
        
        for metric in model_col[args.seed]['metrics']:
            metric_agg_vals = [d['metrics'][metric] for d in model_col.values]

            agg_dict[model_name]['{}_mean'.format(metric)] = np.mean(metric_agg_vals)
            agg_dict[model_name]['{}_std'.format(metric)] = np.std(metric_agg_vals)

            raw_agg_dict[model_name][metric] = metric_agg_vals



    final_df_metrics_agg = pd.DataFrame.from_dict(agg_dict, orient='index')
    final_df_metrics_agg.to_csv(os.path.join(args.output_dir, 'models_iter_mean_std.csv'))

    final_df_metrics_raw = pd.DataFrame.from_dict(raw_agg_dict, orient='index')
    final_df_metrics_raw.to_csv(os.path.join(args.output_dir, 'models_iter_raw.csv'))


    for metric in final_df_metrics_raw.columns:
        metric_num_arrays = [np.array(x) for x in final_df_metrics_raw[metric].values]
        metric_boxplot = sns.boxplot(metric_num_arrays)

        metric_boxplot.set_xticklabels(final_df_metrics_raw.index.str.upper())

        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.title("Boxplot of Model Performances")

        plt.savefig(os.path.join(args.output_dir, '{}_plot.png'.format(metric)))

        plt.show()



    ipdb.set_trace()







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse the input data file and test features on different models.')

    parser.add_argument("--input-file", type=str, 
            default="../data/data_info_feature_table_with_artifactflags_updated_11-07-2022.csv",
            help="Input file to use for the analysis.")

    parser.add_argument("--output-dir", type=str, 
            default='../out_dir/',
            help="Output for model testing and data thereof (if any).")

    parser.add_argument("--seed", type=int, 
            default=9,
            help="Seed for random seed setting (starting point if iterating).")

    parser.add_argument("--num-iters", type=int,
            default=5,
            help="Number of iterations to do.")

    parser.add_argument("--verbose", action='store_true', 
            help="Whether to print metrics for each fit of each model.")


    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    
    print(args)
    print()
    print("Parsing file and analyzing with model...")
    main(args)

