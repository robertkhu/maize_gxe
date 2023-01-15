import argparse
import pandas as pd
import numpy as np
import os, glob, copy
import random

import ipdb

# Base model imports
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, Lars, \
    LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, PoissonRegressor, GammaRegressor, \
    SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor

# Ensemble model imports (standalone)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

# Ensemble model imports (wrapping a base estimator)
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor

# Various imports for splitting, data processing, etc.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Various helper functions
from sklearn.base import clone
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance

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
@ignore_warnings(category=FutureWarning)
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
    ml_data = pd.read_csv(args.input_file, compression='gzip', encoding = "ISO-8859-1")

    # Pull out the real training data and the real eval data based on the existence of yield
    ml_data_train = ml_data[~ml_data['Yield_Mg_ha'].isna()].replace('', pd.NA)
    ml_data_eval = ml_data[ml_data['Yield_Mg_ha'].isna()].replace('', pd.NA)

    # Temporary, TODO: use a proper dataframe with little-no NAs for training and only numeric values in each column
    ml_data_train = ml_data_train.drop_duplicates(['Env', 'Hybrid'])
    #ml_data_train_num = ml_data_train.select_dtypes(include=[np.number]).dropna()
    ml_data_train_num = ml_data_train.select_dtypes(include=[np.number]) # don't drop missing for Hist Gradient Boosting


    # Get actual feature data from the array (TODO: which columns?)

    # Get the yield values for each hybrid
    train_yield_vals = ml_data_train_num['Yield_Mg_ha']


    # True test data
    # TODO: Pull from the ml_data above and get the features for the evaluation data
    ml_data_eval_num = ml_data_eval.select_dtypes(include=[np.number])
     

    # If we have feature selection files, then use those features
    if(args.feature_selection_file):
        feature_cols_df = pd.read_csv(args.feature_selection_file)
        feature_cols_df = feature_cols_df.sort_values(['r2'], ascending=False)

        if(args.remove_weather):
            feature_col_type_origin = pd.read_csv(args.column_type_file)
            weather_feature_df = feature_col_type_origin[['column_names', 'weather']]
            weather_feature_df = weather_feature_df[weather_feature_df['weather']]

            weather_cols_to_remove = weather_feature_df['column_names']

            feature_cols_df = feature_cols_df[~feature_cols_df['Y'].isin(weather_cols_to_remove)]

        keep_cols = feature_cols_df['Y'].head(args.top_features_to_select).values
        keep_cols = np.append(keep_cols, ['PRS10', 'PRS50', 'PRS100', 'PRS200'])
        keep_cols = keep_cols.tolist()

        ml_train_features = ml_data_train_num[keep_cols]
        ml_eval_features = ml_data_eval_num[keep_cols]


    else:
        #ml_train_features = ml_data_train_num.iloc[:, np.r_[1:18, 300:len(ml_data_train_num.columns)]] #MWES including MES
        #ml_train_features = ml_data_train_num.iloc[:, np.r_[1:18, 536:len(ml_data_train_num.columns)]] # MWES including MS
        #ml_train_features = ml_data_train_num.iloc[:, np.r_[1:18]] #MWES including M

        #ml_eval_features = ml_data_eval_num.iloc[:, np.r_[1:18, 300:len(ml_data_eval_num.columns)]] #MWES including MWES
        #ml_eval_features = ml_data_eval_num.iloc[:, np.r_[1:18, 536:len(ml_data_eval_num.columns)]] # MWES including MWS
        #ml_eval_features = ml_data_eval_num.iloc[:, np.r_[1:18]] #MWES including M

        ml_train_features = ml_data_train_num.iloc[:, np.r_[1:len(ml_data_train_num.columns)]]
        ml_eval_features = ml_data_eval_num.iloc[:, np.r_[1:len(ml_data_eval_num.columns)]]

        print("Skipped feature selection; did not select any other columns")





    # don't drop missing for Hist Gradient Boosting
    if(len(ml_eval_features) != len(ml_eval_features.dropna())):
        print("THIS FILE HAS TEST DATA MISSING DATA IN COLUMNS")
        ipdb.set_trace()


    scaler = MinMaxScaler().set_output(transform='pandas')
    ml_train_features = scaler.fit_transform(ml_train_features)
    ml_eval_features = scaler.transform(ml_eval_features)

    #ipdb.set_trace()


    # Model definitions and testing
    # For early stopping we'll use ‘neg_root_mean_squared_error’ as scoring (if supported)

    models = {
                'hist_gbr_iter100k': HistGradientBoostingRegressor(max_iter=100000),
            }



    iter_metrics = {}
    model_iter = {}


    print("Starting model iterations (seeds {} to {}, inclusive)...".format(args.seed, args.seed+args.num_iters-1))
    print("")


    for seed_iter in tqdm(range(args.seed, args.seed+args.num_iters)):
        #print("Iteration {}".format(seed_iter+1), end='\r')

        # Get training and testing data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(ml_train_features.values, train_yield_vals.values, test_size=0.2, random_state=seed_iter)

        # Scale the data and such
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)


        for mname, model in tqdm(models.items(), leave=True):
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

        metric_boxplot.set_xticklabels(final_df_metrics_raw.index.str.upper(), rotation=45)

        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.title("Boxplot of Model Performances")

        plt.savefig(os.path.join(args.output_dir, '{}_plot.png'.format(metric)))

        #plt.show()
        plt.close()



    # Predict on the true testing set, for the purposes of submission
    # Average across all the trained models for a model type, then average across the models

    all_models = []
    all_model_iter_preds = pd.DataFrame()
    all_model_iter_rmses = []
    avg_model_preds = pd.DataFrame()
    avg_model_rmses = {}

    for mname, model in tqdm(models.items(), leave=False):
        curr_model_preds = pd.DataFrame()
        curr_model_rmses = []

        for seed_iter in tqdm(range(args.seed, args.seed+args.num_iters), leave=False):
            iter_model = iter_metrics[seed_iter][mname]['model']
            curr_model_rmse = iter_metrics[seed_iter][mname]['metrics']['rmse']

            curr_model_pred = iter_model.predict(ml_eval_features.values)

            curr_model_preds[seed_iter] = curr_model_pred
            curr_model_rmses.append(curr_model_rmse)

            all_model_iter_preds['{}_{}'.format(mname, seed_iter)] = curr_model_pred
            all_model_iter_rmses.append(curr_model_rmse)
            all_models.append(iter_model)

        avg_model_preds[mname] = curr_model_preds.mean(axis=1)
        avg_model_rmses[mname] = np.mean(curr_model_rmses)



    # ipdb.set_trace()


    out_df = ml_data_eval[['Env', 'Hybrid']].copy()
    out_df['Yield_Mg_ha'] = avg_model_preds.mean(axis=1).values

    out_df.to_csv(os.path.join(args.output_dir, 'predicted_yield_vals.csv'), index=False)


    ipdb.set_trace()







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse the input data file and test features on different models.')

    parser.add_argument("--input-file", type=str, 
            default="../data/data_info_feature_table_with_artifactflags_updated_11-07-2022.csv",
            help="Input file to use for the analysis.")

    parser.add_argument("--output-dir", type=str, 
            default='./out_dir/',
            help="Output for model testing and data thereof (if any).")

    parser.add_argument("--seed", type=int, 
            default=9,
            help="Seed for random seed setting (starting point if iterating).")

    parser.add_argument("--num-iters", type=int,
            default=10,
            help="Number of iterations to do.")

    parser.add_argument("--verbose", action='store_true', 
            help="Whether to print metrics for each fit of each model.")

    parser.add_argument("--feature-selection-file", type=str,
            default=None,
            help="Perform feature selection on the models (that is, select features based on Robert's files)")
    
    parser.add_argument("--top-features-to-select", type=int,
            default=20,
            help="Select the top 20 features from the file above to keep.")

    parser.add_argument("--column-type-file", type=str,
            default='../feature_column_separation.csv',
            help='A file that has info on which file the features came from')

    parser.add_argument("--remove-weather", action='store_true',
            help='Arg to remove weather features')


    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    
    print(args)
    print()
    print("Parsing file and analyzing with model...")
    main(args)

