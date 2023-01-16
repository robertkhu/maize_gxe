import pandas as pd
import matplotlib.pyplot as plt

import ipdb

orig_good_sub_1 = pd.read_csv('../../data/Testing_Data/maize_gxe_submission_1.csv')
bad_sub_2 = pd.read_csv('../../data/Testing_Data/maize_gxe_submission_2.csv')
hgbr_new_sub_pend = pd.read_csv('../overall_model_analysis/hist_gbr_yields_fixed/predicted_yield_vals.csv')
complex_sub_pend = pd.read_csv('../../data/Testing_Data/maize_gxe_submission_complexmodels.csv')
fullyimputed_sub_pend = pd.read_csv('../overall_model_analysis/imputed_results_full/predicted_yield_vals.csv')
fullyimputed_prune_sub_pend = pd.read_csv('../overall_model_analysis/imputed_results_full_modelprune/predicted_yield_vals.csv')
fullyimputed_featureselect_sub_pend = pd.read_csv('../overall_model_analysis/imputed_results_full_modelprune/predicted_yield_vals.csv')

all_subs = orig_good_sub_1.merge(bad_sub_2, on=['Env', 'Hybrid'])
all_subs = all_subs.merge(hgbr_new_sub_pend,  on=['Env', 'Hybrid'])
all_subs = all_subs.rename(columns={'Yield_Mg_ha_x': 'good_sub_1', 'Yield_Mg_ha_y': 'bad_sub_2', 'Yield_Mg_ha': 'hbgr_new_sub_pend'})

all_subs = all_subs.merge(complex_sub_pend,  on=['Env', 'Hybrid'])
all_subs['complex_sub_pend'] = all_subs['Yield_Mg_ha']
all_subs = all_subs.drop(columns=['Yield_Mg_ha'])

all_subs = all_subs.merge(fullyimputed_sub_pend,  on=['Env', 'Hybrid'])
all_subs['fullyimputed_sub_pend'] = all_subs['Yield_Mg_ha']
all_subs = all_subs.drop(columns=['Yield_Mg_ha'])

all_subs = all_subs.merge(fullyimputed_prune_sub_pend,  on=['Env', 'Hybrid'])
all_subs['fullyimputed_prune_bad_sub_3'] = all_subs['Yield_Mg_ha']
all_subs = all_subs.drop(columns=['Yield_Mg_ha'])

ipdb.set_trace()


currCols = ['good_sub_1', 'bad_sub_2', 'fullyimputed_prune_bad_sub_3', 'complex_sub_pend', 'hbgr_new_sub_pend']

hist_handles = [all_subs[col].hist(bins=20) for col in currCols]

plt.gca().legend(currCols, loc='upper left')
plt.show()

