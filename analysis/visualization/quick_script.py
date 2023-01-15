import pandas as pd
import matplotlib.pyplot as plt


orig_good_sub_1 = pd.read_csv('../../data/Training_Data/maize_gxe_submission_1.csv')
bad_sub_2 = pd.read_csv('../../data/Training_Data/maize_gxe_submission_2.csv')
hgbr_new_sub_pend = pd.read_csv('../overall_model_analysis/hist_gbr_yields_fixed/predicted_yield_vals.csv')
complex_sub_pend = pd.read_csv('../../data/maize_gxe_submission_complexmodels.csv')

all_subs = orig_good_sub_1.merge(bad_sub_2, on=['Env', 'Hybrid'])
all_subs = all_subs.merge(hgbr_new_sub_pend,  on=['Env', 'Hybrid'])

all_subs = all_subs.rename(columns={'Yield_Mg_ha_x': 'good_sub_1', 'Yield_Mg_ha_y': 'bad_sub_2', 'Yield_Mg_ha': 'hbgr_new_sub_pend'})

all_subs = all_subs.merge(complex_sub_pend,  on=['Env', 'Hybrid'])

all_subs = all_subs.rename({'Yield_Mg_ha': 'complex_sub_pend'})


currCols = ['good_sub_1', 'bad_sub_2', 'hbgr_new_sub_pend']

hist_handles = [all_subs[col].hist(bins=20) for col in currCols]

plt.gca().legend(currCols, loc='upper left')

plt.show()