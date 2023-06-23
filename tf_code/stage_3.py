# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu

# %% READ CSVs OF METRICS FOR EACH TECHNIQUE

results_folder = './results_imagenet/'

vanilla_csv_path = results_folder + 'vanilla_results.csv'
integrated_csv_path = results_folder + 'integrated_results.csv'
smooth_vanilla_csv_path = results_folder + 'smooth_vanilla_results.csv'
smooth_integrated_csv_path = results_folder + 'smooth_integrated_results.csv'
rise_vanilla_csv_path = results_folder + 'rise_vanilla_results.csv'
rise_integrated_csv_path = results_folder + 'rise_integrated_results.csv'

columns=['prd', 'imp', 'dlt']

# load dataframes with results
vanilla_results_df = pd.read_csv(vanilla_csv_path, header=None, names=columns)
integrated_results_df = pd.read_csv(integrated_csv_path, header=None, names=columns)
smooth_vanilla_results_df = pd.read_csv(smooth_vanilla_csv_path, header=None, names=columns)
smooth_integrated_results_df = pd.read_csv(smooth_integrated_csv_path, header=None, names=columns)
rise_vanilla_results_df = pd.read_csv(rise_vanilla_csv_path, header=None, names=columns)
rise_integrated_results_df = pd.read_csv(rise_integrated_csv_path, header=None, names=columns)

# %% PREPARE FOR COMPUTING RATIOS

# keep rows that do not contain any zeros
# but based on the vanilla results dataframe
# because it is going to be the demoninator

smooth_vanilla_results_df = smooth_vanilla_results_df.loc[(vanilla_results_df != 0.0).all(axis=1)]

rise_vanilla_results_df = rise_vanilla_results_df.loc[(vanilla_results_df != 0.0).all(axis=1)]

vanilla_results_df = vanilla_results_df.loc[(vanilla_results_df != 0.0).all(axis=1)]

# keep rows that do not contain any zeros
# but based on the integrated results dataframe
# because it is going to be the demoninator

smooth_integrated_results_df = smooth_integrated_results_df.loc[(integrated_results_df != 0.0).all(axis=1)]

rise_integrated_results_df = rise_integrated_results_df.loc[(integrated_results_df != 0.0).all(axis=1)]

integrated_results_df = integrated_results_df.loc[(integrated_results_df != 0.0).all(axis=1)]

# %% COMPUTE METRIC RATIOS

def compute_ratios_df(wrapper_results_df, helper_results_df):
    ratios_df = wrapper_results_df / helper_results_df
    return ratios_df

smooth_vanilla_ratios_df = compute_ratios_df(smooth_vanilla_results_df, vanilla_results_df)

smooth_integrated_ratios_df = compute_ratios_df(smooth_integrated_results_df, integrated_results_df)

rise_vanilla_ratios_df = compute_ratios_df(rise_vanilla_results_df, vanilla_results_df)

rise_integrated_ratios_df = compute_ratios_df(rise_integrated_results_df, integrated_results_df)

# %% SHAPIRO-WILK TEST

# assert that ALL of the ratios of any metric of any technique 
# do NOT follow NORMAL distribution which is confirmed 
# when p-value of Shapiro-Wilk test is LESS THAN 0.05 

for name, ratio_df in {
    'smooth_vanilla': smooth_vanilla_ratios_df, 
    'smooth_integrated': smooth_integrated_ratios_df, 
    'rise_vanilla': rise_vanilla_ratios_df, 
    'rise_integrated': rise_integrated_ratios_df
}.items():
    shapiro_df = ratio_df.apply(shapiro, axis=0)
    if (shapiro_df.loc[1,:] < 0.05).all() == True:
        print(f'{name:17s}', ': all metrics do NOT follow normal distribution')
    else:
        print(name, shapiro_df.columns[shapiro_df.loc[1,:] > 0.05].values, 'follows normal distribution')

# %% COMPUTE MEDIAN AND MEDIAN ABSOLUTE DEVIATION OF RATIOS

def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))

def get_median_and_mad(ratios_df):
    median = ratios_df.median(axis=0)
    mad = ratios_df.apply(median_absolute_deviation, axis=0, raw=True, result_type='reduce')
    return median, mad

smooth_vanilla_ratios_median_df, smooth_vanilla_ratios_mad_df = get_median_and_mad(smooth_vanilla_ratios_df)

print('\nSmoothGrad with Vanilla Gradients')
print('Prediction scores median ratio: ' \
    + f'{smooth_vanilla_ratios_median_df["prd"]:.3} +- ' \
    + f'{smooth_vanilla_ratios_mad_df["prd"]:.2}')
print('Importance values median ratio: ' \
    + f'{smooth_vanilla_ratios_median_df["imp"]:.3} +- ' \
    + f'{smooth_vanilla_ratios_mad_df["imp"]:.2}')
print('Deletion AUC median ratio: ' \
    + f'{smooth_vanilla_ratios_median_df["dlt"]:.3} +- ' \
    + f'{smooth_vanilla_ratios_mad_df["dlt"]:.2}')

smooth_integrated_ratios_median_df, smooth_integrated_ratios_mad_df = get_median_and_mad(smooth_integrated_ratios_df)

print('\nSmoothGrad with Integrated Gradients')
print('Prediction scores median ratio: ' \
    + f'{smooth_integrated_ratios_median_df["prd"]:.3} +- ' \
    + f'{smooth_integrated_ratios_mad_df["prd"]:.2}')
print('Importance values median ratio: ' \
    + f'{smooth_integrated_ratios_median_df["imp"]:.3} +- ' \
    + f'{smooth_integrated_ratios_mad_df["imp"]:.2}')
print('Deletion AUC median ratio: ' \
    + f'{smooth_integrated_ratios_median_df["dlt"]:.3} +- ' \
    + f'{smooth_integrated_ratios_mad_df["dlt"]:.2}')

rise_vanilla_ratios_median_df, rise_vanilla_ratios_mad_df = get_median_and_mad(rise_vanilla_ratios_df)

print('\nRISE-Grad with Vanilla Gradients')
print('Prediction scores median ratio: ' \
    + f'{rise_vanilla_ratios_median_df["prd"]:.3} +- ' \
    + f'{rise_vanilla_ratios_mad_df["prd"]:.2}')
print('Importance values median ratio: ' \
    + f'{rise_vanilla_ratios_median_df["imp"]:.3} +- ' \
    + f'{rise_vanilla_ratios_mad_df["imp"]:.2}')
print('Deletion AUC median ratio: ' \
    + f'{rise_vanilla_ratios_median_df["dlt"]:.3} +- ' \
    + f'{rise_vanilla_ratios_mad_df["dlt"]:.2}')

rise_integrated_ratios_median_df, rise_integrated_ratios_mad_df = get_median_and_mad(rise_integrated_ratios_df)

print('\nRISE-Grad with Integrated Gradients')
print('Prediction scores median ratio: ' \
    + f'{rise_integrated_ratios_median_df["prd"]:.3} +- ' \
    + f'{rise_integrated_ratios_mad_df["prd"]:.2}')
print('Importance values median ratio: ' \
    + f'{rise_integrated_ratios_median_df["imp"]:.3} +- ' \
    + f'{rise_integrated_ratios_mad_df["imp"]:.2}')
print('Deletion AUC median ratio: ' \
    + f'{rise_integrated_ratios_median_df["dlt"]:.3} +- ' \
    + f'{rise_integrated_ratios_mad_df["dlt"]:.2}\n')

# %% CREATE PLOTS

plt.figure()
plt.title('MEDIAN ' + results_folder + ' with Vanilla Gradients')
plt.axhline(y=1.0, linewidth=0.5, color='black')
plt.errorbar(
    x=['predictions (↑)', 'importances (↓)', 'delete_area (↓)'], 
    y=smooth_vanilla_ratios_median_df.values, 
    yerr=smooth_vanilla_ratios_mad_df.values, 
    fmt='o', ecolor='red', capsize=7, color='red', label='SmoothGrad')
plt.errorbar(
    x=['predictions (↑)', 'importances (↓)', 'delete_area (↓)'], 
    y=rise_vanilla_ratios_median_df.values, 
    yerr=rise_vanilla_ratios_mad_df.values, 
    fmt='o', ecolor='blue', capsize=7, color='blue', label='RISE-Grad')
plt.legend()
plt.savefig('with_vanilla.png')

plt.figure()
plt.title('MEDIAN ' + results_folder + ' with Integrated Gradients')
plt.axhline(y=1.0, linewidth=0.5, color='black')
plt.errorbar(
    x=['predictions (↑)', 'importances (↓)', 'delete_area (↓)'], 
    y=smooth_integrated_ratios_median_df.values, 
    yerr=smooth_integrated_ratios_mad_df.values, 
    fmt='o', ecolor='red', capsize=7, color='red', label='SmoothGrad')
plt.errorbar(
    x=['predictions (↑)', 'importances (↓)', 'delete_area (↓)'], 
    y=rise_integrated_ratios_median_df.values, 
    yerr=rise_integrated_ratios_mad_df.values, 
    fmt='o', ecolor='blue', capsize=7, color='blue', label='RISE-Grad')
plt.legend()
plt.savefig('with_integrated.png')

# %% MANN-WHITNEY TEST

vanilla_mann_pvalues = mannwhitneyu(smooth_vanilla_ratios_df, rise_vanilla_ratios_df).pvalue

if (vanilla_mann_pvalues < 0.05).all() == True:
    print('all metrics with vanilla differ significantly')
else:
    print('vanilla', smooth_vanilla_ratios_df.columns[vanilla_mann_pvalues > 0.05].values, 'NOT significantly different')

integrated_mann_pvalues = mannwhitneyu(smooth_integrated_ratios_df, rise_integrated_ratios_df).pvalue

if (integrated_mann_pvalues < 0.05).all() == True:
    print('all metrics with integrated differ significantly')
else:
    print('integrated', smooth_integrated_ratios_df.columns[integrated_mann_pvalues > 0.05].values, 'NOT significantly different')

# %%
