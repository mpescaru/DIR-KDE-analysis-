import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'MacOSX' if you're using a Mac and Tk is not installed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import kruskal
from scipy import stats as st
from sklearn.metrics import auc
import os
def normalize_kde(curves, x):
    """Normalize a list of log-KDE curves (log density), using exp and area under curve."""
    return [np.exp(c) / np.trapezoid(np.exp(c), x.squeeze()) for c in curves]
def normalize_kde_one(curve, x):
    """ Normalize a single log-KDE curve (log density), using exp and area under curve. """
    return np.exp(curve) / np.trapezoid(np.exp(curve), x.squeeze())


def average_iso(iso_table, syll_table):
    iso_df = pd.read_excel(iso_table)
    syll_df = pd.read_excel(syll_table)
    real_iso_nightjars = []
    shuffled_iso_nightjars = []
    real_iso_oscines = []
    shuffled_iso_oscines = []
    real_iso_suboscines = []
    shuffled_iso_suboscines = []
    real_iso_hummingbirds = []
    shuffled_iso_hummingbirds = []

    for species_name, species_rows in iso_df.groupby('species_birdtree'):
        species_syll = syll_df.loc[syll_df['species_birdtree'] == species_name]
        clade = species_syll.iloc[0]['our_grouping']
        if clade=='Nightjars':
            for i in range(len(species_rows)):
                real_iso_nightjars.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_nightjars.append(species_rows.iloc[i]['shuffled_%'])
        elif clade == 'Subosciness':
            for i in range(len(species_rows)):
                real_iso_suboscines.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_suboscines.append(species_rows.iloc[i]['shuffled_%'])
        elif clade == 'Hummingbirds':
            for i in range(len(species_rows)):
                real_iso_hummingbirds.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_hummingbirds.append(species_rows.iloc[i]['shuffled_%'])
        elif clade == 'Oscines':
            for i in range(len(species_rows)):
                real_iso_oscines.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_oscines.append(species_rows.iloc[i]['shuffled_%'])

    print('PERCENTAGE ISO averages: N/H/O/S', np.mean(real_iso_nightjars), np.mean(shuffled_iso_nightjars), np.mean(real_iso_hummingbirds), np.mean(shuffled_iso_hummingbirds),
          np.mean(real_iso_oscines), np.mean(shuffled_iso_oscines), np.mean(real_iso_suboscines), np.mean(shuffled_iso_suboscines))
    data_obs = pd.DataFrame({
        'observed_%': real_iso_nightjars + real_iso_suboscines + real_iso_hummingbirds + real_iso_oscines,
        'clade': (['Nightjars'] * len(real_iso_nightjars) +
                  ['Suboscines'] * len(real_iso_suboscines) +
                  ['Hummingbirds'] * len(real_iso_hummingbirds) +
                  ['Oscines'] * len(real_iso_oscines))
    })
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='clade', y='observed_%', data=data_obs, inner='box')
    plt.title("Distribution of Observed Isochrony % per Clade")
    plt.ylabel('Observed % Isochrony')
    plt.xlabel('Clade')
    plt.show()

    # data_shuffled = pd.DataFrame({
    #     'observed_%': shuffled_iso_nightjars + shuffled_iso_suboscines + shuffled_iso_hummingbirds + shuffled_iso_oscines,
    #     'clade': (['Nightjars'] * len(shuffled_iso_nightjars) +
    #               ['Suboscines'] * len(shuffled_iso_suboscines) +
    #               ['Hummingbirds'] * len(shuffled_iso_hummingbirds) +
    #               ['Oscines'] * len(shuffled_iso_oscines))
    # })
    #
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(x='clade', y='observed_%', data=data_shuffled, inner='box')
    # plt.title("Distribution of Shuffled Isochrony % per Clade")
    # plt.ylabel('Shuffled % Isochrony')
    # plt.xlabel('Clade')
    # plt.show()
    d = [real_iso_nightjars , real_iso_suboscines , real_iso_hummingbirds , real_iso_oscines, shuffled_iso_nightjars , shuffled_iso_suboscines , shuffled_iso_hummingbirds, shuffled_iso_oscines]
    plt.boxplot(d, labels=['Nightjars', 'Suboscines', 'Hummingbirds', 'Oscines', 'Shuffled Nightjars', 'Shuffled Suboscines', 'Shuffled Hummingbirds', 'Shuffled Oscines'])
    plt.title("Boxplot of Observed & Random Isocriny")
    plt.show()


    h_stat, p_val = kruskal(real_iso_nightjars,
                            real_iso_suboscines,
                            real_iso_hummingbirds,
                            real_iso_oscines)

    print(f"Kruskal-Wallis H={h_stat:.3f}, p={p_val:.4f}")


def scatter_with_fit(x, y, title, xlabel, ylabel, fn):
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.6)
    slope, intercept, r_value, p_value, _ = st.linregress(x, y)
    plt.plot(x, slope * np.array(x) + intercept, color='red', label=f'R={r_value:.2f}, p={p_value:.2g}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/maria/Desktop/sakata_lab/dir_analysis/correlation_{fn}.png')


def compare_iso_auc(iso_table, auc_table, syll_table):
    iso_df = pd.read_excel(iso_table)
    syll_df = pd.read_excel(syll_table)
    auc_df = pd.read_csv(auc_table)

    observed_auc = auc_df['AUC_real'].tolist()
    shuffled_auc = auc_df['AUC_shuffled'].tolist()
    observed_per = iso_df['observed_%'].tolist()
    shuffled_per = iso_df['shuffled_%'].tolist()

    real_auc_n = auc_df.loc[auc_df['our_grouping'] == 'Nightjars', 'AUC_real'].tolist()
    real_auc_h = auc_df.loc[auc_df['our_grouping'] == 'Hummingbirds', 'AUC_real'].tolist()
    real_auc_o = auc_df.loc[auc_df['our_grouping'] == 'Oscines', 'AUC_real'].tolist()
    real_auc_s = auc_df.loc[auc_df['our_grouping'] == 'Subosciness', 'AUC_real'].tolist()

    real_iso_nightjars = []
    shuffled_iso_nightjars = []
    real_iso_oscines = []
    shuffled_iso_oscines = []
    real_iso_suboscines = []
    shuffled_iso_suboscines = []
    real_iso_hummingbirds = []
    shuffled_iso_hummingbirds = []

    for species_name, species_rows in iso_df.groupby('species_birdtree'):
        species_syll = syll_df.loc[syll_df['species_birdtree'] == species_name]
        if species_syll.empty:
            continue
        clade = species_syll.iloc[0]['our_grouping']
        for i in range(len(species_rows)):
            if clade == 'Nightjars':
                real_iso_nightjars.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_nightjars.append(species_rows.iloc[i]['shuffled_%'])
            elif clade == 'Subosciness':
                real_iso_suboscines.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_suboscines.append(species_rows.iloc[i]['shuffled_%'])
            elif clade == 'Hummingbirds':
                real_iso_hummingbirds.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_hummingbirds.append(species_rows.iloc[i]['shuffled_%'])
            elif clade == 'Oscines':
                real_iso_oscines.append(species_rows.iloc[i]['observed_%'])
                shuffled_iso_oscines.append(species_rows.iloc[i]['shuffled_%'])

    # Compute per-species difference metrics
    auc_df['AUC_diff'] = auc_df['AUC_real'] - auc_df['AUC_shuffled']
    iso_df['percent_diff'] = iso_df['observed_%'] - iso_df['shuffled_%']

    # Merge per species for all 3 comparisons
    merged_obs = pd.merge(
        auc_df[['species_birdtree', 'AUC_real']],
        iso_df[['species_birdtree', 'observed_%']],
        on='species_birdtree',
        how='inner'
    ).dropna()

    merged_shuf = pd.merge(
        auc_df[['species_birdtree', 'AUC_shuffled']],
        iso_df[['species_birdtree', 'shuffled_%']],
        on='species_birdtree',
        how='inner'
    ).dropna()

    merged_diff = pd.merge(
        auc_df[['species_birdtree', 'AUC_diff']],
        iso_df[['species_birdtree', 'percent_diff']],
        on='species_birdtree',
        how='inner'
    ).dropna()

    # Compute species-level Pearson correlations
    r_obs, p_obs = st.pearsonr(merged_obs['AUC_real'], merged_obs['observed_%'])
    r_shuf, p_shuf = st.pearsonr(merged_shuf['AUC_shuffled'], merged_shuf['shuffled_%'])
    r_diff, p_diff = st.pearsonr(merged_diff['AUC_diff'], merged_diff['percent_diff'])

    # Print results
    print(f'\n[Species-level] Correlation AUC_real vs observed_%: r = {r_obs:.3f}, p = {p_obs:.3g}')
    print(f'[Species-level] Correlation AUC_shuffled vs shuffled_%: r = {r_shuf:.3f}, p = {p_shuf:.3g}')
    print(f'[Species-level] Correlation AUC_diff vs percent_diff: r = {r_diff:.3f}, p = {p_diff:.3g}')
    print('Average ISO percentage (real): N/H/O/S', np.mean(real_iso_nightjars), np.mean(real_iso_hummingbirds), np.mean(real_iso_oscines), np.mean(real_iso_suboscines))
    print('Average ISO AUC (real): N/H/O/S', np.mean(real_auc_n), np.mean(real_auc_h), np.mean(real_auc_o), np.mean(real_auc_s))


    # === Scatter plots using filtered merged species-level data ===
    scatter_with_fit(
        merged_obs['AUC_real'], merged_obs['observed_%'],
        'Observed AUC vs Observed % (species-level)',
        'AUC_real', 'Observed %', 'obs_percetageVauc'
    )
    scatter_with_fit(
        merged_shuf['AUC_shuffled'], merged_shuf['shuffled_%'],
        'Shuffled AUC vs Shuffled % (species-level)',
        'AUC_shuffled', 'Shuffled %', 'shuffled_percetageVauc'
    )
    scatter_with_fit(
        merged_diff['AUC_diff'], merged_diff['percent_diff'],
        'AUC Difference vs ISO % Difference (species-level)',
        'AUC_real - AUC_shuffled', 'Observed % - Shuffled %', 'difference_percetageVauc'
    )

def add_IOI_DIR_from_onsets(df):
    """
    Efficiently compute IOI and DIR from 'onset_msec' within each song,
    identified by 'species_birdtree' and 'seg_id'.
    """
    df = df.sort_values(by=["species_birdtree", "seg_id", "i_syll_in_song"]).copy()
    df["IOI"] = np.nan
    df["DIR"] = np.nan

    for (species, seg_id), group in df.groupby(["species_birdtree", "seg_id"]):
        if group.shape[0] < 3:
            continue
        idx = group.index
        onset = group["onset_msec"].values
        seg = group["i_syll_in_song"].values

        # Calculate IOIs only where i_syll_in_song is consecutive
        consecutive = np.where(seg[1:] == seg[:-1] + 1)[0]
        iois = np.full(len(onset), np.nan)
        iois[consecutive] = onset[consecutive + 1] - onset[consecutive]
        df.loc[idx, "IOI"] = iois

        # Calculate DIRs where two consecutive IOIs are valid
        valid_ioi_idx = np.where(~np.isnan(iois))[0]
        valid_pairs = valid_ioi_idx[valid_ioi_idx + 1 < len(iois)]
        dirs = np.full(len(onset), np.nan)
        for i in valid_pairs:
            if not np.isnan(iois[i+1]):
                dirs[i] = iois[i] / (iois[i] + iois[i+1])
        df.loc[idx, "DIR"] = dirs

    return df

def make_valid_column_names(names):
    seen = {}
    valid = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            valid.append(name)
        else:
            count = seen[name]
            new_name = f"{name}_{count}"
            while new_name in seen:
                count += 1
                new_name = f"{name}_{count}"
            seen[name] = count + 1
            seen[new_name] = 1
            valid.append(new_name)
    return valid