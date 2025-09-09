import pandas as pd
import numpy as np
import os
import helpers as h


def percent_iso_music(file, n_iter=500):
    import pandas as pd
    import numpy as np
    import os
    import helpers as h

    print('Calculating percentage of music DIRs within the isochrony zone.')
    df = pd.read_csv(file)
    obs_rand_mean = []
    rand_matrix = []
    file_labels = []

    for song_id, song_df in df.groupby("title"):
        print(f"Processing {song_id}")

        # Extract relevant arrays once to avoid repeated DataFrame access
        is_note = song_df['is_note'].values
        duration = song_df['duration'].values
        phrase_ids = song_df['phrase_id'].values

        # Compute real IOIs within phrases
        iois = [duration[i] for i in range(len(duration) - 1)
                if is_note[i] and is_note[i + 1] and phrase_ids[i] == phrase_ids[i + 1]]

        if len(iois) < 11:
            print(f"Skipping {song_id} due to insufficient IOIs")
            continue

        real_dirs = np.array([iois[i] / (iois[i] + iois[i + 1]) for i in range(len(iois) - 1)])
        prop_real = 100 * np.sum((real_dirs > 0.444) & (real_dirs < 0.555)) / len(real_dirs)

        prop_rand = np.full(n_iter, np.nan)
        grouped = list(song_df.groupby("phrase_id"))

        for iter in range(n_iter):
            all_shuffled_dirs = []

            for _, phrase_df in grouped:
                is_note_phrase = phrase_df['is_note'].values
                duration_phrase = phrase_df['duration'].values
                iois_phrase = [duration_phrase[i] for i in range(len(duration_phrase) - 1)
                               if is_note_phrase[i] and is_note_phrase[i + 1]]
                if len(iois_phrase) < 2:
                    continue

                shuffled_iois = np.random.permutation(iois_phrase)
                shuffled_dirs = [shuffled_iois[i] / (shuffled_iois[i] + shuffled_iois[i + 1])
                                 for i in range(len(shuffled_iois) - 1)]
                all_shuffled_dirs.extend(shuffled_dirs)

            if len(all_shuffled_dirs) >= 10:
                all_shuffled_dirs = np.array(all_shuffled_dirs)
                prop_rand[iter] = 100 * np.sum((all_shuffled_dirs > 0.444) & (all_shuffled_dirs < 0.555)) / len(all_shuffled_dirs)

        mean_rand = np.nanmean(prop_rand)
        obs_rand_mean.append([song_id, prop_real, mean_rand])
        rand_matrix.append(prop_rand)
        file_labels.append(song_id)

    obs_mean_random_df = pd.DataFrame(obs_rand_mean, columns=["title", "observed_%", "shuffled_%"])
    rand_matrix_df = pd.DataFrame(np.array(rand_matrix).T,
                                  columns=h.make_valid_column_names(file_labels))

    folder, base = os.path.split(file)
    base_name, _ = os.path.splitext(base)
    output_path = os.path.join(folder, f"{base_name}_percent_iso.xlsx")

    with pd.ExcelWriter(output_path) as writer:
        obs_mean_random_df.to_excel(writer, sheet_name="obs+shuffled_mean", index=False)
        rand_matrix_df.to_excel(writer, sheet_name="500_shuffled_all", index=False)

    print(f"Saved output Excel to {output_path}")


def compute_percent_iso(file, n_iter=500):
    # Load input
    df = pd.read_excel(file)
    print(
        'Calculating percentage of music DIRs within the isochorony zone.',
        'Shuffling is done within song, real and random percentages are within species.')

    # Add IOI and DIR columns
    df = h.add_IOI_DIR_from_onsets(df)

    # Drop rows with NaNs in IOI or DIR
    df = df.dropna(subset=["IOI", "DIR"])

    # Convert species to string
    df["species_birdtree"] = df["species_birdtree"].astype(str)
    unique_species = df["species_birdtree"].unique()

    obs_rand_mean = []
    rand_matrix = []
    species_labels = []

    for species in unique_species:
        print(species)
        df_species = df[df["species_birdtree"] == species].copy()
        if df_species.empty:
            continue

        observed_dir = df_species["DIR"].dropna().values
        if len(observed_dir) < 10:
            continue

        # Observed % in isochrony range
        prop_obs = 100 * np.sum((observed_dir > 0.444) & (observed_dir < 0.555)) / len(observed_dir)

        # Shuffle IOIs within each song for the species
        prop_rand = np.zeros(n_iter)
        song_groups = df_species.groupby("seg_id")

        for j in range(n_iter):
            all_random_dirs = []

            for _, song_df in song_groups:
                song_iois = song_df["IOI"].dropna().values
                if len(song_iois) < 2:
                    continue
                shuffled_ioi = np.random.permutation(song_iois)
                random_dir = [shuffled_ioi[k] / (shuffled_ioi[k] + shuffled_ioi[k + 1])
                              for k in range(len(shuffled_ioi) - 1)]
                all_random_dirs.extend(random_dir)

            if len(all_random_dirs) == 0:
                prop_rand[j] = np.nan
            else:
                all_random_dirs = np.array(all_random_dirs)
                prop_rand[j] = 100 * np.sum((all_random_dirs > 0.444) & (all_random_dirs < 0.555)) / len(
                    all_random_dirs)

        if np.all(np.isnan(prop_rand)):
            continue

        mean_rand = np.nanmean(prop_rand)
        obs_rand_mean.append([species, prop_obs, mean_rand])
        rand_matrix.append(prop_rand)
        species_labels.append(species)

    # Build output tables
    obs_rand_df = pd.DataFrame(obs_rand_mean, columns=["species_birdtree", "observed_%", "shuffled_%"])
    rand_matrix_df = pd.DataFrame(np.array(rand_matrix).T,
                                  columns=h.make_valid_column_names(species_labels))

    # Save output
    folder, base = os.path.split(file)
    base_name, _ = os.path.splitext(base)
    output_path = os.path.join(folder, f"{base_name}_percent_iso.xlsx")

    with pd.ExcelWriter(output_path) as writer:
        obs_rand_df.to_excel(writer, sheet_name="obs+shuffled_mean", index=False)
        rand_matrix_df.to_excel(writer, sheet_name="500_shuffled_all", index=False)

    print(f"Saved output Excel to {output_path}")
