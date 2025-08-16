import pandas as pd
import numpy as np
def load_data(player_stats_file, player_catalog_file, team_catalog_file, player_season_catalog_file):
    df = pd.read_csv(player_stats_file)

    if player_catalog_file is not None:
        players = pd.read_csv(player_catalog_file)
        df = df.merge(players, on= 'player_id', how= 'left')

    if team_catalog_file is not None:
        teams = pd.read_csv(team_catalog_file)
        df = df.merge(teams, on="team_id", how="left")

    if player_season_catalog_file is not None:
        player_season = pd.read_csv(player_season_catalog_file)
        player_season_subset = (
            player_season[['competition_id', 'season_id', 'competition_name', 'season_name']]
            .drop_duplicates(subset=['competition_id', 'season_id'])
        )
        df = df.merge(player_season_subset, on=['competition_id', 'season_id'], how="left")

    return df

def add_player_age(df):
    df['season_start_year'] = df['season_name'].str[:4].astype(int)

    # Convert birth_date to datetime if not already
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

    # Calculate age in years
    df['Age'] = df['season_start_year'] - df['birth_date'].dt.year

    return df

def filter_players(df, minutes_column='player_season_minutes',
                   female_column='player_female',
                   position_column='primary_position'):
    q1_threshold = df[minutes_column].quantile(0.25)

    df_clean = df[
    (df[minutes_column] > q1_threshold) &
    (df[female_column] == False) &
    (df[position_column] != 'Goalkeeper')
]


    return df_clean, q1_threshold

def find_age_group (df, age_column= 'Age'):
    bins = [16, 20, 24, 27, 29, 32, float('inf')]
    labels = ['17-20', '21-24', '25-27', '28-29', '30-32', '33+']
    df['age_group'] = pd.cut(df['Age'], bins= bins, labels=labels, right= True)
    return df

def filter_stats (df, stat_columns = None):
    if stat_columns is None:
        # these are the 10 relevant stats picked for the study + row identifiers (player name, season, etc.)
        stat_columns = ['player_season_deep_progressions_90', 'player_season_key_passes_90', 'player_season_np_xg_90', 
                    'player_season_npxgxa_90', 'player_season_np_psxg_90', 'player_season_padj_tackles_and_interceptions_90',
                    'player_season_xa_90', 'player_season_lbp_completed_90', 'player_season_obv_90',
                    'player_season_turnovers_90','age_group', 'Age', 'competition_name','season_name','player_name_y','team_name', 
                    'season_start_year', 'primary_position']
    cols_to_keep = [col for col in stat_columns if col in df.columns]
    return df[cols_to_keep]

def find_prev_season (row, df, id_col =  'player_name_y', season_col = 'season_start_year'):
    prev_season_year = row[season_col] - 1
    

    prev_season = df[(df[id_col] == row[id_col]) & (df[season_col] == prev_season_year)]
    return prev_season.iloc[0] if not prev_season.empty else None

def find_prev_stat (row, df, stat_columns):
    prev_row = find_prev_season (row, df, id_col =  'player_name_y', season_col = 'season_start_year')
    if prev_row is not None:
        return prev_row [stat_columns]
    return None

def create_pivot_df ( df, stat_columns):


    player_season_dict = {} # create dictionary
    for _, current_row in df.iterrows(): # tuple. name, season stats
        player_name = current_row['player_name_y']
        season = current_row['season_start_year']

        if player_name not in player_season_dict: # if player_name not yet in dict, create dict of dict
            player_season_dict[player_name] = {} 
        player_season_dict[player_name][season]={stat : current_row[stat] for stat in stat_columns} # fill it w season stats

    records = []
    for _, current_row in df.iterrows():
        player_name = current_row['player_name_y']
        current_season = current_row['season_start_year']
        prev_season = current_season - 1

        if player_name in player_season_dict and prev_season in player_season_dict[player_name]: 
            prev_stats = player_season_dict[player_name][prev_season]
            row_data = {'player_name_y' : player_name, 'season_start_year': current_season,
                    'prev_season_year': prev_season}

            for stat in stat_columns:
                row_data[stat] = current_row[stat] # row data current stats
            for stat in stat_columns:
                row_data[stat + '_prev']= prev_stats[stat] # row data prev stats. adds string to distinguish

            records.append(row_data) # append records series w row data

    return pd.DataFrame(records) # return records, which should be the pivot style df.

def create_template_matrix (df):
    origin_leagues = set(df['competition_name_prev'].unique())
    destination_leagues = set(df['competition_name'].unique())

    all_leagues = sorted(origin_leagues.union(destination_leagues))

    template_matrix = pd.DataFrame(np.nan, index= all_leagues, columns = all_leagues, dtype = float)

    return template_matrix

def find_stat_diff (df, stat_columns):
    for stat in stat_columns:
        stat_prev = stat + '_prev'
        stat_diff = stat + '_diff'
        df[stat_diff] = df[stat] - df[stat_prev]

    return df

def populate_matrix(df, stat_columns, template_matrix):
    matrices = {}

    for stat in stat_columns:
        diff_col = stat + '_diff'

        # Group and filter
        grouped = df.groupby(['competition_name_prev', 'competition_name'])[diff_col].agg(['mean', 'size'])
        filtered = grouped[grouped['size'] >= 5]

        # Dictionary for fast lookups
        transition_dict = {
            (origin, destination): row['mean']
            for (origin, destination), row in filtered.iterrows()
        }

        # Fill matrix
        matrix = template_matrix.copy().astype(float)
        for origin in matrix.index:
            for destination in matrix.columns:
                if (origin, destination) in transition_dict:
                    matrix.at[origin, destination] = round(transition_dict[(origin, destination)], 4)

        matrices[stat] = matrix

    return matrices

def populate_custom_matrix (df, stat_columns, template_matrix, age_group= None, primary_position= None):
    df_filtered = df.copy()

      # Age group filter
    if age_group and 'All' not in age_group:
        df_filtered = df_filtered[df_filtered['age_group'].isin(age_group)]

    # Primary position filter
    if primary_position and 'All' not in primary_position:
        df_filtered = df_filtered[df_filtered['primary_position'].isin(primary_position)]

    return populate_matrix(df_filtered, stat_columns, template_matrix)

def transition_count_matrix (df, template_matrix):
    transition_count = df.groupby(['competition_name_prev','competition_name']).size()

    count_dict = {
                        (origin, destination): count
                        for (origin, destination), count in transition_count.items()}

    count_matrix = template_matrix.copy().astype(float)
    for origin in count_matrix.index:
        for destination in count_matrix.columns:
            key = (origin, destination)
            if key in count_dict:
                count_matrix.at[origin, destination] = count_dict[key]
    return count_matrix

def build_rs_matrix(df, template_matrix, alpha=0.15, min_count=10, power=1): # combine 4 functions into 1 influence matrix
    # Build powered transition matrix once
    transition_count = df.groupby(['competition_name_prev', 'competition_name']).size()
    filtered_count = transition_count[transition_count >= min_count]

    count_matrix = template_matrix.copy().astype(float)
    for (origin, destination), count in filtered_count.items():
        count_matrix.at[origin, destination] = count

    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0)
    matrix_powered = np.linalg.matrix_power(prob_matrix.to_numpy(), power)
    matrix_scaled_restart = (1 - alpha) * matrix_powered #rwr

    # Identity minus P_t
    identity = np.eye(len(template_matrix))
    M = identity - matrix_scaled_restart

    # Precompute inverse once
    M_inv = np.linalg.inv(M)

    # e_s vectors are  identity rows np.eye()
    leagues = list(template_matrix.index)
    rs_df = pd.DataFrame(alpha * M_inv, index=leagues, columns=leagues)

    return rs_df

def fill_missing_values_from_influence(matrices, rs_df):
    
    filled_matrices = {}
    leagues = matrices[next(iter(matrices))].index

    # Ensure rs_df order matches league order
    rs_df = rs_df.loc[leagues, leagues]

    for stat, matrix in matrices.items():
        filled_matrix = matrix.copy()

        for origin in leagues:
            r_s = rs_df.loc[origin]

            for destination in leagues:
                if pd.isna(filled_matrix.at[origin, destination]):

                    intermediate_info = {
                        k: (r_s[k], matrix.at[k, destination])
                        for k in leagues
                        if k != origin and k != destination and pd.notna(matrix.at[k, destination])
                    }

                    if intermediate_info:
                        numer = sum(infl * val for infl, val in intermediate_info.values())
                        denom = sum(infl for infl, _ in intermediate_info.values())

                        if denom > 0:
                            filled_matrix.at[origin, destination] = numer / denom

        filled_matrices[stat] = filled_matrix

    return filled_matrices

def style_matrix(df):
    def color_values(val):
        if pd.isna(val) or val == 0:
            return ''
        elif val > 0:
            return 'background-color: green; color: white;'
        else:  # val < 0
            return 'background-color: red; color: white;'
    return df.style.applymap(color_values).format("{:.4f}")

