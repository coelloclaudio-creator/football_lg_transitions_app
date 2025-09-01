import pandas as pd 
import numpy as np
import streamlit as st
from analytics_functions import load_data, add_player_age, filter_players, find_age_group, filter_stats, \
    find_prev_season, find_prev_stat, create_pivot_df, create_template_matrix, find_stat_diff, populate_matrix,\
    populate_custom_matrix, transition_count_matrix, build_rs_matrix, fill_missing_values_from_influence, style_matrix

if 'df' not in st.session_state:
    st.error("Please load the player stats data first.")
    st.stop()

df = st.session_state.df.copy()

st.title('Build Transition Matrices')

df = add_player_age(df)
df, q1_threshold = filter_players(df)
df = find_age_group(df)
stat_columns_df = filter_stats(df)
pivot_df = create_pivot_df(df, stat_columns_df.columns)

exclude_columns = ['age_group', 'Age', 'competition_name', 'season_name','player_name_y', 'team_name', 'season_start_year', 'primary_position']
stat_columns = [col for col in stat_columns_df.columns if col not in exclude_columns]

st.session_state.pivot_df = pivot_df
st.session_state.stat_columns = stat_columns

st.subheader('Origin Year and Destination Year Table Preview')
st.dataframe(pivot_df.head(n=10))

st.write('Choose one of the statistics to build transition matrices for player movements between leagues.')
st.write('The matrices show the average change in the selected stat when players move from one league (row), to another league (column).')

selected_stat = st.selectbox('Select Stat for Transition Analysis', options = stat_columns, index=0)
template_matrix = create_template_matrix(pivot_df)
pivot_df = find_stat_diff(pivot_df, [selected_stat])
original_matrix = populate_matrix(pivot_df, [selected_stat], template_matrix)

st.subheader(f'Original Matrix for {selected_stat} (NaN values mean transition not observed)')
st.dataframe(style_matrix(original_matrix[selected_stat]))

league_exclusion = st.multiselect('Exclude Leagues from Matrix (optional)', options=['All'] + sorted(pivot_df['competition_name'].dropna().unique().tolist()),
    default='All')
excluded_leagues = [] if 'All' in league_exclusion else league_exclusion

age_group_selection = st.multiselect('Select Age Group(s) (optional)', options= ['All'] +sorted(pivot_df['age_group'].dropna().unique().tolist()),
                                    default='All')
age_group = None if 'All' in age_group_selection else age_group_selection

position_selection = st.multiselect('Select Primary Position (optional)', options= ['All'] + sorted(pivot_df['primary_position'].dropna().unique().tolist()),
                                    default='All')
primary_position = None if 'All' in position_selection else position_selection

custom_matrix = populate_custom_matrix(pivot_df, [selected_stat], template_matrix, age_group= age_group, primary_position= primary_position)
st.dataframe(style_matrix(custom_matrix[selected_stat]))

rs_df = build_rs_matrix(pivot_df, template_matrix, alpha=0.15, min_count=10, power=1)
st.session_state.rs_df = rs_df

if st.button('Fill Missing Values from Influence Matrix'):
    filled_matrices = fill_missing_values_from_influence(matrices ={selected_stat: custom_matrix[selected_stat]},
                                                         rs_df = st.session_state.rs_df)
    
    filled_matrix = filled_matrices[selected_stat]

    st.subheader(f"Filled matrix for {selected_stat}")
    st.dataframe(style_matrix(filled_matrix.round(4)))

st.write('The filled in values are derived from a matrix that calculates the influence of each league on others based on player transitions.' \
' These values are predicted changes in the selected stat based on intermediate leagues.'\
     ' Please refer to the glosssary for more information on the calculations.')







