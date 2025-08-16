import streamlit as st
import pandas as pd
import numpy as np
from analytics_functions import load_data, add_player_age, filter_players, find_age_group, filter_stats, \
    find_prev_season, find_prev_stat, create_pivot_df, create_template_matrix, find_stat_diff, populate_matrix,\
    populate_custom_matrix, transition_count_matrix, build_rs_matrix, fill_missing_values_from_influence

st.title('Football League Transitions Analysis')
st.write('This app analyzes trends in player transitions between football leagues.')
st.write('The transition matrices in the next page show the average change in selected statistics when players move from one league to another.')
st.write('Upload your data in CSV format to get started.')

player_stats_file = st.file_uploader('Upload Player Stats CSV', type= ['csv'])
player_catalog_file = st.file_uploader('Upload Player Catalog CSV (optional)', type= ['csv'])
team_catalog_file = st.file_uploader('Upload Team Catalog CSV (optional)', type= ['csv'])
player_season_catalog_file = st.file_uploader('Upload Player Season Catalog CSV (optional)', type= ['csv'])

if player_stats_file is not None:
    try:
        df = load_data(player_stats_file, player_catalog_file, team_catalog_file, player_season_catalog_file)
        st.session_state.df = df
        st.success('Data loaded successfully')
        st.write('### Preview of your dataset:')
        st.dataframe(df.head(n=20))
    except Exception as e:
        st.error(f"Error loading data: {e}")


    