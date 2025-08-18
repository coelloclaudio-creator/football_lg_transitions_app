# Football league transitions. App for transfer analytics.

Claudio Coello 

Project for Penta 

Directed by Santiago Fernandez del Castillo.

# Structure of the tasks performed 
- 1. Data loading and processing
- 2. Matrix creation
- 3. Imputation of missing values
- 4. Streamlit app

# Data loading and processing

Statsbomb data was used for the completion of this project. The data compiles player season statistics from 2019 to 2024, and it ranges multiple professional leagues, from Europe to North and South America.

The data was processed using the pandas library, where a pivot table with back to back seasons for each player was created. (Eg. Son Heung Min 2020, and 2019 stats).

# Matrix creation

Using this back to back season table, we compiled all of the players who had identical transitions from one league to another (eg. all players who went from Bundesliga to the EPL), and calculated the average change of their statistical outputs in the origin and destination leagues.

# Imputation of missing values

Many transitions were either not observed, or happened in small numbers, so the point of this project was to find a way to predict these missing values using all the transitions that we had in our data. 

We
