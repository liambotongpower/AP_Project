import pandas as pd
import numpy as np

# Load league table and fixtures data
league_table_df = pd.read_csv('q2_pdfs/LeagueTable.csv')
fixtures_df = pd.read_csv('q2_pdfs/FixtureTable.csv')

# Filter league table relevant columns
league_table_df_filtered = league_table_df[['Team', 'MP', 'GF', 'GA', 'Points']].dropna()

# Calculate average goals scored (GF) and conceded (GA) per match
league_table_df_filtered['Avg_GF_per_match'] = league_table_df_filtered['GF'] / league_table_df_filtered['MP']
league_table_df_filtered['Avg_GA_per_match'] = league_table_df_filtered['GA'] / league_table_df_filtered['MP']

# Prepare fixtures for simulation
fixtures_df_filtered = fixtures_df[['Match Day', 'Home Team', 'Away Team', 'Likely Home Goal', 'Likely Away Goal']].dropna()

# Monte Carlo simulation parameters
num_simulations = 1000
remaining_matches = fixtures_df_filtered.shape[0]

# Initialize a dictionary to hold final point forecasts for each team across simulations
team_final_points = {team: [] for team in league_table_df_filtered['Team']}

# Run Monte Carlo simulations
for _ in range(num_simulations):
    # Initialize a dictionary to hold points for this simulation run
    points_simulation = league_table_df_filtered.set_index('Team')['Points'].to_dict()

    # Simulate remaining matches
    for _, row in fixtures_df_filtered.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goal_avg = row['Likely Home Goal']
        away_goal_avg = row['Likely Away Goal']

        # Simulate goals using Poisson distribution
        home_goals = np.random.poisson(home_goal_avg)
        away_goals = np.random.poisson(away_goal_avg)

        # Assign points based on match outcome
        if home_goals > away_goals:
            points_simulation[home_team] += 3
        elif home_goals < away_goals:
            points_simulation[away_team] += 3
        else:
            points_simulation[home_team] += 1
            points_simulation[away_team] += 1

    # Store the points from this simulation run
    for team in team_final_points:
        team_final_points[team].append(points_simulation.get(team, 0))

# Compute average final points per team after simulations
forecasted_points = {team: np.mean(points) for team, points in team_final_points.items()}
forecasted_points_df = pd.DataFrame(list(forecasted_points.items()), columns=['Team', 'Forecasted Points']).sort_values(by='Forecasted Points', ascending=False)

# Display forecasted final points
print("Forecasted Final Points for Premier League Teams")
print(forecasted_points_df)
