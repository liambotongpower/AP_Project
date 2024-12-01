import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

# Load league table and fixtures data
league_table_df = pd.read_csv('q2_pdfs/LeagueTable.csv')
fixtures_df = pd.read_csv('q2_pdfs/FixtureTable.csv')

# Filter relevant columns from the league table
league_table_df_filtered = league_table_df[['Team', 'MP', 'GF', 'GA', 'Points', 'Poss', 'CrdY', 'CrdR']].dropna()

# Calculate average goals scored (GF) and conceded (GA) per match
league_table_df_filtered['Avg_GF_per_match'] = league_table_df_filtered['GF'] / league_table_df_filtered['MP']
league_table_df_filtered['Avg_GA_per_match'] = league_table_df_filtered['GA'] / league_table_df_filtered['MP']

# Prepare fixtures for simulation
fixtures_df_filtered = fixtures_df[['Match Day', 'Home Team', 'Away Team', 'Likely Home Goal', 'Likely Away Goal']].dropna()

# Monte Carlo simulation parameters
num_simulations = 1000

# Initialize a dictionary to hold final point forecasts for each team across simulations
team_final_points = {team: [] for team in league_table_df_filtered['Team']}

# Helper function to adjust goal averages
def adjust_goals(base_goals, possession, yellow_cards, red_cards):
    possession_factor = 1 + (possession - 50) / 100  # Scale possession effect
    discipline_factor = max(0.9, 1 - (0.05 * yellow_cards + 0.1 * red_cards))  # Penalize for cards
    return base_goals * possession_factor * discipline_factor

# Run Monte Carlo simulations with a progress bar
for _ in tqdm(range(num_simulations), desc="Simulating Matches", unit="simulation"):
    # Initialize a dictionary to hold points for this simulation run
    points_simulation = league_table_df_filtered.set_index('Team')['Points'].to_dict()

    # Simulate remaining matches
    for _, row in fixtures_df_filtered.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goal_avg = row['Likely Home Goal']
        away_goal_avg = row['Likely Away Goal']
        
        # Retrieve possession and card metrics for both teams
        home_stats = league_table_df_filtered[league_table_df_filtered['Team'] == home_team].iloc[0]
        away_stats = league_table_df_filtered[league_table_df_filtered['Team'] == away_team].iloc[0]

        home_possession = home_stats['Poss']
        home_yellow_cards = home_stats['CrdY']
        home_red_cards = home_stats['CrdR']

        away_possession = away_stats['Poss']
        away_yellow_cards = away_stats['CrdY']
        away_red_cards = away_stats['CrdR']

        # Adjust goals based on possession and discipline factors
        home_goal_avg = adjust_goals(home_goal_avg, home_possession, home_yellow_cards, home_red_cards)
        away_goal_avg = adjust_goals(away_goal_avg, away_possession, away_yellow_cards, away_red_cards)

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
