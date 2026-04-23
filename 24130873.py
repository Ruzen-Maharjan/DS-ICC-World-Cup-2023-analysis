# ==============================================================
# MINI GROUP PROJECT
# PRINCIPLES OF DATA PRESENTATION
# Analysis of Player and Team Performance in ICC World Cup 2023
# ==============================================================

# ==============================================================
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================
# Load Data
# ==============================================================

# Load the deliveries dataset
deliveries_df = pd.read_csv("deliveries.csv")

#Load matches data
matches_df = pd.read_csv("matches.csv")

# ===============================================================
# Data Inspection
# ===============================================================

# Display first few rows
deliveries_df.head()

# Show dataset structure
deliveries_df.info()

# Show statistical summary
deliveries_df.describe()

#Check for missing values
print(deliveries_df.isnull().sum())    

# Extract and Inspect matches dataset
matches_df.head()
matches_df.info()
matches_df.describe()

# Check for missing values
print(matches_df.isnull().sum())

# Display the column names
deliveries_df.columns
#matches_df.columns

#Value caounts for categorical data.
print(deliveries_df['innings'].value_counts())
print(deliveries_df['wicket_type'].value_counts())

# =================================================================
# Data Cleaning
# =================================================================

# Fill misisng values in extra runs column with 0
deliveries_df['wides'].fillna(0, inplace=True)
deliveries_df['noballs'].fillna(0, inplace=True)
deliveries_df['byes'].fillna(0, inplace=True)
deliveries_df['legbyes'].fillna(0, inplace=True)
deliveries_df['penalty'].fillna(0, inplace=True)

# Fill missing values for wicket-related columns with 'None'
deliveries_df['wicket_type'].fillna('None', inplace=True)
deliveries_df['player_dismissed'].fillna('None', inplace=True)
deliveries_df['other_wicket_type'].fillna('None', inplace=True)
deliveries_df['other_player_dismissed'].fillna('None', inplace=True)

# Check for missing values again after cleaning
print(deliveries_df.isnull().sum())


# ==================================================================
# Feature Creation
# ==================================================================

# Runs scored from bat
bat_runs = deliveries_df['runs_off_bat']

# Extra runs
wides = deliveries_df['wides']
noballs = deliveries_df['noballs']
byes = deliveries_df['byes']
legbyes = deliveries_df['legbyes']
penalty = deliveries_df['penalty']

# Total runs calculation
deliveries_df['total_runs'] = bat_runs + wides + noballs + byes + legbyes + penalty

# Save and cleaned dataset
deliveries_df.to_csv('deliveryfile1.csv', index=False)

# ===================================================================
# Batsman Performance Analysis
# ===================================================================

# Group data by batsman
batsman_stats = deliveries_df.groupby('striker').agg({
    'runs_off_bat': 'sum',
    'ball': 'count'
}).reset_index()

 # Rename columns
batsman_stats.columns = ['batsman','runs', 'balls_faced']

# calculating average : total runs / no of time dismissal
# Dismissal Stats
dismissals = deliveries_df['player_dismissed'].value_counts().reset_index()
dismissals.columns = ['batsman' , 'dismissals']

# Merge and calculate data
batsman_stats = pd.merge(batsman_stats, dismissals, on='batsman', how='left')
batsman_stats['dismissals'] = batsman_stats['dismissals'].fillna(0)

# Calculate batting average
batsman_stats['batting_average'] = np.where(
    batsman_stats['dismissals'] > 0,
    batsman_stats['runs'] / batsman_stats['dismissals'],
    np.inf
)

# Calculate Strike Rate
batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls_faced']) * 100

# Select and display Top 10
top_10 = batsman_stats.sort_values(by='runs', ascending=False).head(10)
print(top_10[['batsman', 'runs', 'batting_average', 'strike_rate']])
print("Top 10 Batsmen Statistics:")


# ======================================================================
# Top 10 Batsman Visualization
# ======================================================================

# Sort data
top_10 = top_10.sort_values(by='runs', ascending=False)

# Set style
sns.set_style("whitegrid")

plt.figure()

# Create colorful palette
palette = sns.color_palette("viridis", len(top_10))

ax = sns.barplot(
    x='batsman',
    y='runs',
    data=top_10,
    palette=palette
)

# Add value labels
for i in ax.containers:
    ax.bar_label(i)

plt.xticks(rotation=45)
plt.xlabel('Batsman')
plt.ylabel('Runs')
plt.title('Top 10 Batsmen by Runs (Colorful Visualization)')

plt.tight_layout()
plt.savefig('Top10_batsman.png')
plt.show()

# ==========================================================================================
# Batting Avg vs Strike rate
# ==========================================================================================

# A. Scatter Plot: All Batsmen (Avg vs SR)
plt.figure(figsize=(12, 8))
top_scorers = batsman_stats.sort_values(by='runs', ascending=False).head(20)

sns.scatterplot(
    data=batsman_stats,
    x='batting_average',
    y='strike_rate', 
    size='runs',
    hue='runs',
    palette='viridis',
    alpha=0.6, 
    sizes=(20, 500)
    )

# Label top players
for i, row in top_scorers.head(10).iterrows():
    plt.text(row['batting_average'] + 1, row['strike_rate'], row['batsman'], fontsize=9)

plt.title('All Batsmen: Batting Average vs Strike Rate', fontsize=15)
plt.xlabel('Batting Average', fontsize=12)
plt.ylabel('Strike Rate', fontsize=12)
plt.savefig('Batting_vs_Strike.png')
plt.show()
plt.close()


# ============================================================================================
# Bowling Performance (Wickets)

# Define specific dismissal types credited to a bowler
bowler_wicket_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']

# 2. Filter the data for these specific wickets
# We use .isin() for accuracy and avoid 'retired hurt' or 'run out'

wickets_df = deliveries_df[deliveries_df['wicket_type'].str.lower().isin(bowler_wicket_types)]

# Get top 10 bowlers
top_10 = (
    wickets_df.groupby('bowler')
    .size()
    .reset_index(name='wickets')
    .sort_values(by='wickets', ascending=False)
    .head(10)
)
print("Top 10 Bowlers by Wickets:")
print(top_10)

# Create the visualization
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

bar_plot = sns.barplot(
    data=top_10,
    x='wickets',
    y='bowler',
    palette='magma',
    hue='bowler',
    legend=False
)

# Add wicket counts as labels on the bars
for container in bar_plot.containers:
    bar_plot.bar_label(container, padding=5, fontsize=11, fontweight='bold')

# Formatting the chart
plt.title('Top 10 Wicket Takers (Bowler-Credited Wickets)', fontsize=16, pad=20)
plt.xlabel('Total Wickets', fontsize=12)
plt.ylabel('Bowler', fontsize=12)

plt.tight_layout()
plt.savefig('top_10_wickets.png')
plt.show()

# ===============================================================================================
# Calculate bowler runs
deliveries_df['bowler_runs'] = deliveries_df['runs_off_bat'] + deliveries_df['wides'] + deliveries_df['noballs']

# Identify legal balls
deliveries_df['is_legal_ball'] = ((deliveries_df['wides'] == 0) & (deliveries_df['noballs'] == 0)).astype(int)

# Aggregate data per bowler
stats = deliveries_df.groupby('bowler').agg(
    total_runs=('bowler_runs', 'sum'),
    total_legal_balls=('is_legal_ball', 'sum')
).reset_index()

# Calculate Overs and Economy Rate
stats['overs'] = stats['total_legal_balls'] / 6
# Filter for minimum 15 overs to exclude part-time bowlers with low volume
top_economy = stats[stats['overs'] >= 15].copy()
top_economy['economy_rate'] = top_economy['total_runs'] / top_economy['overs']

#  Top 10 (Lowest Economy)
top_10 = top_economy.sort_values(by='economy_rate').head(10)

# Visualization
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

plot = sns.barplot(data=top_10, x='economy_rate', y='bowler', palette='mako')

# Add 4-decimal place labels to the bars
for i, val in enumerate(top_10['economy_rate']):
    plt.text(val + 0.05, i, f'{val:.4f}', va='center', fontweight='bold')

plt.title('Top 10 Bowlers by Economy Rate (Min. 15 Overs)', fontsize=15)
plt.xlabel('Economy Rate (Runs per Over)')
plt.ylabel('Bowler')
plt.tight_layout()
plt.savefig('Top10_bowlers.png')
plt.show()

# ===============================================================================================
# Team performance
# ===============================================================================================


team_wins = matches_df['winner'].value_counts().head(10)

team_wins_df = team_wins.reset_index()
team_wins_df.columns = ['Team', 'Wins']

sns.barplot(x='Wins', y='Team', data=team_wins_df, palette='Set2')

plt.title("Top Teams by Number of Wins")
plt.xlabel("Wins")
plt.ylabel("Teams")
plt.savefig('Top_teams.png')
plt.show()

# ==============================================================================================
# Toss Analysis
# ==============================================================================================

# Average winning margins
avg_runs = matches_df['winner_runs'].mean()
avg_wickets = matches_df['winner_wickets'].mean()

print(f"\nAverage Winning Margin (Runs): {avg_runs:.4f}")
print(f"Average Winning Margin (Wickets): {avg_wickets:.4f}")

# Impact of winning the toss on the match outcome
matches_df['toss_match_winner'] = matches_df['toss_winner'] == matches_df['winner']
toss_win_impact = matches_df['toss_match_winner'].value_counts(normalize=True) * 100

print("\nToss Win Impact on Match Result (Percentage):")
print(toss_win_impact.map('{:,.4f}%'.format))

# Plot toss decision
plt.figure(figsize=(8, 6))
sns.countplot(data=matches_df, x='toss_decision', palette='coolwarm')
plt.title('Distribution of Toss Decisions')
plt.xlabel('Decision')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('toss_decision.png')