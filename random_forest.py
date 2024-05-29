import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3, pandas as pd

# Łączenie z bazą danych
conn = sqlite3.connect('database.sqlite')
query_match = "SELECT * FROM Match WHERE B365H IS NOT NULL AND B365D IS NOT NULL AND B365A IS NOT NULL;"
query_team = "SELECT * FROM Team;"
matches = pd.read_sql(query_match, conn)
teams = pd.read_sql(query_team, conn)
conn.close()

def get_team_name(api_id):
    team = teams[teams['team_api_id'] == int(api_id)]
    return team['team_long_name'].iloc[0]

# Dodanie kolumny 'result' na podstawie liczby strzelonych bramek
def determine_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 'H'
    elif row['home_team_goal'] < row['away_team_goal']:
        return 'A'
    else:
        return 'D'

matches['result'] = matches.apply(determine_result, axis=1)

# Filtruj mecze z sezonu 2015/2016 jako dane testowe
matches['date'] = pd.to_datetime(matches['date'])
train_data = matches[matches['season'] != '2015/2016']
test_data = matches[matches['season'] == '2015/2016']

def create_features(data):
    data = data.sort_values(by=['date'])
    data['home_team_goal_avg'] = data.groupby('home_team_api_id')['home_team_goal'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())
    data['away_team_goal_avg'] = data.groupby('away_team_api_id')['away_team_goal'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())
    data['home_team_conceded_avg'] = data.groupby('home_team_api_id')['away_team_goal'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())
    data['away_team_conceded_avg'] = data.groupby('away_team_api_id')['home_team_goal'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())

    # Dodanie nowych cech
    data['home_attack_vs_away_defense'] = data['home_team_goal_avg'] - data['away_team_conceded_avg']
    data['away_attack_vs_home_defense'] = data['away_team_goal_avg'] - data['home_team_conceded_avg']

    data = data.dropna(
        subset=['home_team_goal_avg', 'away_team_goal_avg', 'home_team_conceded_avg', 'away_team_conceded_avg',
                'home_attack_vs_away_defense', 'away_attack_vs_home_defense'])

    return data

train_data = create_features(train_data)
test_data = create_features(test_data)

features = ['home_team_goal_avg', 'away_team_goal_avg', 'home_team_conceded_avg', 'away_team_conceded_avg',
            'home_attack_vs_away_defense', 'away_attack_vs_home_defense']

X_train = train_data[features]
y_train_home = train_data['home_team_goal']
y_train_away = train_data['away_team_goal']

X_test = test_data[features]
y_test_home = test_data['home_team_goal']
y_test_away = test_data['away_team_goal']

# Model lasu losowego dla drużyny gospodarzy
model_home_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_home_rf.fit(X_train, y_train_home)

# Model lasu losowego dla drużyny gości
model_away_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_away_rf.fit(X_train, y_train_away)

predicted_home_goals_rf = model_home_rf.predict(X_test)
predicted_away_goals_rf = model_away_rf.predict(X_test)

test_data_reset = test_data.reset_index(drop=True)
print(test_data_reset)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_probability(match_id, index):
    match = test_data_reset.iloc[index]
    match_predicted_home_goals = predicted_home_goals_rf[index]
    match_predicted_away_goals = predicted_away_goals_rf[index]

    print(f'Przewidywana liczba bramek gospodarzy: {match_predicted_home_goals:.2f}')
    print(f'Przewidywana liczba bramek gości: {match_predicted_away_goals:.2f}')
    print(f"Rzeczywisty wynik: {get_team_name(match['home_team_api_id'])} {match['home_team_goal']} - {match['away_team_goal']} {get_team_name(match['away_team_api_id'])}")



    home_win_prob = sigmoid(match_predicted_home_goals - match_predicted_away_goals)
    away_win_prob = sigmoid(match_predicted_away_goals - match_predicted_home_goals)

    # Normalizacja prawdopodobieństw do 100%
    total_prob = home_win_prob + away_win_prob
    home_win_prob = (home_win_prob / total_prob) * 100
    away_win_prob = (away_win_prob / total_prob) * 100

    print(f'Szansa na zwycięstwo gospodarzy: {home_win_prob:.2f}%')
    print(f'Szansa na zwycięstwo gości: {away_win_prob:.2f}%')

    if home_win_prob > away_win_prob:
        return 'H'
    elif away_win_prob > home_win_prob:
        return 'A'
    return 'D'

good_guesses = 0
total_guesses = test_data.shape[0]



for index, row in test_data_reset.iterrows():
    if row['result'] == calculate_probability(row['id'], index):
        good_guesses += 1

print(f"accuracy:", good_guesses / total_guesses)

