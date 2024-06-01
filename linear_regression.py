import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
from kelly_criterion import kelly_criterion

# Łączenie z bazą danych
conn = sqlite3.connect('database.sqlite')
query_match = "SELECT * FROM Match WHERE B365A IS NOT NULL AND B365H IS NOT NULL;"
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

# Model regresji liniowej dla drużyny gospodarzy
model_home = LinearRegression()
model_home.fit(X_train, y_train_home)

# Model regresji liniowej dla drużyny gości
model_away = LinearRegression()
model_away.fit(X_train, y_train_away)


def calculate_probability(match_id):
    # Przykładowy mecz
    sample_match = test_data[
        test_data['id'] == match_id]  # Można też podać konkretny mecz, np. test_data[test_data['id'] == specific_id]

    # Wyodrębnienie cech dla przykładowego meczu
    sample_features = sample_match[features]

    # Predykcja liczby bramek dla drużyn gospodarzy i gości
    predicted_home_goals = model_home.predict(sample_features)[0]
    predicted_away_goals = model_away.predict(sample_features)[0]

    print(f'Przewidywana liczba bramek gospodarzy: {predicted_home_goals:.2f}')
    print(f'Przewidywana liczba bramek gości: {predicted_away_goals:.2f}')
    print(
        f"Rzeczywisty wynik: {get_team_name(sample_match['home_team_api_id'].iloc[0])} {sample_match['home_team_goal'].iloc[0]} - {sample_match['away_team_goal'].iloc[0]} {get_team_name(sample_match['away_team_api_id'].iloc[0])}")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Obliczanie prawdopodobieństw
    goal_difference = predicted_home_goals - predicted_away_goals
    home_win_prob = sigmoid(predicted_home_goals - predicted_away_goals)
    away_win_prob = sigmoid(predicted_away_goals - predicted_home_goals)
    draw_prob = 1 - (home_win_prob + away_win_prob)

    # Zakładamy, że remis ma miejsce, gdy różnica bramek jest bliska zeru
    # Użyjemy rozkładu normalnego do modelowania tego prawdopodobieństwa
    mean = 0
    std_dev = 0.1  # Standardowe odchylenie, można dostosować na podstawie danych
    draw_prob = (1 - (home_win_prob + away_win_prob)) * (
            np.exp(-0.5 * (goal_difference / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi)))

    # Normalizacja prawdopodobieństw do 100%
    total_prob = home_win_prob + away_win_prob + draw_prob
    home_win_prob = (home_win_prob / total_prob) * 100
    away_win_prob = (away_win_prob / total_prob) * 100
    draw_prob = (draw_prob / total_prob) * 100

    print(f'Szansa na zwycięstwo gospodarzy: {home_win_prob:.2f}%')
    print(f'Szansa na zwycięstwo gości: {away_win_prob:.2f}%')
    print(f'Szansa na remis: {draw_prob:.2f}%')

    if home_win_prob > away_win_prob:
        return 'H', home_win_prob / 100
    return 'A', away_win_prob / 100


def calculate_accuracy():
    good_guesses = 0
    total_guesses = test_data[test_data['result'] != 'D'].shape[0]

    for index, row in test_data.iterrows():
        if row['result'] == calculate_probability(row['id'])[0]:
            good_guesses += 1

    print(f"accuracy:", good_guesses / total_guesses)


def bet_with_kelly_criterion(match_id, budget):
    predicted_winner, probability = calculate_probability(match_id)
    sample_match = test_data[test_data['id'] == match_id]
    home_odds = sample_match['B365H'].iloc[0]
    print("Kurs na gospodarza", home_odds)
    away_odds = sample_match['B365A'].iloc[0]
    print("Kurs na gości", away_odds)

    if predicted_winner == 'H':
        odd_in_consideration = home_odds
        suggested_bet = kelly_criterion(probability, odd_in_consideration)
    elif predicted_winner == 'A':
        odd_in_consideration = away_odds
        suggested_bet = kelly_criterion(probability, odd_in_consideration)

    suggested_stake = budget * suggested_bet

    print("Sugerowana stawka zakładu to: ", suggested_stake,
          f'na drużynę {"Gospodarzy" if predicted_winner == "H" else "gości"}')

    return suggested_stake, predicted_winner, odd_in_consideration


random_matches = test_data.sample(n=10)

total_diff = 0

for i in range(1):
    budget = 10
    for index, match in random_matches.iterrows():
        suggested_bet, predicted_winner, odd = bet_with_kelly_criterion(match['id'], budget)
        print(suggested_bet, odd)
        print("Budżet przed zakładem", budget)
        if match['result'] == predicted_winner and suggested_bet > 0:
            budget -= suggested_bet
            budget += suggested_bet * odd
        elif suggested_bet < 0:
            print("Nie obstawiono meczu")
        else:
            budget -= suggested_bet
        print("Budżet po zakladzie:", budget)

    print("Budżet koncowy", budget)
    total_diff += budget -10


print(total_diff)
