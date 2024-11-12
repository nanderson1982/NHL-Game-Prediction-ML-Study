"""
Author: Nathan Anderson
Description: Helper function for NHL_Game_Prediction.ipynb
Start Date: 10/26/2024
"""


# -------------------------- Python coding libraries -----------------------------------

import pandas as pd
import numpy as np
import io
import urllib.request
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, make_scorer


# --------------------- Data Capture, Engineering, Slicing -----------------------------

# Function to do an api call and return raw data from an url
def api_call(url: str):
    """Returns the result of an API call."""
   
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers= {'User-Agent': user_agent,} 

    request = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    rawData = pd.read_csv(io.StringIO(data.decode('utf-8')))
    
    return rawData


def team_acronyms(df):
    """Function to fix duplicate acronyms and map acronyms for team relocation."""
    
    # Define the mapping for the replacements
    team_mapping = {
        'S.J': 'SJS',
        'N.J': 'NJD',
        'T.B': 'TBL',
        'L.A': 'LAK',
        'ATL': 'WPG',
        'ARI': 'UTA'
    }
    
    # Replacing duplicate team name acronyms across multiple columns
    columns_to_replace = ['opposingTeam', 'team', 'name', 'playerTeam']

    # Apply the replace function across all the relevant columns
    for col in columns_to_replace:
        df[col] = df[col].replace(team_mapping)
    
    return df  # Return the modified DataFrame



def basic_feat_eng(df):
    """ Creating new basic features. """
    
    # Adding columns
    shootout_game = np.where((df['situation'] == 'all') & (df['goalsFor'] == df['goalsAgainst']), 1, 0)
    df.insert(loc = 6, column = 'Shootout Game', value = shootout_game)

    ot_game = np.where(df['iceTime'] > 3600.0, 1, 0)
    df.insert(loc = 7, column = 'OT Game', value = ot_game)

    win = np.where(df['goalsFor'] > df['goalsAgainst'], 1, 0)
    df.insert(loc = 8, column = 'Win', value = win)

    loss = np.where((df['OT Game'] == 0) & (df['goalsFor'] < df['goalsAgainst']), 1, 0)
    df.insert(loc = 9, column = 'Loss', value = loss)

    tie_score = np.where(df['goalsFor'] == df['goalsAgainst'], 1,0)
    df.insert(loc = 11, column = 'TieScore', value = tie_score)

    # ot_loss = np.where((df['OT Game'] == 1) & (df['goalsFor'] < df['goalsAgainst']), 1, 0)
    # df.insert(loc = 10, column = 'OT Loss', value = ot_loss)
    
    # Save percentage
    save_percentage = (df['shotsOnGoalAgainst'] - df['goalsAgainst']) / df['shotsOnGoalAgainst']
    df.insert(loc = 10, column = 'SavePercentage', value = save_percentage)

    # Adding date columns
    df['gameDate'] = pd.to_datetime(df['gameDate'],format='%Y%m%d')
    #df['year'] = pd.DatetimeIndex(df['gameDate']).year
    #df['month'] = pd.DatetimeIndex(df['gameDate']).month
    #df['day'] = pd.DatetimeIndex(df['gameDate']).day

    # Adding new columns based on categorical columns changing to numerical numerical
    #le = preprocessing.LabelEncoder()
    #df['home_or_away#'] = le.fit_transform(df['home_or_away'])
    #df['team#'] = le.fit_transform(df['team'])
    #df['opposingTeam#'] = le.fit_transform(df['opposingTeam'])

    return df



def calculate_metrics(df):
    """Function to calculate various metrics for NHL games."""

    # Total playoff and regular season games
    total_playoff_games = df[df['playoffGame'] == 1].shape[0]
    total_regular_season_games = df[df['playoffGame'] == 0].shape[0]

    # Games by season
    games_by_season = df.groupby('season').size()

    # Games by home or away
    games_by_home_away = df.groupby('home_or_away').size()

    # Games by team
    games_by_team = df.groupby('team').size()

    # Total wins by team
    total_wins_by_team = df.groupby('team')['Win'].sum()

    # Games by home/away and season
    games_by_home_away_season = df.groupby(['season', 'home_or_away']).size()

    # Combine metrics into a summary dictionary
    metrics_summary = {
        "Total Playoff Games": total_playoff_games,
        "Total Regular Season Games": total_regular_season_games,
        "Games by Season": games_by_season.to_dict(),
        "Games by Home/Away": games_by_home_away.to_dict(),
        "Games by Team": games_by_team.to_dict(),
        "Total Wins by Team": total_wins_by_team.to_dict(),
        "Games by Home/Away and Season": games_by_home_away_season.to_dict()
    }

    # Print each metric in a readable format
    for metric, result in metrics_summary.items():
        print(f"\n{metric}:")
        print(result)

    return metrics_summary



# Define the split_data function for training and testing only
def split_data(features, labels, test_size=0.2, random_state=10):
    """
    Purpose:
        Split data into Training and Testing datasets.

    Args:
        features (numpy.ndarray or pandas.DataFrame): Input features.
        labels (numpy.ndarray or pandas.Series): Target labels.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int): Random seed for reproducibility. Defaults to 10.

    Returns:
        Tuple[numpy.ndarray or pandas.DataFrame]: X_train, X_test
        Tuple[numpy.ndarray or pandas.Series]: y_train, y_test
    """
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)

    return X_train, X_test, y_train, y_test






def evaluate_logistic_regression(features, labels, n_splits=5, apply_pca=False, random_state=10):
    """
    Purpose:
        Evaluate a logistic regression model using stratified k-fold cross-validation.

    Args:
        features (numpy.ndarray or pandas.DataFrame): Input features.
        labels (numpy.ndarray or pandas.Series): Target labels.
        n_splits (int): Number of folds for cross-validation. Defaults to 5.
        apply_pca (bool): Whether to apply PCA for dimensionality reduction. Defaults to False.
        random_state (int): Random seed for reproducibility. Defaults to 10.

    Returns:
        dict: A dictionary containing average accuracy, log loss, and AUC scores from cross-validation.
    """
    
    # Initialize scaler and logistic regression model
    scaler = StandardScaler()
    logistic_model = LogisticRegression(random_state=random_state, max_iter=1000)

    # Standardize the features
    standardized_features = scaler.fit_transform(features)
    
    # Apply PCA if selected
    if apply_pca:
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        standardized_features = pca.fit_transform(standardized_features)

    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    accuracy_scores = []
    log_loss_scores = []
    auc_scores = []

    # Perform cross-validation
    for train_index, test_index in skf.split(standardized_features, labels):
        X_train, X_test = standardized_features[train_index], standardized_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Fit the logistic regression model
        logistic_model.fit(X_train, y_train)

        # Predict probabilities and classes
        y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]
        y_pred_class = logistic_model.predict(X_test)

        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred_class))
        log_loss_scores.append(log_loss(y_test, y_pred_prob))
        auc_scores.append(roc_auc_score(y_test, y_pred_prob))

    # Average metrics
    results = {
        'accuracy': np.mean(accuracy_scores),
        'log_loss': np.mean(log_loss_scores),
        'auc': np.mean(auc_scores)
    }

    return results

