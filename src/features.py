import pandas as pd
import os


RAW_DATA = "data/processed/matches_normalized.csv"
PROCESSED_DATA = "data/processed/epl_features.csv"

os.makedirs("data/processed", exist_ok=True)

def load_and_clean():
    df = pd.read_csv(RAW_DATA)

    # Standardize column names to lower case (from normalized dataset)
    df = df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']]

    # Rename to match our feature engineering pipeline
    df = df.rename(columns={
        'date': 'Date',
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_goals': 'FTHG',
        'away_goals': 'FTAG',
        'result': 'FTR'
    })

    df['Date'] = pd.to_datetime(df['Date'])
    return df


def create_features(df):
    features = []

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    for team in teams:
        team_home = df[df['HomeTeam'] == team][['Date', 'FTHG', 'FTAG', 'FTR']].copy()
        team_away = df[df['AwayTeam'] == team][['Date', 'FTHG', 'FTAG', 'FTR']].copy()

        # Rename columns consistently
        team_home = team_home.rename(columns={'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst', 'FTR': 'Result'})
        team_away = team_away.rename(columns={'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst', 'FTR': 'Result'})

        team_home['Venue'] = 'Home'
        team_away['Venue'] = 'Away'

        team_matches = pd.concat([team_home, team_away]).sort_values('Date')
        team_matches['Team'] = team

        # Rolling averages for form
        team_matches['GF_avg'] = team_matches['GoalsFor'].rolling(5, min_periods=1).mean()
        team_matches['GA_avg'] = team_matches['GoalsAgainst'].rolling(5, min_periods=1).mean()

        features.append(team_matches)

    return pd.concat(features).reset_index(drop=True)

if __name__ == "__main__":
    print("🔄 Loading and cleaning data...")
    df = load_and_clean()

    print("✨ Creating features...")
    features_df = create_features(df)

    # Ensure processed folder exists
    import os
    os.makedirs("data/processed", exist_ok=True)

    # Save features
    features_df.to_csv("data/processed/features.csv", index=False)
    print("✅ Features saved to data/processed/features.csv")


