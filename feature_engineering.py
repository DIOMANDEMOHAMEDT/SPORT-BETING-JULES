import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import unittest

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Models (Copied from etl_pipeline.py for portability) ---
# In a larger project, these models would be in a shared 'models.py' file.
Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'

    match_id = Column(Integer, primary_key=True)
    match_date = Column(DateTime(timezone=True))
    home_team = Column(String)
    away_team = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)
    xG = Column(Float)
    shots = Column(Float)
    possession = Column(Float)

    odds = relationship("Odd", back_populates="match")

    def __repr__(self):
        return f"<Match(id={self.match_id}, home='{self.home_team}', away='{self.away_team}')>"

class Odd(Base):
    __tablename__ = 'odds'

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'))
    bookmaker = Column(String)
    type_pari = Column(String)
    cote = Column(Float)
    horodatage = Column(DateTime(timezone=True))

    match = relationship("Match", back_populates="odds")

    def __repr__(self):
        return f"<Odd(match_id={self.match_id}, bookmaker='{self.bookmaker}', cote={self.cote})>"

# --- Feature Engineering logic will be added below ---

def load_data_from_db(engine) -> (pd.DataFrame, pd.DataFrame):
    """
    Connects to the database using an existing engine and loads the matches and odds tables.

    Args:
        engine: A SQLAlchemy engine instance.

    Returns:
        A tuple containing two DataFrames: (matches_df, odds_df).
    """
    logging.info(f"Loading data from database...")
    try:
        with engine.connect() as connection:
            matches_df = pd.read_sql_table('matches', connection, parse_dates=['match_date'])
            odds_df = pd.read_sql_table('odds', connection, parse_dates=['horodatage'])
            logging.info(f"Loaded {len(matches_df)} matches and {len(odds_df)} odds records.")
            return matches_df, odds_df
    except Exception as e:
        logging.error(f"Failed to load data from database: {e}", exc_info=True)
        # Return empty DataFrames on failure
        return pd.DataFrame(), pd.DataFrame()

class FeatureEngineer:
    """
    Generates analytical features from the raw match and odds data.
    """
    def __init__(self, matches_df: pd.DataFrame, odds_df: pd.DataFrame):
        # Sort matches by date to ensure correct chronological order for rolling stats
        self.matches_df = matches_df.sort_values(by='match_date').copy()
        self.odds_df = odds_df.copy()

    def generate_features(self) -> pd.DataFrame:
        """
        Main method to generate all features and return the enriched DataFrame.
        """
        logging.info("Starting feature engineering process...")

        # 1. Calculate market consensus from odds
        market_consensus_df = self._calculate_market_consensus()

        # 2. Prepare team-level data for rolling calculations
        team_level_df = self._get_team_level_data()

        # 3. Calculate rolling stats and form
        team_stats_df = self._calculate_rolling_stats(team_level_df)

        # 4. Merge team stats back into the main match DataFrame
        enriched_df = self._merge_features(team_stats_df)

        # 5. Add match-level features
        enriched_df = self._add_match_importance(enriched_df)
        enriched_df = self._add_rest_days(enriched_df, team_stats_df)

        # 6. Merge market consensus
        enriched_df = enriched_df.merge(market_consensus_df, on='match_id', how='left')

        logging.info("Feature engineering complete.")
        return enriched_df.fillna(0) # Fill any remaining NaNs with 0

    def _calculate_market_consensus(self) -> pd.DataFrame:
        """Calculates the average odds for key markets."""
        relevant_odds = self.odds_df[self.odds_df['type_pari'].isin(['Over 2.5', 'Over 1.5'])]
        consensus = relevant_odds.groupby(['match_id', 'type_pari'])['cote'].mean().reset_index()

        pivot_consensus = consensus.pivot(index='match_id', columns='type_pari', values='cote').reset_index()
        pivot_consensus.columns.name = None
        pivot_consensus.rename(columns={
            'Over 1.5': 'market_consensus_O1.5',
            'Over 2.5': 'market_consensus_O2.5'
        }, inplace=True)

        return pivot_consensus

    def _get_team_level_data(self) -> pd.DataFrame:
        """Transforms the match-level DataFrame into a team-level one."""
        home = self.matches_df.rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'goals_for', 'away_score': 'goals_against'})
        away = self.matches_df.rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'goals_for', 'home_score': 'goals_against'})
        home['was_home'] = 1
        away['was_home'] = 0

        team_level_df = pd.concat([home, away], ignore_index=True)
        team_level_df = team_level_df.sort_values(by=['team', 'match_date'])

        team_level_df['result'] = np.where(team_level_df['goals_for'] > team_level_df['goals_against'], 'W',
                                           np.where(team_level_df['goals_for'] == team_level_df['goals_against'], 'D', 'L'))

        return team_level_df

    def _calculate_rolling_stats(self, team_level_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rolling averages for various stats."""
        df = team_level_df.copy()
        grouped = df.groupby('team')

        for window in [5, 10]:
            # Use shift(1) to ensure we are using data from *before* the current match
            df[f'avg_goals_for_{window}m'] = grouped['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_{window}m'] = grouped['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_xG_{window}m'] = grouped['xG'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Home/Away specific stats
            df[f'avg_goals_for_home_{window}m'] = df.where(df['was_home'] == 1).groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_for_away_{window}m'] = df.where(df['was_home'] == 0).groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_home_{window}m'] = df.where(df['was_home'] == 1).groupby('team')['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_away_{window}m'] = df.where(df['was_home'] == 0).groupby('team')['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Forward fill the location-specific stats to apply them to the current match's context
        ffill_cols = [col for col in df.columns if '_home_' in col or '_away_' in col]
        df[ffill_cols] = grouped[ffill_cols].transform(lambda x: x.ffill())

        form_map = {'W': 3, 'D': 1, 'L': 0}
        df['form_points'] = df['result'].map(form_map)
        df['form_score_5m'] = grouped['form_points'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())

        return df

    def _merge_features(self, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merges the calculated team features back into the original match DataFrame."""
        home_stats = team_stats_df[team_stats_df['was_home'] == 1]
        away_stats = team_stats_df[team_stats_df['was_home'] == 0]

        stats_cols = [col for col in team_stats_df.columns if 'avg_' in col or 'form_' in col]

        home_stats_renamed = home_stats[['match_id', 'team'] + stats_cols].rename(columns={col: f'home_{col}' for col in stats_cols})
        away_stats_renamed = away_stats[['match_id', 'team'] + stats_cols].rename(columns={col: f'away_{col}' for col in stats_cols})

        df = self.matches_df.merge(home_stats_renamed, left_on=['match_id', 'home_team'], right_on=['match_id', 'team'], how='left')
        df = df.merge(away_stats_renamed, left_on=['match_id', 'away_team'], right_on=['match_id', 'team'], how='left')

        df.drop(columns=['team_x', 'team_y'], inplace=True, errors='ignore')
        return df

    def _add_match_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a placeholder binary feature for match importance (e.g., derbies)."""
        derby_teams = [('United', 'City'), ('Chelsea', 'Arsenal', 'Spurs')]

        def is_derby(row):
            for group in derby_teams:
                # Check if team names from the same group are playing each other
                in_group = [team for team in group if team in row['home_team'] or team in row['away_team']]
                if len(in_group) > 1 and any(t in row['home_team'] for t in in_group) and any(t in row['away_team'] for t in in_group):
                    return 1
            return 0

        df['is_derby'] = df.apply(is_derby, axis=1)
        return df

    def _add_rest_days(self, enriched_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rest days for each team and merges it into the enriched DataFrame."""
        team_stats_df['rest_days'] = team_stats_df.groupby('team')['match_date'].diff().dt.days

        home_rest = team_stats_df[team_stats_df['was_home'] == 1][['match_id', 'rest_days']].rename(columns={'rest_days': 'home_rest_days'})
        away_rest = team_stats_df[team_stats_df['was_home'] == 0][['match_id', 'rest_days']].rename(columns={'rest_days': 'away_rest_days'})

        df = enriched_df.merge(home_rest, on='match_id', how='left')
        df = df.merge(away_rest, on='match_id', how='left')

        return df

# --- Demo Setup (to make the script runnable) ---
# The following classes are simplified versions from etl_pipeline.py
# to populate a demo database. To make rolling stats meaningful, we generate more data.

class DemoDataIngestor:
    def get_matches_data(self):
        # Create a more extensive dataset for meaningful rolling averages
        teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F']
        data = []
        for i in range(50): # 50 matches
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            if home_team == away_team: away_team = teams[(i + 2) % len(teams)]

            match = {
                'id': 1000 + i,
                'utcDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(i * 2, 'days'),
                'homeTeam': {'name': home_team},
                'awayTeam': {'name': away_team},
                'score': {'fullTime': {'homeTeam': (i % 4), 'awayTeam': (i % 3)}},
                'stats': {'xG': 1.0 + ((i % 5) * 0.3)}
            }
            data.append(match)

        return pd.DataFrame(data)

    def get_odds_data(self, match_ids):
        mock_odds_data = []
        for match_id in match_ids:
            mock_odds_data.extend([
                {'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 2.5', 'cote': 1.85 + ((match_id % 5) * 0.05), 'timestamp': datetime.now(timezone.utc).isoformat()},
                {'match_id': match_id, 'bookmaker': 'BookieB', 'type_pari': 'Over 2.5', 'cote': 1.90 + ((match_id % 5) * 0.05), 'timestamp': datetime.now(timezone.utc).isoformat()},
                {'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 1.5', 'cote': 1.30, 'timestamp': datetime.now(timezone.utc).isoformat()},
            ])
        return pd.DataFrame(mock_odds_data)

class DemoDataCleaner:
    def clean_matches_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['home_team'] = df['homeTeam'].apply(lambda x: x['name'])
        df['away_team'] = df['awayTeam'].apply(lambda x: x['name'])
        df['home_score'] = df['score'].apply(lambda x: x['fullTime']['homeTeam'])
        df['away_score'] = df['score'].apply(lambda x: x['fullTime']['awayTeam'])
        df['xG'] = df['stats'].apply(lambda x: x.get('xG', 0))
        df['match_date'] = pd.to_datetime(df['utcDate'], utc=True)
        clean_df = df[['id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'xG']].copy()
        clean_df.rename(columns={'id': 'match_id'}, inplace=True)
        clean_df['shots'] = 10.0 # Dummy data
        clean_df['possession'] = 50.0 # Dummy data
        return clean_df

    def clean_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['cote'] = pd.to_numeric(df['cote'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.dropna(subset=['match_id'], inplace=True)
        df['match_id'] = df['match_id'].astype(int)
        return df

def setup_demo_database(engine):
    """Populates the database using the given engine."""
    logging.info("Setting up demo database with sample data...")
    Base.metadata.create_all(engine)

    ingestor = DemoDataIngestor()
    cleaner = DemoDataCleaner()

    raw_matches = ingestor.get_matches_data()
    clean_matches = cleaner.clean_matches_data(raw_matches)

    match_ids = clean_matches['match_id'].tolist()
    raw_odds = ingestor.get_odds_data(match_ids)
    clean_odds = cleaner.clean_odds_data(raw_odds)

    with engine.connect() as connection:
        with connection.begin(): # Use a transaction
            clean_matches.to_sql('matches', connection, if_exists='replace', index=False)
            clean_odds.to_sql('odds', connection, if_exists='replace', index=False)

    logging.info("Demo database setup complete.")


# --- Main Execution Block ---
def main_features():
    """
    Main function to run the feature engineering script.
    """
    DB_URL = "sqlite:///:memory:"

    # 1. Create a single engine to be shared for the in-memory database
    engine = create_engine(DB_URL)

    # 2. Setup a temporary DB with sample data using the shared engine
    setup_demo_database(engine)

    # 3. Load the data back from the DB using the same engine
    matches_df, odds_df = load_data_from_db(engine)

    if matches_df.empty:
        logging.error("No data loaded, cannot proceed with feature engineering.")
        return

    # 4. Generate features
    feature_engineer = FeatureEngineer(matches_df, odds_df)
    enriched_df = feature_engineer.generate_features()

    # 5. Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    logging.info("--- Enriched DataFrame (Head) ---")
    print(enriched_df.head())

    feature_cols = [
        'home_team', 'away_team',
        'home_avg_goals_for_5m', 'away_avg_goals_for_5m',
        'home_form_score_5m', 'away_form_score_5m',
        'home_rest_days', 'away_rest_days',
        'market_consensus_O2.5'
    ]
    existing_feature_cols = [col for col in feature_cols if col in enriched_df.columns]

    logging.info("\n--- Key Features (Tail) ---")
    print(enriched_df[existing_feature_cols].tail())


# --- Unit Tests ---
class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        """Create a small, controlled dataset for testing."""
        matches_data = {
            'match_id': [1, 2, 3, 4],
            'match_date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-08', '2023-01-12']),
            'home_team': ['Team A', 'Team B', 'Team A', 'Team B'],
            'away_team': ['Team B', 'Team A', 'Team C', 'Team D'],
            'home_score': [1, 2, 3, 0],
            'away_score': [0, 2, 1, 1],
            'xG': [1.1, 2.2, 2.8, 0.9],
            'shots': [10, 12, 15, 8],
            'possession': [55, 48, 60, 45]
        }
        self.matches_df = pd.DataFrame(matches_data)

        odds_data = {
            'match_id': [1, 1, 2, 2, 3, 3],
            'bookmaker': ['A', 'B', 'A', 'B', 'A', 'B'],
            'type_pari': ['Over 2.5', 'Over 2.5', 'Over 2.5', 'Over 2.5', 'Over 2.5', 'Over 2.5'],
            'cote': [1.8, 2.0, 1.5, 1.6, 2.1, 2.1],
            'horodatage': pd.to_datetime(['2023-01-01'] * 6)
        }
        self.odds_df = pd.DataFrame(odds_data)

    def test_generate_features(self):
        """Test the main feature generation method."""
        engineer = FeatureEngineer(self.matches_df, self.odds_df)
        enriched_df = engineer.generate_features()

        # --- Assertions ---
        self.assertEqual(len(enriched_df), 4)

        # Test rolling average for Team A's second match (match_id 3)
        # It should be based on its first match (match_id 1), where it was home and scored 1 goal.
        team_a_match_3 = enriched_df[enriched_df['match_id'] == 3]
        # Goals for: Team A scored 1 in match 1 and 2 in match 2. Average is 1.5.
        self.assertEqual(team_a_match_3['home_avg_goals_for_5m'].iloc[0], 1.5)
        # Goals against: Team A conceded 0 in match 1 and 2 in match 2. Average is 1.0.
        self.assertEqual(team_a_match_3['home_avg_goals_against_5m'].iloc[0], 1.0)

        # Test form score for Team A's second match (match_id 3)
        # Match 1 was a win (3 pts), Match 2 was a draw (1 pt). Total = 4.
        self.assertEqual(team_a_match_3['home_form_score_5m'].iloc[0], 4.0)

        # Test rest days for Team A's second match (match_id 3)
        # Previous match for Team A was on 01-05, current is 01-08. Diff = 3 days.
        self.assertEqual(team_a_match_3['home_rest_days'].iloc[0], 3.0)

        # Test rest days for Team B's second match (match_id 4)
        # D4 - D2 = 7 days.
        team_b_match_4 = enriched_df[enriched_df['match_id'] == 4]
        self.assertEqual(team_b_match_4['home_rest_days'].iloc[0], 7.0)

        # Test market consensus for match_id 2
        # Odds were 1.5 and 1.6, so mean is 1.55
        match_2 = enriched_df[enriched_df['match_id'] == 2]
        self.assertAlmostEqual(match_2['market_consensus_O2.5'].iloc[0], 1.55)

        # Test a value that should be 0 (first match for a team)
        match_1 = enriched_df[enriched_df['match_id'] == 1]
        self.assertEqual(match_1['away_avg_goals_for_5m'].iloc[0], 0)
        self.assertEqual(match_1['away_form_score_5m'].iloc[0], 0)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # To run tests, use: python feature_engineering.py test
        sys.argv.pop(1)
        unittest.main(argv=sys.argv, exit=False)
    else:
        main_features()
