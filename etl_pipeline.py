import os
import pandas as pd
import requests
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import unittest
from unittest.mock import patch, MagicMock

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Class Implementation ---

class DataIngestor:
    """
    Handles data ingestion from sports APIs.
    For this example, it uses mock data. In a real-world scenario,
    the methods would make HTTP requests to the actual APIs.
    """
    def __init__(self, api_keys=None):
        """
        Initializes the DataIngestor.
        Args:
            api_keys (dict): A dictionary containing API keys for various services.
                             Example: {'football-data': 'YOUR_KEY', 'odds-api': 'YOUR_KEY'}
        """
        self.api_keys = api_keys or {}
        # Example: self.football_data_key = self.api_keys.get('football-data')

    def get_matches_data(self):
        """
        Fetches match data. This method simulates an API call.
        In a real implementation, you would use the 'requests' library to fetch data from a URL.
        """
        logging.info("Fetching matches data (from mock source)...")
        # TODO: Replace this mock data with a real API call.
        # Example using requests:
        # headers = {'X-Auth-Token': self.football_data_key}
        # response = requests.get('https://api.football-data.org/v2/competitions/PL/matches', headers=headers)
        # response.raise_for_status()  # Raise an exception for bad status codes
        # raw_data = response.json()

        mock_matches_data = {
            'count': 3,
            'matches': [
                {
                    'id': 101,
                    'utcDate': '2023-08-12T19:00:00Z',
                    'status': 'FINISHED',
                    'homeTeam': {'id': 61, 'name': 'Chelsea FC'},
                    'awayTeam': {'id': 64, 'name': 'Liverpool'},
                    'score': {'fullTime': {'homeTeam': 1, 'awayTeam': 1}},
                    'stats': {'xG': 1.2, 'shots': 10, 'possession': 45.5}  # Fictional stats
                },
                {
                    'id': 102,
                    'utcDate': '2023-08-13T15:30:00Z',
                    'status': 'FINISHED',
                    'homeTeam': {'id': 66, 'name': 'Manchester United'},
                    'awayTeam': {'id': 73, 'name': 'Tottenham Hotspur FC'},
                    'score': {'fullTime': {'homeTeam': 2, 'awayTeam': 2}},
                    'stats': {'xG': 2.1, 'shots': 15, 'possession': 55.0}  # Fictional stats
                },
                {
                    'id': 103,
                    'utcDate': '2023-08-14T20:00:00Z',
                    'status': 'FINISHED',
                    'homeTeam': {'id': 57, 'name': 'Arsenal'},
                    'awayTeam': {'id': 65, 'name': 'Manchester City'},
                    'score': {'fullTime': {'homeTeam': None, 'awayTeam': None}},  # Missing data example
                    'stats': {}  # Missing stats
                }
            ]
        }

        # Convert to DataFrame to mimic a more realistic data processing flow
        df = pd.DataFrame(mock_matches_data['matches'])
        logging.info(f"Successfully fetched {len(df)} matches.")
        return df

    def get_odds_data(self, match_ids):
        """
        Fetches odds data for a list of match IDs. This method simulates an API call.
        """
        logging.info(f"Fetching odds data for {len(match_ids)} matches (from mock source)...")
        # TODO: Replace with a real API call to an odds provider.
        # response = requests.get(f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={self.api_keys.get("odds-api")}&regions=uk&markets=h2h,totals')
        # raw_odds = response.json()

        mock_odds_data = []
        for match_id in match_ids:
            mock_odds_data.extend([
                {'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 2.5', 'cote': 1.95, 'timestamp': datetime.now(timezone.utc).isoformat()},
                {'match_id': match_id, 'bookmaker': 'BookieB', 'type_pari': 'Over 2.5', 'cote': 1.98, 'timestamp': datetime.now(timezone.utc).isoformat()},
                {'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 1.5', 'cote': 1.30, 'timestamp': datetime.now(timezone.utc).isoformat()},
                {'match_id': match_id, 'bookmaker': 'BookieB', 'type_pari': 'Over 1.5', 'cote': 1.32, 'timestamp': datetime.now(timezone.utc).isoformat()},
            ])

        df = pd.DataFrame(mock_odds_data)
        logging.info(f"Successfully fetched odds for {len(match_ids)} matches.")
        return df


class DataCleaner:
    """
    Cleans and transforms raw data fetched by the DataIngestor.
    """
    def __init__(self):
        # This mapping is crucial for ensuring team name consistency.
        # It should be expanded based on the data sources used.
        self.team_name_mapping = {
            "Manchester United": "Man United",
            "Tottenham Hotspur FC": "Spurs",
            "Chelsea FC": "Chelsea",
            "Liverpool": "Liverpool",
            "Arsenal": "Arsenal",
            "Manchester City": "Man City"
        }

    def clean_matches_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the matches DataFrame.
        """
        logging.info("Cleaning matches data...")

        # Extract nested data from JSON-like columns
        df['home_team'] = df['homeTeam'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
        df['away_team'] = df['awayTeam'].apply(lambda x: x['name'] if isinstance(x, dict) else None)

        df['home_score'] = df['score'].apply(lambda x: x['fullTime']['homeTeam'] if isinstance(x, dict) and x.get('fullTime') else None)
        df['away_score'] = df['score'].apply(lambda x: x['fullTime']['awayTeam'] if isinstance(x, dict) and x.get('fullTime') else None)

        # Extract stats, handling missing 'stats' dictionary
        df['xG'] = df['stats'].apply(lambda x: x.get('xG') if isinstance(x, dict) else None)
        df['shots'] = df['stats'].apply(lambda x: x.get('shots') if isinstance(x, dict) else None)
        df['possession'] = df['stats'].apply(lambda x: x.get('possession') if isinstance(x, dict) else None)

        # Harmonize team names
        df['home_team'] = df['home_team'].map(self.team_name_mapping).fillna(df['home_team'])
        df['away_team'] = df['away_team'].map(self.team_name_mapping).fillna(df['away_team'])

        # Handle missing values
        # For scores, a value of 0 is a reasonable default.
        df['home_score'] = df['home_score'].fillna(0).astype(int)
        df['away_score'] = df['away_score'].fillna(0).astype(int)
        # For stats, we can fill with 0 or median, depending on the strategy. Here we use 0.
        for col in ['xG', 'shots', 'possession']:
            df[col] = df[col].fillna(0).astype(float)

        # Convert date column to datetime objects in UTC
        df['match_date'] = pd.to_datetime(df['utcDate'], utc=True)

        # Select and rename columns for the final DataFrame
        clean_df = df[['id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'xG', 'shots', 'possession']].copy()
        clean_df.rename(columns={'id': 'match_id'}, inplace=True)

        logging.info("Matches data cleaning complete.")
        return clean_df

    def clean_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the odds DataFrame.
        """
        logging.info("Cleaning odds data...")

        # Ensure correct data types
        df['cote'] = pd.to_numeric(df['cote'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Drop rows where match_id is missing
        df.dropna(subset=['match_id'], inplace=True)
        df['match_id'] = df['match_id'].astype(int)

        logging.info("Odds data cleaning complete.")
        return df


# --- Database Models (SQLAlchemy ORM) ---
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

    # Relationship to odds
    odds = relationship("Odd", back_populates="match")

    def __repr__(self):
        return f"<Match(id={self.match_id}, home='{self.home_team}', away='{self.away_team}')>"

class Odd(Base):
    __tablename__ = 'odds'

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'))
    bookmaker = Column(String)
    type_pari = Column(String) # e.g., 'Over 2.5'
    cote = Column(Float)
    horodatage = Column(DateTime(timezone=True))

    # Relationship to match
    match = relationship("Match", back_populates="odds")

    def __repr__(self):
        return f"<Odd(match_id={self.match_id}, bookmaker='{self.bookmaker}', cote={self.cote})>"


# --- Data Saver Class ---
class DataSaver:
    """
    Saves cleaned data to a PostgreSQL database.
    """
    def __init__(self, db_url):
        """
        Initializes the DataSaver with a database connection.
        Args:
            db_url (str): The database connection string.
                          Example: 'postgresql://user:password@host:port/database'
        """
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_database(self):
        """
        Creates the database tables based on the ORM models.
        """
        logging.info("Creating database tables if they don't exist...")
        Base.metadata.create_all(self.engine)
        logging.info("Tables created successfully.")

    def save_data(self, matches_df: pd.DataFrame, odds_df: pd.DataFrame):
        """
        Saves the cleaned DataFrames to the database.
        Uses an 'upsert' logic to avoid duplicates for matches.
        """
        session = self.Session()
        logging.info(f"Saving {len(matches_df)} matches and {len(odds_df)} odds records to the database.")

        try:
            # Upsert matches
            for _, row in matches_df.iterrows():
                match_id = row['match_id']
                existing_match = session.query(Match).filter_by(match_id=match_id).first()

                match_data = row.to_dict()

                if existing_match:
                    # Update existing record
                    for key, value in match_data.items():
                        setattr(existing_match, key, value)
                else:
                    # Insert new record
                    new_match = Match(**match_data)
                    session.add(new_match)

            # For odds, we can decide on a strategy. A simple one is to delete old odds
            # for the matches being updated and insert the new ones.
            # For this example, we'll just append, assuming we get new odds data each time.
            for _, row in odds_df.iterrows():
                # First, ensure the match exists to satisfy the foreign key constraint
                match_id = row['match_id']
                match_exists = session.query(Match).filter_by(match_id=match_id).first()
                if match_exists:
                    new_odd = Odd(
                        match_id=row['match_id'],
                        bookmaker=row['bookmaker'],
                        type_pari=row['type_pari'],
                        cote=row['cote'],
                        horodatage=row['timestamp'] # Column name in df is 'timestamp'
                    )
                    session.add(new_odd)
                else:
                    logging.warning(f"Skipping odds for non-existent match_id: {match_id}")

            session.commit()
            logging.info("Data saved successfully.")
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")
            session.rollback()
            raise
        finally:
            session.close()


# --- Main Execution ---
def main():
    """
    Main function to run the ETL pipeline.
    """
    logging.info("Starting ETL pipeline...")

    # --- Configuration ---
    # For demonstration, we use an in-memory SQLite database.
    # To use PostgreSQL, replace this with your actual database URL, e.g.:
    # DB_URL = "postgresql://user:password@localhost:5432/sports_betting_db"
    # Make sure you have created the database 'sports_betting_db' beforehand.
    DB_URL = "sqlite:///:memory:"

    # API keys would be passed here in a real scenario
    # api_keys = {"football-data": "YOUR_KEY", "odds-api": "YOUR_KEY"}

    # --- Initialization ---
    ingestor = DataIngestor()
    cleaner = DataCleaner()
    saver = DataSaver(db_url=DB_URL)

    try:
        # 1. Create database schema
        saver.create_database()

        # 2. Ingest data
        raw_matches_df = ingestor.get_matches_data()

        # 3. Clean data
        clean_matches_df = cleaner.clean_matches_data(raw_matches_df)

        # Ensure we only fetch odds for matches we have cleaned successfully
        match_ids = clean_matches_df['match_id'].unique().tolist()
        if not match_ids:
            logging.warning("No valid match IDs found after cleaning. Skipping odds processing.")
            return

        raw_odds_df = ingestor.get_odds_data(match_ids=match_ids)
        clean_odds_df = cleaner.clean_odds_data(raw_odds_df)

        # 4. Save data
        saver.save_data(matches_df=clean_matches_df, odds_df=clean_odds_df)

        logging.info("ETL pipeline finished successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the ETL pipeline execution: {e}", exc_info=True)

# --- Unit Tests ---
class TestETLPipeline(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.ingestor = DataIngestor()
        self.cleaner = DataCleaner()
        # Use an in-memory SQLite database for testing, which is clean for every test.
        self.saver = DataSaver(db_url="sqlite:///:memory:")
        self.saver.create_database()

    def test_clean_matches_data(self):
        """Test the data cleaning process for matches."""
        raw_df = self.ingestor.get_matches_data()
        clean_df = self.cleaner.clean_matches_data(raw_df)

        # Check columns
        expected_cols = ['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'xG', 'shots', 'possession']
        self.assertTrue(all(col in clean_df.columns for col in expected_cols))

        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['match_id']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(clean_df['match_date']))
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['home_score']))

        # Check team name harmonization
        self.assertIn("Man United", clean_df['home_team'].values)
        self.assertNotIn("Manchester United", clean_df['home_team'].values)

        # Check handling of missing data (match 103)
        missing_data_row = clean_df[clean_df['match_id'] == 103]
        self.assertEqual(missing_data_row['home_score'].iloc[0], 0)
        self.assertEqual(missing_data_row['away_score'].iloc[0], 0)
        self.assertEqual(missing_data_row['xG'].iloc[0], 0.0)

    def test_clean_odds_data(self):
        """Test the data cleaning process for odds."""
        raw_df = self.ingestor.get_odds_data(match_ids=[101, 102])
        clean_df = self.cleaner.clean_odds_data(raw_df)

        # Check data types
        self.assertTrue(pd.api.types.is_float_dtype(clean_df['cote']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(clean_df['timestamp']))
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['match_id']))

    def test_full_pipeline_with_in_memory_db(self):
        """
        Test the full pipeline interaction, saving to and reading from an in-memory DB.
        This is more of an integration test.
        """
        # Run the pipeline with the test saver
        ingestor = DataIngestor()
        cleaner = DataCleaner()

        raw_matches = ingestor.get_matches_data()
        clean_matches = cleaner.clean_matches_data(raw_matches)

        match_ids = clean_matches['match_id'].tolist()
        raw_odds = ingestor.get_odds_data(match_ids)
        clean_odds = cleaner.clean_odds_data(raw_odds)

        self.saver.save_data(clean_matches, clean_odds)

        # Verify data was saved correctly
        session = self.saver.Session()
        match_count = session.query(Match).count()
        odds_count = session.query(Odd).count()

        self.assertEqual(match_count, 3)
        # 4 odds per match * 3 matches = 12
        self.assertEqual(odds_count, 12)

        # Check a specific record
        test_match = session.query(Match).filter_by(match_id=101).first()
        self.assertEqual(test_match.home_team, "Chelsea")
        self.assertEqual(test_match.home_score, 1)
        session.close()


if __name__ == "__main__":
    # This allows running the main ETL function or the tests separately.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # To run tests, use: python etl_pipeline.py test
        # We remove the 'test' argument so unittest doesn't get confused.
        sys.argv.pop(1)
        unittest.main(argv=sys.argv, exit=False)
    else:
        main()
