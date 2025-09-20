import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Models (Copied for portability) ---
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

class Odd(Base):
    __tablename__ = 'odds'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'))
    bookmaker = Column(String)
    type_pari = Column(String)
    cote = Column(Float)
    horodatage = Column(DateTime(timezone=True))
    match = relationship("Match", back_populates="odds")

# --- Data Loading and Feature Engineering (Copied for portability) ---

def load_data_from_db(engine):
    logging.info("Loading data from database...")
    try:
        with engine.connect() as connection:
            matches_df = pd.read_sql_table('matches', connection, parse_dates=['match_date'])
            odds_df = pd.read_sql_table('odds', connection, parse_dates=['horodatage'])
            return matches_df, odds_df
    except Exception as e:
        logging.error(f"Failed to load data from database: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

class FeatureEngineer:
    def __init__(self, matches_df: pd.DataFrame, odds_df: pd.DataFrame):
        self.matches_df = matches_df.sort_values(by='match_date').copy()
        self.odds_df = odds_df.copy()
    def generate_features(self) -> pd.DataFrame:
        logging.info("Starting feature engineering process...")
        market_consensus_df = self._calculate_market_consensus()
        team_level_df = self._get_team_level_data()
        team_stats_df = self._calculate_rolling_stats(team_level_df)
        enriched_df = self._merge_features(team_stats_df)
        enriched_df = self._add_match_importance(enriched_df)
        enriched_df = self._add_rest_days(enriched_df, team_stats_df)
        enriched_df = enriched_df.merge(market_consensus_df, on='match_id', how='left')
        logging.info("Feature engineering complete.")
        return enriched_df.fillna(0)
    def _calculate_market_consensus(self) -> pd.DataFrame:
        relevant_odds = self.odds_df[self.odds_df['type_pari'].isin(['Over 2.5', 'Over 1.5'])]
        consensus = relevant_odds.groupby(['match_id', 'type_pari'])['cote'].mean().reset_index()
        pivot_consensus = consensus.pivot(index='match_id', columns='type_pari', values='cote').reset_index()
        pivot_consensus.columns.name = None
        pivot_consensus.rename(columns={'Over 1.5': 'market_consensus_O1.5','Over 2.5': 'market_consensus_O2.5'}, inplace=True)
        return pivot_consensus
    def _get_team_level_data(self) -> pd.DataFrame:
        home = self.matches_df.rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'goals_for', 'away_score': 'goals_against'})
        away = self.matches_df.rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'goals_for', 'home_score': 'goals_against'})
        home['was_home'] = 1
        away['was_home'] = 0
        team_level_df = pd.concat([home, away], ignore_index=True)
        team_level_df = team_level_df.sort_values(by=['team', 'match_date'])
        team_level_df['result'] = np.where(team_level_df['goals_for'] > team_level_df['goals_against'], 'W', np.where(team_level_df['goals_for'] == team_level_df['goals_against'], 'D', 'L'))
        return team_level_df
    def _calculate_rolling_stats(self, team_level_df: pd.DataFrame) -> pd.DataFrame:
        df = team_level_df.copy()
        grouped = df.groupby('team')
        for window in [5, 10]:
            df[f'avg_goals_for_{window}m'] = grouped['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_{window}m'] = grouped['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_xG_{window}m'] = grouped['xG'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_for_home_{window}m'] = df.where(df['was_home'] == 1).groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_for_away_{window}m'] = df.where(df['was_home'] == 0).groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_home_{window}m'] = df.where(df['was_home'] == 1).groupby('team')['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_away_{window}m'] = df.where(df['was_home'] == 0).groupby('team')['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        ffill_cols = [col for col in df.columns if '_home_' in col or '_away_' in col]
        df[ffill_cols] = grouped[ffill_cols].transform(lambda x: x.ffill())
        form_map = {'W': 3, 'D': 1, 'L': 0}
        df['form_points'] = df['result'].map(form_map)
        df['form_score_5m'] = grouped['form_points'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        return df
    def _merge_features(self, team_stats_df: pd.DataFrame) -> pd.DataFrame:
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
        derby_teams = [('United', 'City'), ('Chelsea', 'Arsenal', 'Spurs')]
        def is_derby(row):
            for group in derby_teams:
                in_group = [team for team in group if team in row['home_team'] or team in row['away_team']]
                if len(in_group) > 1 and any(t in row['home_team'] for t in in_group) and any(t in row['away_team'] for t in in_group): return 1
            return 0
        df['is_derby'] = df.apply(is_derby, axis=1)
        return df
    def _add_rest_days(self, enriched_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        team_stats_df['rest_days'] = team_stats_df.groupby('team')['match_date'].diff().dt.days
        home_rest = team_stats_df[team_stats_df['was_home'] == 1][['match_id', 'rest_days']].rename(columns={'rest_days': 'home_rest_days'})
        away_rest = team_stats_df[team_stats_df['was_home'] == 0][['match_id', 'rest_days']].rename(columns={'rest_days': 'away_rest_days'})
        df = enriched_df.merge(home_rest, on='match_id', how='left')
        df = df.merge(away_rest, on='match_id', how='left')
        return df

class DemoDataIngestor:
    def get_matches_data(self):
        teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F']
        data = []
        for i in range(50):
            home_team, away_team = teams[i % len(teams)], teams[(i + 1) % len(teams)]
            if home_team == away_team: away_team = teams[(i + 2) % len(teams)]
            data.append({'id': 1000 + i, 'utcDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(i * 2, 'days'), 'homeTeam': {'name': home_team}, 'awayTeam': {'name': away_team}, 'score': {'fullTime': {'homeTeam': (i % 4), 'awayTeam': (i % 3)}}, 'stats': {'xG': 1.0 + ((i % 5) * 0.3)}})
        return pd.DataFrame(data)
    def get_odds_data(self, match_ids):
        mock_odds_data = []
        for match_id in match_ids:
            mock_odds_data.extend([{'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 2.5', 'cote': 1.85 + ((match_id % 5) * 0.05), 'timestamp': datetime.now(timezone.utc).isoformat()}, {'match_id': match_id, 'bookmaker': 'BookieB', 'type_pari': 'Over 2.5', 'cote': 1.90 + ((match_id % 5) * 0.05), 'timestamp': datetime.now(timezone.utc).isoformat()}, {'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 1.5', 'cote': 1.30, 'timestamp': datetime.now(timezone.utc).isoformat()}])
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
        clean_df['shots'], clean_df['possession'] = 10.0, 50.0
        return clean_df
    def clean_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['cote'] = pd.to_numeric(df['cote'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.dropna(subset=['match_id'], inplace=True)
        df['match_id'] = df['match_id'].astype(int)
        return df

def setup_demo_database(engine):
    logging.info("Setting up demo database with sample data...")
    Base.metadata.create_all(engine)
    ingestor, cleaner = DemoDataIngestor(), DemoDataCleaner()
    raw_matches = ingestor.get_matches_data()
    clean_matches = cleaner.clean_matches_data(raw_matches)
    match_ids = clean_matches['match_id'].tolist()
    raw_odds = ingestor.get_odds_data(match_ids)
    clean_odds = cleaner.clean_odds_data(raw_odds)
    with engine.connect() as connection:
        with connection.begin():
            clean_matches.to_sql('matches', connection, if_exists='replace', index=False)
            clean_odds.to_sql('odds', connection, if_exists='replace', index=False)
    logging.info("Demo database setup complete.")

# --- Model Training logic ---

def train_model(X: pd.DataFrame, y: pd.Series):
    logging.info("--- Starting Model Training ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lgbm = lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=31)
    calibrated_lgbm = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=5)
    logging.info("Training and calibrating the model...")
    calibrated_lgbm.fit(X_train_scaled, y_train)
    logging.info("Model training and calibration complete.")
    model_payload = {'model': calibrated_lgbm, 'scaler': scaler}
    model_filename = 'over_2_5_goals_model.joblib'
    joblib.dump(model_payload, model_filename)
    logging.info(f"Model and scaler saved to {model_filename}")
    return calibrated_lgbm, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test_scaled, y_test):
    logging.info("--- Evaluating Model ---")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    print(f"  Brier Score: {brier:.4f} (Lower is better)")
    print(f"  AUC:         {auc:.4f} (Higher is better)")
    print(f"  Log Loss:    {logloss:.4f} (Lower is better)")
    return {"brier": brier, "auc": auc, "logloss": logloss}

def predict_match(features: pd.DataFrame, model_path='over_2_5_goals_model.joblib'):
    logging.info(f"--- Making a Single Prediction ---")
    try:
        model_payload = joblib.load(model_path)
        model = model_payload['model']
        scaler = model_payload['scaler']
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please train the model first.")
        return None
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[:, 1]
    logging.info(f"Predicted probability of Over 2.5 goals: {probability[0]:.4f}")
    return probability[0]

def load_and_prepare_data() -> (pd.DataFrame, pd.Series, pd.DataFrame):
    logging.info("--- Starting Data Preparation ---")
    DB_URL = "sqlite:///:memory:"
    engine = create_engine(DB_URL)
    setup_demo_database(engine)
    matches_df, odds_df = load_data_from_db(engine)
    if matches_df.empty:
        logging.error("No data loaded, aborting.")
        return pd.DataFrame(), pd.Series(), None
    feature_engineer = FeatureEngineer(matches_df, odds_df)
    enriched_df = feature_engineer.generate_features()
    enriched_df['total_goals'] = enriched_df['home_score'] + enriched_df['away_score']
    y = (enriched_df['total_goals'] > 2.5).astype(int)
    features_to_drop = ['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'total_goals']
    X = enriched_df.drop(columns=features_to_drop)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X.columns = [str(col) for col in X.columns]
    logging.info(f"Data preparation complete. Feature matrix shape: {X.shape}")
    logging.info(f"Target vector distribution:\n{y.value_counts(normalize=True)}")
    return X, y, enriched_df

def main_training():
    X, y, enriched_df = load_and_prepare_data()
    if X.empty:
        logging.error("Halting execution due to data preparation failure.")
        return
    model, scaler, X_test_scaled, y_test = train_model(X, y)
    evaluate_model(model, X_test_scaled, y_test)
    original_X_test = X.loc[y_test.index]
    sample_features = original_X_test.head(1)
    print("\n--- Demonstrating single prediction on a sample from the test set ---")
    predict_match(features=sample_features)

if __name__ == "__main__":
    main_training()
