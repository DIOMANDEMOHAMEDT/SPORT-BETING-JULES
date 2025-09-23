import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
from itertools import combinations
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Models ---
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
class Odd(Base):
    __tablename__ = 'odds'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'))
    bookmaker = Column(String)
    type_pari = Column(String)
    cote = Column(Float)
    horodatage = Column(DateTime(timezone=True))

# --- Data Generation and Feature Engineering classes ---
class DemoDataIngestor:
    def get_matches_data(self, num_matches=200):
        teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H']
        data = []
        for i in range(num_matches):
            home_team, away_team = teams[i % len(teams)], teams[(i + 1) % len(teams)]
            if home_team == away_team: away_team = teams[(i + 2) % len(teams)]
            data.append({'id': 1000 + i, 'utcDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(i * 1, 'days'), 'homeTeam': {'name': home_team}, 'awayTeam': {'name': away_team}, 'score': {'fullTime': {'homeTeam': (i % 4), 'awayTeam': (i % 3)}}, 'stats': {'xG': 1.0 + ((i % 5) * 0.3)}})
        return pd.DataFrame(data)
    def get_odds_data(self, match_ids):
        mock_odds_data = []
        for match_id in match_ids:
            mock_odds_data.extend([{'match_id': match_id, 'bookmaker': 'BookieA', 'type_pari': 'Over 2.5', 'cote': 1.85 + ((match_id % 5) * 0.05), 'timestamp': datetime.now(timezone.utc).isoformat()}])
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
class FeatureEngineer:
    def __init__(self, matches_df: pd.DataFrame):
        self.matches_df = matches_df.sort_values(by='match_date').copy()
    def generate_features(self) -> pd.DataFrame:
        team_level_df = self._get_team_level_data()
        team_stats_df = self._calculate_rolling_stats(team_level_df)
        enriched_df = self._merge_features(team_stats_df)
        return enriched_df.fillna(0)
    def _get_team_level_data(self) -> pd.DataFrame:
        home = self.matches_df.rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'goals_for', 'away_score': 'goals_against'})
        away = self.matches_df.rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'goals_for', 'home_score': 'goals_against'})
        home['was_home'] = 1
        away['was_home'] = 0
        team_level_df = pd.concat([home, away], ignore_index=True)
        team_level_df = team_level_df.sort_values(by=['team', 'match_date'])
        return team_level_df
    def _calculate_rolling_stats(self, team_level_df: pd.DataFrame) -> pd.DataFrame:
        df = team_level_df.copy()
        grouped = df.groupby('team')
        for window in [5, 10]:
            df[f'avg_goals_for_{window}m'] = grouped['goals_for'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_goals_against_{window}m'] = grouped['goals_against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'avg_xG_{window}m'] = grouped['xG'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        return df
    def _merge_features(self, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        home_stats = team_stats_df[team_stats_df['was_home'] == 1]
        away_stats = team_stats_df[team_stats_df['was_home'] == 0]
        stats_cols = [col for col in team_stats_df.columns if 'avg_' in col]
        home_stats_renamed = home_stats[['match_id', 'team'] + stats_cols].rename(columns={col: f'home_{col}' for col in stats_cols})
        away_stats_renamed = away_stats[['match_id', 'team'] + stats_cols].rename(columns={col: f'away_{col}' for col in stats_cols})
        df = self.matches_df.merge(home_stats_renamed, left_on=['match_id', 'home_team'], right_on=['match_id', 'team'], how='left')
        df = df.merge(away_stats_renamed, left_on=['match_id', 'away_team'], right_on=['match_id', 'team'], how='left')
        df.drop(columns=['team_x', 'team_y'], inplace=True, errors='ignore')
        return df

# --- Model Training and Prediction classes/functions ---
class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
    def train(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        lgbm = lgb.LGBMClassifier(random_state=42)
        self.model = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=3)
        self.model.fit(X_train_scaled, y_train)
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

# --- Betting Engine class ---
class BettingEngine:
    def __init__(self, kelly_fraction: float = 0.25, max_stake_pct: float = 0.02):
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
    def find_value_bets(self, potential_bets: pd.DataFrame, value_margin: float = 0.03) -> pd.DataFrame:
        potential_bets['market_prob'] = 1 / potential_bets['odds']
        value_bets = potential_bets[potential_bets['model_prob'] > (potential_bets['market_prob'] + value_margin)].copy()
        value_bets['edge'] = value_bets['model_prob'] - value_bets['market_prob']
        return value_bets
    def calculate_stake(self, odds: float, model_prob: float, bankroll: float) -> float:
        if (model_prob * odds - 1) <= 0: return 0.0
        kelly_percentage = (model_prob * odds - 1) / (odds - 1)
        stake_pct = min(self.max_stake_pct, kelly_percentage * self.kelly_fraction)
        return bankroll * stake_pct

# --- Backtesting Functions ---

def simulate_bankroll(initial_bankroll=100.0, training_split_date='2023-04-01'):
    logging.info("--- Starting Backtest Simulation ---")
    ingestor = DemoDataIngestor()
    matches_df = ingestor.get_matches_data(num_matches=150)
    odds_df = ingestor.get_odds_data(matches_df['id'].tolist())
    cleaner = DemoDataCleaner()
    matches_df = cleaner.clean_matches_data(matches_df)
    odds_df = cleaner.clean_odds_data(odds_df)
    daily_odds = odds_df[odds_df['bookmaker'] == 'BookieA'].rename(columns={'cote': 'odds'})
    full_data = pd.merge(matches_df, daily_odds[['match_id', 'odds']], on='match_id')
    feature_engineer = FeatureEngineer(full_data)
    enriched_data = feature_engineer.generate_features()
    train_df = enriched_data[enriched_data['match_date'] < pd.to_datetime(training_split_date, utc=True)]
    backtest_df = enriched_data[enriched_data['match_date'] >= pd.to_datetime(training_split_date, utc=True)]
    logging.info(f"Training model on {len(train_df)} matches (before {training_split_date}).")
    logging.info(f"Backtesting on {len(backtest_df)} matches (from {training_split_date}).")
    model_trainer = ModelTrainer()
    X_train = train_df.drop(columns=['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'odds'])
    X_train.columns = [str(col) for col in X_train.columns]
    y_train = (train_df['home_score'] + train_df['away_score'] > 2.5).astype(int)
    model_trainer.train(X_train, y_train)
    bankroll = initial_bankroll
    bankroll_history = [(backtest_df['match_date'].min() - pd.Timedelta(days=1), initial_bankroll)]
    bets_history = []
    betting_engine = BettingEngine()
    for day in sorted(backtest_df['match_date'].unique()):
        daily_matches = backtest_df[backtest_df['match_date'] == day].copy()
        if daily_matches.empty: continue
        X_today = daily_matches.drop(columns=['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'odds'])
        X_today.columns = [str(col) for col in X_today.columns]
        daily_matches['model_prob'] = model_trainer.predict(X_today)
        value_bets = betting_engine.find_value_bets(daily_matches)
        if value_bets.empty:
            bankroll_history.append((day, bankroll))
            continue
        for _, bet in value_bets.iterrows():
            stake = betting_engine.calculate_stake(bet['odds'], bet['model_prob'], bankroll)
            if stake > 0 and bankroll - stake >= 0:
                bankroll -= stake
                true_outcome = (bet['home_score'] + bet['away_score']) > 2.5
                payout = (stake * bet['odds']) if true_outcome else 0
                profit = payout - stake
                bankroll += payout
                bets_history.append({'date': day, 'match_id': bet['match_id'], 'stake': stake, 'odds': bet['odds'], 'model_prob': bet['model_prob'], 'result': 'win' if true_outcome else 'loss', 'profit': profit, 'bankroll_after_bet': bankroll})
        bankroll_history.append((day, bankroll))
    logging.info("--- Backtest Simulation Finished ---")
    return pd.DataFrame(bets_history), pd.DataFrame(bankroll_history, columns=['date', 'bankroll'])

def calculate_metrics(bets_history: pd.DataFrame, bankroll_history: pd.DataFrame, initial_bankroll: float):
    if bets_history.empty:
        logging.warning("No bets were placed, cannot calculate metrics.")
        return
    logging.info("--- Backtest Performance Metrics ---")
    final_bankroll = bankroll_history['bankroll'].iloc[-1]
    total_profit = final_bankroll - initial_bankroll
    total_wagered = bets_history['stake'].sum()
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    peaks = bankroll_history['bankroll'].cummax()
    drawdowns = (bankroll_history['bankroll'] - peaks) / peaks
    max_drawdown = drawdowns.min() * 100 if not drawdowns.empty else 0
    print(f"  Ending Bankroll: ${final_bankroll:.2f}")
    print(f"  Total Profit:    ${total_profit:.2f}")
    print(f"  Total Wagered:   ${total_wagered:.2f}")
    print(f"  ROI / Yield:     {roi:.2f}%")
    print(f"  Max Drawdown:    {max_drawdown:.2f}%")
    print(f"  Total Bets:      {len(bets_history)}")

def plot_results(bankroll_history: pd.DataFrame, bets_history: pd.DataFrame):
    if bankroll_history.empty:
        logging.warning("No bankroll history to plot.")
        return
    logging.info("--- Generating Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_history['date'], bankroll_history['bankroll'], marker='.', linestyle='-')
    plt.title('Bankroll Evolution Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Bankroll ($)', fontsize=12)
    plt.tight_layout()
    bankroll_plot_path = 'bankroll_evolution.png'
    plt.savefig(bankroll_plot_path)
    plt.close()
    logging.info(f"Bankroll evolution plot saved to {bankroll_plot_path}")
    if not bets_history.empty:
        bets_history['roi_pct'] = (bets_history['profit'] / bets_history['stake']) * 100
        plt.figure(figsize=(10, 6))
        plt.hist(bets_history['roi_pct'], bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of ROI per Bet', fontsize=16)
        plt.xlabel('ROI (%)', fontsize=12)
        plt.ylabel('Number of Bets', fontsize=12)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.tight_layout()
        roi_plot_path = 'roi_distribution.png'
        plt.savefig(roi_plot_path)
        plt.close()
        logging.info(f"ROI distribution plot saved to {roi_plot_path}")

def main_backtest():
    """
    Main function to orchestrate the backtesting simulation and reporting.
    """
    initial_bankroll = 100.0

    # 1. Run the simulation
    bets_history, bankroll_history = simulate_bankroll(initial_bankroll=initial_bankroll)

    if bets_history.empty:
        logging.warning("Backtest completed, but no bets were placed. No report to generate.")
        return

    # 2. Calculate and display metrics
    calculate_metrics(bets_history, bankroll_history, initial_bankroll)

    # 3. Generate and save plots
    plot_results(bankroll_history, bets_history)

    # 4. Save detailed bet history to CSV
    results_path = 'backtest_results.csv'
    bets_history.to_csv(results_path, index=False)
    logging.info(f"Detailed backtest results saved to {results_path}")

if __name__ == "__main__":
    main_backtest()
