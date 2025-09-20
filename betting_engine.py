import numpy as np
import pandas as pd
from itertools import combinations
import logging

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_match_proba(match_features: dict) -> float:
    """
    Placeholder for the real model prediction function.
    In a real implementation, this would load the trained model and scaler
    from 'over_2_5_goals_model.joblib' and return a real prediction.

    For this demo, it returns a dummy probability based on a simple feature.
    """
    # This dummy logic allows us to simulate different probabilities for different matches.
    # In a real scenario, match_features would be a DataFrame row or similar structure.
    # We use a simple heuristic for the demo.
    avg_goals = match_features.get('avg_goals', 2.5) # Default to 2.5 if not provided
    # A simple formula to generate varied probabilities
    prob = 0.4 + (avg_goals - 2.0) * 0.1
    # Clip the probability to be within a realistic range [0.05, 0.95]
    return np.clip(prob, 0.05, 0.95)

# --- Betting Engine logic will be added below ---

def main_engine():
    """
    Main function to demonstrate the BettingEngine.
    """
    logging.info("--- Running Betting Engine Demonstration ---")

    # 1. Create a list of potential single bets for the day
    # In a real application, this data would come from the feature engineering step
    # and the model prediction step. We adjust the features to ensure some value bets are found.
    potential_singles = [
        {'name': 'PSG vs Lyon', 'competition': 'Ligue 1', 'odds': 1.6, 'features': {'avg_goals': 3.2}},
        {'name': 'Man City vs Chelsea', 'competition': 'Premier League', 'odds': 1.8, 'features': {'avg_goals': 4.5}}, # High avg_goals to create value
        {'name': 'Real Madrid vs Barcelona', 'competition': 'La Liga', 'odds': 1.75, 'features': {'avg_goals': 3.1}},
        {'name': 'Bayern vs Dortmund', 'competition': 'Bundesliga', 'odds': 1.5, 'features': {'avg_goals': 5.0}}, # High avg_goals to create value
        {'name': 'Inter vs AC Milan', 'competition': 'Serie A', 'odds': 2.2, 'features': {'avg_goals': 2.1}},
        {'name': 'Ajax vs PSV', 'competition': 'Eredivisie', 'odds': 1.65, 'features': {'avg_goals': 4.8}}, # High avg_goals to create value
        {'name': 'Avispa Fukuoka vs Kashima Antlers', 'competition': 'J-League', 'odds': 2.4, 'features': {'avg_goals': 1.9}},
    ]

    # Add model probability to each bet using the placeholder function
    for bet in potential_singles:
        bet['model_prob'] = predict_match_proba(bet['features'])

    # 2. Initialize the betting engine with a bankroll
    engine = BettingEngine(bankroll=1000.0)

    # 3. Construct the daily parlays
    recommended_parlays = engine.construct_daily_parlays(potential_singles, num_parlays=2)

    # 4. Print the results
    if not recommended_parlays:
        logging.info("No parlays to recommend today.")
    else:
        print("\n--- Daily Recommended Parlays ---")
        for i, parlay in enumerate(recommended_parlays, 1):
            print(f"\n--- Parlay #{i} ---")
            print(f"  Matches: {', '.join(parlay['matches'])}")
            print(f"  Total Odds: {parlay['total_odds']:.2f}")
            print(f"  Model Probability: {parlay['total_prob']:.2%}")
            print(f"  Recommended Stake: ${parlay['stake']:.2f}")

class BettingEngine:
    """
    Implements the logic for finding value bets and constructing parlays.
    """
    def __init__(self, bankroll: float, kelly_fraction: float = 0.25, max_stake_pct: float = 0.02):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        logging.info(f"BettingEngine initialized with bankroll: ${self.bankroll:.2f}")

    def _find_value_bets(self, potential_bets: list, value_margin: float = 0.03) -> list:
        """
        Filters a list of potential bets to find those with a positive value edge.

        Args:
            potential_bets (list): A list of dicts, each representing a single bet.
            value_margin (float): The required edge over the market's implied probability.

        Returns:
            A list of bets that are determined to have value.
        """
        value_bets = []
        for bet in potential_bets:
            market_prob = 1 / bet['odds']
            model_prob = bet['model_prob']

            if model_prob > (market_prob + value_margin):
                bet['edge'] = model_prob - market_prob
                value_bets.append(bet)
                logging.info(f"Value found for {bet['name']}: Model Prob={model_prob:.2%}, Market Prob={market_prob:.2%}, Edge={bet['edge']:.2%}")

        return value_bets

    def calculate_stake(self, odds: float, model_prob: float) -> float:
        """
        Calculates the stake for a bet using the fractional Kelly criterion.
        Returns 0 if there is no edge.
        """
        if (model_prob * odds - 1) <= 0:
            return 0.0

        kelly_percentage = (model_prob * odds - 1) / (odds - 1)

        # Apply fractional Kelly and cap at max stake percentage
        stake_pct = min(
            self.max_stake_pct,
            kelly_percentage * self.kelly_fraction
        )

        return self.bankroll * stake_pct

    def construct_daily_parlays(self, potential_singles: list, num_parlays: int = 2) -> list:
        """
        Constructs the best parlays (combinés) from a list of available single bets.
        """
        logging.info("--- Constructing Daily Parlays ---")

        # 1. Find all single bets that have value
        value_singles = self._find_value_bets(potential_singles)

        if not value_singles:
            logging.warning("No value bets found for today. No parlays will be constructed.")
            return []

        # 2. Generate all possible parlays of size 2 and 3
        possible_parlays = []
        for size in [2, 3]:
            if len(value_singles) >= size:
                possible_parlays.extend(list(combinations(value_singles, size)))

        # 3. Filter parlays based on rules
        valid_parlays = []
        for parlay_tuple in possible_parlays:
            parlay = list(parlay_tuple)

            # Rule: Total odds <= 2.5
            total_odds = np.prod([bet['odds'] for bet in parlay])
            if total_odds > 2.5:
                continue

            # Rule: Exclude correlated matches (simple check: same competition)
            competitions = [bet['competition'] for bet in parlay]
            if len(set(competitions)) != len(competitions):
                continue # Has duplicates, so correlated

            # Calculate total probability and value
            total_prob = np.prod([bet['model_prob'] for bet in parlay])

            valid_parlays.append({
                'matches': [bet['name'] for bet in parlay],
                'total_odds': total_odds,
                'total_prob': total_prob,
                'parlay_edge': total_prob - (1 / total_odds)
            })

        if not valid_parlays:
            logging.warning("No valid parlays could be constructed from the value bets.")
            return []

        # 4. Rank parlays (e.g., by highest edge) and select the top N
        ranked_parlays = sorted(valid_parlays, key=lambda x: x['parlay_edge'], reverse=True)
        top_parlays = ranked_parlays[:num_parlays]

        # 5. Calculate stake for the selected parlays
        final_bets = []
        for parlay in top_parlays:
            parlay['stake'] = self.calculate_stake(parlay['total_odds'], parlay['total_prob'])
            final_bets.append(parlay)

        logging.info(f"Successfully constructed {len(final_bets)} parlays.")
        return final_bets

if __name__ == "__main__":
    main_engine()
