import numpy as np
import pandas as pd
from itertools import combinations
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
import os
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Security & User Models ---

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# In a real application, this would be a real database.
fake_users_db = {}

# --- Security Configuration ---
SECRET_KEY = "a_very_secret_key_that_should_be_in_an_env_file"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Security Utilities ---

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- FastAPI App Initialization ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Sports Betting Prediction API",
    description="API to serve daily betting parlays and historical results.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Business Logic (Copied from betting_engine.py for self-containment) ---

def predict_match_proba(match_features: dict) -> float:
    avg_goals = match_features.get('avg_goals', 2.5)
    prob = 0.4 + (avg_goals - 2.0) * 0.1
    return np.clip(prob, 0.05, 0.95)

class BettingEngine:
    def __init__(self, bankroll: float, kelly_fraction: float = 0.25, max_stake_pct: float = 0.02):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct

    def _find_value_bets(self, potential_bets: list, value_margin: float = 0.03) -> list:
        value_bets = []
        for bet in potential_bets:
            market_prob = 1 / bet['odds']
            model_prob = bet['model_prob']
            if model_prob > (market_prob + value_margin):
                bet['edge'] = model_prob - market_prob
                value_bets.append(bet)
        return value_bets

    def calculate_stake(self, odds: float, model_prob: float) -> float:
        if (model_prob * odds - 1) <= 0: return 0.0
        kelly_percentage = (model_prob * odds - 1) / (odds - 1)
        stake_pct = min(self.max_stake_pct, kelly_percentage * self.kelly_fraction)
        return self.bankroll * stake_pct

    def construct_daily_parlays(self, potential_singles: list, num_parlays: int = 2) -> list:
        value_singles = self._find_value_bets(potential_singles)
        if not value_singles: return []
        possible_parlays = []
        for size in [2, 3]:
            if len(value_singles) >= size:
                possible_parlays.extend(list(combinations(value_singles, size)))
        valid_parlays = []
        for parlay_tuple in possible_parlays:
            parlay = list(parlay_tuple)
            total_odds = np.prod([bet['odds'] for bet in parlay])
            if total_odds > 2.5: continue
            competitions = [bet['competition'] for bet in parlay]
            if len(set(competitions)) != len(competitions): continue
            total_prob = np.prod([bet['model_prob'] for bet in parlay])
            valid_parlays.append({'matches': [bet['name'] for bet in parlay], 'total_odds': total_odds, 'total_prob': total_prob, 'parlay_edge': total_prob - (1 / total_odds)})
        if not valid_parlays: return []
        ranked_parlays = sorted(valid_parlays, key=lambda x: x['parlay_edge'], reverse=True)
        top_parlays = ranked_parlays[:num_parlays]
        final_bets = []
        for parlay in top_parlays:
            parlay['stake'] = self.calculate_stake(parlay['total_odds'], parlay['total_prob'])
            final_bets.append(parlay)
        return final_bets

# --- API Endpoints ---

@app.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(fake_users_db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=User)
@limiter.limit("5/minute")
async def register_user(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username in fake_users_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    hashed_password = get_password_hash(form_data.password)
    user_in_db = UserInDB(username=form_data.username, hashed_password=hashed_password)
    fake_users_db[form_data.username] = user_in_db.dict()
    return User(username=form_data.username)

def send_notification(parlays: list):
    logging.info(f"--- NOTIFICATION: {len(parlays)} new parlays recommended. ---")
    pass

@app.get("/get_today_bets", summary="Get Today's Recommended Bets")
@limiter.limit("20/minute")
def get_today_bets(request: Request, current_user: User = Depends(get_current_user)):
    logging.info("Endpoint /get_today_bets called.")
    potential_singles = [
        {'name': 'PSG vs Lyon', 'competition': 'Ligue 1', 'odds': 1.6, 'features': {'avg_goals': 3.2}},
        {'name': 'Man City vs Chelsea', 'competition': 'Premier League', 'odds': 1.8, 'features': {'avg_goals': 4.5}},
        {'name': 'Real Madrid vs Barcelona', 'competition': 'La Liga', 'odds': 1.75, 'features': {'avg_goals': 3.1}},
        {'name': 'Bayern vs Dortmund', 'competition': 'Bundesliga', 'odds': 1.5, 'features': {'avg_goals': 5.0}},
        {'name': 'Inter vs AC Milan', 'competition': 'Serie A', 'odds': 2.2, 'features': {'avg_goals': 2.1}},
        {'name': 'Ajax vs PSV', 'competition': 'Eredivisie', 'odds': 1.65, 'features': {'avg_goals': 4.8}},
    ]
    for bet in potential_singles:
        bet['model_prob'] = predict_match_proba(bet['features'])
    engine = BettingEngine(bankroll=1000.0)
    recommended_parlays = engine.construct_daily_parlays(potential_singles, num_parlays=2)
    if recommended_parlays:
        send_notification(recommended_parlays)
    return {"bets": recommended_parlays}

@app.get("/get_history", summary="Get Historical Bet Results")
@limiter.limit("20/minute")
def get_history(request: Request, current_user: User = Depends(get_current_user)):
    logging.info("Endpoint /get_history called.")
    results_path = 'backtest_results.csv'
    if not os.path.exists(results_path):
        logging.warning(f"History file not found at {results_path}.")
        return {"history": []}
    try:
        history_df = pd.read_csv(results_path)
        history = history_df.to_dict(orient='records')
        return {"history": history}
    except Exception as e:
        logging.error(f"Error reading or processing history file: {e}")
        return {"history": [], "error": str(e)}

@app.post("/user/disable", summary="Disable user account (Self-Exclusion)")
@limiter.limit("5/minute")
async def disable_user(request: Request, current_user: User = Depends(get_current_user)):
    logging.info(f"Disabling account for user: {current_user.username}")
    fake_users_db[current_user.username]['disabled'] = True
    return {"message": f"User {current_user.username} has been disabled."}

@app.delete("/user/delete", summary="Delete user account (GDPR Right to be Forgotten)")
@limiter.limit("5/minute")
async def delete_user(request: Request, current_user: User = Depends(get_current_user)):
    logging.info(f"Deleting account for user: {current_user.username}")
    if current_user.username in fake_users_db:
        del fake_users_db[current_user.username]
    return {"message": f"User {current_user.username} has been deleted."}

@app.get("/", summary="Root endpoint with legal warning")
@limiter.limit("20/minute")
def read_root(request: Request):
    return {
        "message": "Welcome to the Sports Betting API.",
        "legal_warning": "Les paris comportent un risque, ne jouez pas au-delà de vos moyens. Gambling involves risks. Do not play beyond your means."
    }
