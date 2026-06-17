# 2026 World Cup Predictor

A statistical model that predicts match outcomes and simulates the 2026 FIFA World Cup using Elo ratings, Poisson regression, and Monte Carlo simulation.

🔗 **Live app:** _add your Streamlit Cloud link here once deployed_

## Overview

This project builds a full prediction pipeline from raw historical match data to a live, interactive web app:

1. **Elo rating system** — custom implementation based on the official [eloratings.net](https://www.eloratings.net/about) formula, with tournament-tier K-values, a goal-difference multiplier, and manual elite-team boosts to correct for confederation strength imbalances.
2. **Poisson regression model** — two GLMs (home goals, away goals) trained on Elo ratings, FIFA rank difference, neutral venue, and tournament importance.
3. **Dixon-Coles correction** — adjusts low-scoring outcome probabilities (0-0, 1-0, 0-1, 1-1) to better match real-world draw frequency.
4. **Monte Carlo simulation** — 10,000+ simulated tournaments to estimate group stage, knockout, and championship probabilities for all 48 teams.
5. **Streamlit app** — interactive dashboard with champion odds, head-to-head predictor, group stage viewer, bracket simulator, team profiles, and a live tracker that updates predictions as real results come in.

## Features

- **🏆 Champion Odds** — probability of each team reaching every stage, from Round of 32 to champion
- **⚔️ Head-to-Head** — pick any two teams and see win/draw/loss probability, expected goals, and most likely scorelines
- **📊 Group Stage** — qualification odds for all 16 groups, plus a standalone group simulator
- **🏟️ Bracket Simulator** — auto-simulate the full tournament or manually pick winners round by round
- **👤 Team Profile** — Elo rating, FIFA rank, group, and stage-by-stage probabilities for any team
- **🔴 Live Tracker** — paste completed match results to get updated odds as the tournament progresses, without overwriting the original pre-tournament prediction

## Methodology

### Elo Ratings

Ratings update after every match using:

```
R_new = R_old + K × (W - We)
```

where `We` is the win expectation from the logistic Elo formula, and `K` is scaled by tournament tier (20 for friendlies up to 60 for World Cup finals) and a goal-difference multiplier for margin of victory.

### Poisson Goal Model

```
log(home_xG) = β₀ + β₁·home_elo + β₂·away_elo + β₃·elo_diff + β₄·neutral + β₅·tournament_weight + β₆·fifa_rank_diff
```

Fit via maximum likelihood as a generalized linear model with a Poisson family, separately for home and away goals.

### Dixon-Coles Correction

Applies a correction factor (ρ = -0.13) to low-scoring outcomes to fix the standard Poisson model's tendency to underestimate draws.

### Simulation

Each Monte Carlo run simulates the full group stage (with real tiebreaker rules — points, goal difference, goals scored) followed by knockout rounds (extra time and penalties when level), repeated 10,000 times to produce stable stage-advancement probabilities.

## Project Structure

```
├── app.py                      # Streamlit app
├── requirements.txt            # Python dependencies
├── elo_all_teams_v2.csv        # Elo ratings for all rated teams
├── group_stages_clean.csv      # 2026 WC group assignments
├── fixtures_clean.csv          # 2026 WC group stage fixtures
├── fifa_rankings.csv           # FIFA rankings reference
├── simulation_results_v2.csv   # Pre-tournament Monte Carlo results
├── home_model_v2.pkl           # Trained Poisson home-goals model
├── away_model_v2.pkl           # Trained Poisson away-goals model
└── notebooks/                  # Full pipeline (data cleaning → Elo → features → training → backtesting → simulation)
```

## Running Locally

```bash
git clone https://github.com/<your-username>/world-cup-2026-predictor.git
cd world-cup-2026-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Backtesting

The model was backtested against the actual 2018 and 2022 World Cups by rebuilding Elo ratings and retraining the Poisson model using only pre-tournament data, then scoring predictions against real results:

| Metric | 2018 | 2022 |
|---|---|---|
| Accuracy | ~51.6% | ~51.6% |
| Brier Score | ~0.595 | ~0.605 |
| Log Loss | ~0.998 | ~1.026 |

(Random baseline: 33.3% accuracy, 0.667 Brier score)

## Disclaimer

This is a statistical model for educational and entertainment purposes. Football is inherently unpredictable — upsets, injuries, and form on the day matter more than any model can fully capture.

## License

MIT
