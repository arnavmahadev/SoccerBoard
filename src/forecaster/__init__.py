"""Competition forecaster: probabilistic scorelines (Dixon-Coles) + Monte Carlo
tournament simulation, sharing SoccerBoard's data layer, FastAPI backend and
frontend shell with the xG engine.

The scoreline model is competition-agnostic; what differs between competitions
is the *format* (how matches are scheduled and how teams advance), which lives
behind the pluggable `CompetitionFormat` interface in `forecaster.formats`.
"""
