"""Pluggable competition-format layer.

The scoreline model (Dixon-Coles) is competition-agnostic. What differs between
competitions is the *format* — how matches are scheduled and how teams advance.
Each format implements the `CompetitionFormat` interface; a single generic
Monte Carlo driver simulates any of them. World Cup is fully implemented; league
and Champions League are documented stubs behind the same interface.
"""

from forecaster.formats.base import CompetitionFormat, MatchSampler, monte_carlo

__all__ = ["CompetitionFormat", "MatchSampler", "monte_carlo"]
