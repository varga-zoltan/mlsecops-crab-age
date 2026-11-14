"""Acceptance gate: decides whether the challenger may replace the champion.

This is the "never get worse" part of the project. A new model only gets the
``champion`` alias if

1. its test MAE is not above an absolute limit (``train.gate.max_mae``), and
2. its test MAE is not worse than the current champion's MAE on the same test set.

If there is no champion yet, only the absolute limit applies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    """Result of the acceptance gate."""

    accepted: bool
    reason: str
    challenger_mae: float
    champion_mae: float | None


def acceptance_gate(challenger_mae: float, champion_mae: float | None, max_mae: float) -> GateDecision:
    """Compare the challenger with the absolute limit and the current champion."""
    if challenger_mae > max_mae:
        accepted = False
        reason = f"challenger MAE {challenger_mae:.4f} is above the limit {max_mae:.4f}"
    elif champion_mae is None:
        accepted = True
        reason = f"no champion yet and challenger MAE {challenger_mae:.4f} is within the limit {max_mae:.4f}"
    elif challenger_mae > champion_mae:
        accepted = False
        reason = f"challenger MAE {challenger_mae:.4f} is worse than champion MAE {champion_mae:.4f}"
    else:
        accepted = True
        reason = f"challenger MAE {challenger_mae:.4f} is not worse than champion MAE {champion_mae:.4f}"
    return GateDecision(accepted=accepted, reason=reason, challenger_mae=challenger_mae, champion_mae=champion_mae)
