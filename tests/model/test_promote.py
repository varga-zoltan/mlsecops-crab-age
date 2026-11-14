"""Tests for the acceptance gate."""

import pytest

from crab_age.model.promote import acceptance_gate


@pytest.mark.parametrize(
    ("challenger", "champion", "max_mae", "accepted", "reason_part"),
    [
        (1.5, None, 2.0, True, "no champion yet"),
        (2.0, None, 2.0, True, "no champion yet"),
        (2.1, None, 2.0, False, "above the limit"),
        (1.4, 1.5, 2.0, True, "not worse"),
        (1.5, 1.5, 2.0, True, "not worse"),
        (1.6, 1.5, 2.0, False, "worse than champion"),
        (2.5, 3.0, 2.0, False, "above the limit"),
    ],
)
def test_acceptance_gate(
    challenger: float, champion: float | None, max_mae: float, accepted: bool, reason_part: str
) -> None:
    decision = acceptance_gate(challenger, champion, max_mae)
    assert decision.accepted is accepted
    assert reason_part in decision.reason
    assert decision.challenger_mae == challenger
    assert decision.champion_mae == champion
