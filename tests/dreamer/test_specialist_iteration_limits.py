import pytest
from pydantic import ValidationError

from src.config import DreamSettings, settings
from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist


def test_defaults_match_previous_hardcoded_limits():
    assert DeductionSpecialist().get_max_iterations() == 12
    assert InductionSpecialist().get_max_iterations() == 10


def test_specialists_read_their_own_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "DEDUCTION_MAX_TOOL_ITERATIONS", 20)
    monkeypatch.setattr(settings.DREAM, "INDUCTION_MAX_TOOL_ITERATIONS", 7)
    # The card-refresh limit must not leak into deduction or induction.
    monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 3)

    assert DeductionSpecialist().get_max_iterations() == 20
    assert InductionSpecialist().get_max_iterations() == 7


@pytest.mark.parametrize(
    "field", ["DEDUCTION_MAX_TOOL_ITERATIONS", "INDUCTION_MAX_TOOL_ITERATIONS"]
)
@pytest.mark.parametrize("value", [0, 51])
def test_limits_are_validated(field: str, value: int):
    with pytest.raises(ValidationError):
        DreamSettings.model_validate({field: value})
