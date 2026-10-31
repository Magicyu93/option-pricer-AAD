"""Tenors as nominal years, built on the one tenor parser."""

import pytest

from torch_pricer.conventions import tenor_label, tenor_years
from torch_pricer.errors import ValidationError


@pytest.mark.parametrize(
    "label,years", [("1M", 1 / 12), ("3M", 0.25), ("1Y", 1.0), ("30Y", 30.0), ("2W", 14 / 365)]
)
def test_tenor_years(label, years):
    assert tenor_years(label) == pytest.approx(years)


@pytest.mark.parametrize("label,years", [("1M", 1 / 12), ("3M", 0.25), ("1Y", 1.0), ("30Y", 30.0)])
def test_tenor_labels_round_trip(label, years):
    assert tenor_label(years) == label
    assert tenor_years(tenor_label(years)) == pytest.approx(years)


@pytest.mark.parametrize("bad", ["3 months", "M3", "3Q", ""])
def test_unparseable_tenor_is_rejected(bad):
    with pytest.raises(ValidationError):
        tenor_years(bad)


def test_fractional_months_have_no_label():
    with pytest.raises(ValidationError):
        tenor_label(0.3)
