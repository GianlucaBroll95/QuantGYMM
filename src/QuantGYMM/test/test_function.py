import pandas as pd
import pytest

from ..utils import is_target_holiday


@pytest.mark.parametrize("date, output", [(pd.to_datetime("2023-01-01"), True),
                                          (pd.to_datetime("2023-04-09"), True),
                                          (pd.to_datetime("2023-04-07"), True),
                                          (pd.to_datetime("2023-04-10"), True),
                                          (pd.to_datetime("2023-12-25"), True),
                                          (pd.to_datetime("2023-12-26"), True),
                                          (pd.to_datetime("2025-11-04"), False)])
def test_holiday(date, output):
    assert is_target_holiday(date) == output
