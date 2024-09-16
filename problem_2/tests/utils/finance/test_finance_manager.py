import pytest

import datetime
from utils.finance.finance_manager import TickerManager

class TestTickerManager:
    
    @pytest.mark.parametrize('illegal_interval', ['1.5m', '60m', '1.5w', '1.5d'])
    def test_init_throws_value_error_when_interval_not_accepted(self, illegal_interval):
        with pytest.raises(ValueError):
            _ = TickerManager(
                ['AAPL'],
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 31),
                interval=illegal_interval)