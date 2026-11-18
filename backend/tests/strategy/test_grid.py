import pytest

from app.strategy.grid import expand_grid, structure_key
from tests.strategy.helpers import rsi_strategy


def test_expand_grid_counts():
    s = rsi_strategy(
        length={"min": 7, "max": 21, "step": 7}, buy_level={"min": 25, "max": 35, "step": 5}
    )
    grid = expand_grid(s)
    # length appears in both buy and sell operands but is one range each -> 3 ranges? No:
    # each RangeSpec occurrence is its own dimension.
    assert len(grid) == 3 * 3 * 3
    lengths = {g.buy.items[0].left.params["length"] for g in grid}
    assert lengths == {7.0, 14.0, 21.0}


def test_expand_grid_two_ranges_gives_9():
    s = rsi_strategy(
        buy_level={"min": 25, "max": 35, "step": 5}, sell_level={"min": 65, "max": 75, "step": 5}
    )
    assert len(expand_grid(s)) == 9


def test_expand_grid_no_ranges_returns_same():
    s = rsi_strategy()
    assert expand_grid(s) == [s]


def test_expand_grid_rejects_4_ranges():
    s = rsi_strategy(
        length={"min": 7, "max": 21, "step": 7},
        buy_level={"min": 25, "max": 35, "step": 5},
        sell_level={"min": 65, "max": 75, "step": 5},
        stop_loss={"kind": "percent", "value": {"min": 1, "max": 3, "step": 1}},
    )
    with pytest.raises(ValueError, match="3"):
        expand_grid(s)


def test_expand_grid_rejects_over_200():
    s = rsi_strategy(
        buy_level={"min": 1, "max": 30, "step": 1}, sell_level={"min": 50, "max": 99, "step": 1}
    )
    with pytest.raises(ValueError, match="200"):
        expand_grid(s)


def test_structure_key_ignores_params_but_not_indicators():
    a = rsi_strategy(buy_level=30, length=14)
    b = rsi_strategy(buy_level=25, length=10)
    c = rsi_strategy(buy_level={"min": 20, "max": 30, "step": 5})
    assert structure_key(a) == structure_key(b) == structure_key(c)
    d = a.model_copy(deep=True)
    d.buy.items[0].left.id = "mfi"
    d.buy.items[0].left.output = "mfi"
    assert structure_key(d) != structure_key(a)
