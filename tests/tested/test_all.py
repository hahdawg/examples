# pylint: disable=W0613
from unittest.mock import patch
import pytest

import examples.tested.lib as tl
import examples.tested.main as tm


def test_car():
    model = "foo"
    price = 10
    car = tl.Car(model, price)
    assert car.model == model
    assert car.price == price
    car.save()


@patch("examples.tested.lib.add_slow")
def test_add_slow_and_multiply(mock_add_slow):
    x = 2
    y = 3
    mock_add_slow.return_value = x + y  # set return value here
    c = 4
    assert tl.add_slow_and_multiply(x, y, c) == c*mock_add_slow.return_value


@patch("examples.tested.main.tl.add_slow")
@patch.object(tl.Car, "save", return_value=None)
def test_main_runs(mock_add_slow, mock_car_save):
    tm.main()


def add(x, y):
    return x + y


def test_add_slow_and_multiply_monkey(monkeypatch):
    x = 2
    y = 3
    monkeypatch.setattr(tl, "add_slow", add)
    c = 4
    assert tl.add_slow_and_multiply(x, y, c) == c*(x + y)


def test_main_runs_monkey(monkeypatch):
    # We patch add_slow in tm.tl instead of tl
    monkeypatch.setattr(tm.tl, "add_slow", add)

    # Mock method
    def car_save(self: tm.tl.Car):
        return None

    # We patch tm.tl.Car instead of tl.Car
    monkeypatch.setattr(tm.tl.Car, "save", car_save)
    tm.main()
