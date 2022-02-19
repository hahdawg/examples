# pylint: disable=W0613
from unittest.mock import patch

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
    mock_add_slow.return_value = x + y
    c = 4
    assert tl.add_slow_and_multiply(x, y, c) == c*mock_add_slow.return_value


@patch("examples.tested.main.tl.add_slow")
@patch.object(tl.Car, "save", return_value=None)
def test_main_runs(mock_add_slow, mock_car_save):
    tm.main()
