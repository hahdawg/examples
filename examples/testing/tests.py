import os
import pickle
from unittest.mock import (
    create_autospec,
    patch,
    MagicMock
)

import pytest

import examples.testing.module as m


def test_query_db_autospec():
    """
    Use autospec to mock query_db.
    """
    mock_query_db = create_autospec(m.query_db)
    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()
    with pytest.raises(TypeError):
        mock_query_db("foo", "bar")


def test_query_db_magicmock():
    """
    Use MagicMock to mock query_db.
    """
    mock_query_db = MagicMock(spec_set=m.query_db)
    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()

    # NOTE: MagicMock doesn't use the function signature so the line below
    # won't raise an error.
    mock_query_db("foo", "bar")


@patch("examples.testing.module.query_db")
def test_query_db_patch_without_autospec(mock_query_db):
    """
    Use patch to mock query_db.

    If we don't pass autospec=True, the mock function can have a different
    function signature than the original function.
    """
    mock_query_db("foo", "bar")
    mock_query_db.assert_called_with("foo", "bar")
    mock_query_db.assert_called_once()


@patch("examples.testing.module.query_db", autospec=True)
def test_query_db_patch_with_autospec(mock_query_db):
    """
    Use patch to mock query_db.

    Passing autospec=True makes the mock function can have the same
    function signature as the original function.
    """
    with pytest.raises(TypeError):
        mock_query_db("foo", "bar")

    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()


@patch.object(m, "query_db", autospec=True)
def test_query_db_patch_object(mock_query_db):
    with pytest.raises(TypeError):
        mock_query_db("foo", "bar")

    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()


def check_query_db_inputs(colname: str):
    """
    Use this as a side effect to validate that the mock function got called
    with a string.
    """
    assert isinstance(colname, str)


def test_query_db_inputs_using_side_effect():
    """
    Here, we add a check_query_db_inputs as a side effect to validate that the
    mock function got called with a string.

    NOTE: The side effect function gets passed the same parameters as the mock.
    """
    mock_query_db = create_autospec(m.query_db, side_effect=check_query_db_inputs)
    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()
    with pytest.raises(AssertionError):
        mock_query_db(1)


def test_query_db_context_manager():
    """
    Use context managers to patch
    """
    # mocked function only lives inside context
    with patch("examples.testing.module.query_db", autospec=True) as mock_query_db:
        mock_query_db("foo")
        mock_query_db.assert_called_with("foo")
        mock_query_db.assert_called_once()

    # here, we mock time.sleep but call the original function
    with patch("time.sleep", autospec=True) as mock_sleep:
        output = m.query_db("foo")
        mock_sleep.assert_called_once()  # check if mock was actually called
        assert isinstance(output, m.DbOutput)


def test_main_autospec(monkeypatch):
    """
    Test main where we replace query_db with a mock created from autospec.
    """
    mock_test_path = "/tmp/main_output.p"
    expected = ("a", "abc")
    mock_query_db = create_autospec(m.query_db, return_value=m.DbOutput("a", ["a", "b", "c"]))
    monkeypatch.setattr(m, "OUTPUT_PATH", mock_test_path)
    monkeypatch.setattr(m, "query_db", mock_query_db)
    m.main()
    mock_query_db.assert_called_once()
    mock_query_db.assert_called_with("foo")
    assert os.path.exists(mock_test_path)
    with open(mock_test_path, "rb") as f:
        output = pickle.load(f)
    assert output == expected


def test_main_patch():
    """
    Test main where we replace query_db with a patch.
    """
    mock_test_path = "/tmp/main_output.p"
    expected = ("a", "abc")
    with patch.object(
        m, "query_db", autospec=True, return_value=m.DbOutput("a", ["a", "b", "c"])
    ) as mock_query_db, patch.object(
        m, "OUTPUT_PATH", new=mock_test_path
    ):
        m.main()
        mock_query_db.assert_called_once()
        mock_query_db.assert_called_with("foo")
        assert os.path.exists(mock_test_path)
        with open(mock_test_path, "rb") as f:
            output = pickle.load(f)
        assert output == expected
