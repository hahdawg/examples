"""
Tests for cfg, lib, and main code. Normally the tests would live in different modules.
"""
import os
import pickle
import time
from unittest.mock import (
    create_autospec,
    patch,
)

import pytest

import examples.testing.config as cfg
import examples.testing.lib as lib
import examples.testing.main as m


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (lib.DbOutput("a", []), ("a", "")),
        (lib.DbOutput("a", ["a"]), ("a", "a")),
        (lib.DbOutput("a", ["a", "b"]), ("a", "ab")),
    ]
)
def test_process_data(test_input, expected):
    assert lib.process_data(test_input) == expected


def test_query_db_autospec():
    """
    Use autospec to mock query_db.

    NOTE: create_autospec makes the mock function have the same signature assert_called_with
    the mocked function.
    """
    mock_query_db = create_autospec(m.query_db)
    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()
    with pytest.raises(TypeError):
        mock_query_db("foo", "bar")


@patch.object(m, "query_db", autospec=True)
def test_query_db_patch_object(mock_query_db):
    """
    NOTE: With patch.object, we pass module as module and function as string. Note that
    patch.object requires less typing than patch if we've imported the module.

    This is the best approach.
    """
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
    with pytest.raises(AssertionError):  # side effect should raise AssertionError
        mock_query_db(1)


def test_query_db_context_manager():
    """
    Use context managers to patch

    NOTE: This is a bit easier than test_main_autospec because we don't have to use monkeypatch.
    """
    # mocked function only lives inside context
    with patch.object(m, "query_db", autospec=True) as mock_query_db:
        mock_query_db("foo")
        mock_query_db.assert_called_with("foo")
        mock_query_db.assert_called_once()

    # here, we mock time.sleep but call the original function
    with patch.object(time, "sleep", autospec=True) as mock_sleep:
        output = m.query_db("foo")
        mock_sleep.assert_called_once()
        assert isinstance(output, lib.DbOutput)


def test_main_autospec(monkeypatch):
    """
    Test main where we replace query_db with a mock created from autospec.

    NOTE: This is a bit more verbose than patch (see below) because we have to
    use monkeypatch to use the mocks.
    """
    mock_test_path = "/tmp/main_output.p"
    expected = ("a", "abc")
    mock_query_db = create_autospec(
        m.query_db, return_value=lib.DbOutput("a", ["a", "b", "c"])
    )
    monkeypatch.setattr(cfg, "output_path", mock_test_path)
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

    NOTE: This is the best approach.
    """
    mock_test_path = "/tmp/main_output.p"
    expected = ("a", "abc")
    with patch.object(
        m, "query_db", autospec=True, return_value=lib.DbOutput("a", ["a", "b", "c"])
    ) as mock_query_db, patch.object(
        cfg, "output_path", new=mock_test_path
    ):
        m.main()
        mock_query_db.assert_called_once()
        mock_query_db.assert_called_with("foo")
        assert os.path.exists(mock_test_path)
        with open(mock_test_path, "rb") as f:
            output = pickle.load(f)
        assert output == expected
