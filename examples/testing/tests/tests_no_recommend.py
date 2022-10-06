"""
Discouraged tests for cfg, lib, and main code. Normally the tests would live in different modules.
"""
from unittest.mock import (
    patch,
    MagicMock
)

import pytest

import examples.testing.main as m


def test_query_db_magicmock():
    """
    Use MagicMock to mock query_db.

    NOTE: MagicMock can't do autospec, so probably don't use it.
    """
    mock_query_db = MagicMock(spec_set=m.query_db)
    mock_query_db("foo")
    mock_query_db.assert_called_with("foo")
    mock_query_db.assert_called_once()

    # NOTE: MagicMock doesn't use the function signature so the line below
    # won't raise an error.
    mock_query_db("foo", "bar")


@patch("examples.testing.main.query_db")
def test_query_db_patch_without_autospec(mock_query_db):
    """
    Use patch to mock query_db.

    NOTE: If we don't pass autospec=True, the mock function can have a different
    function signature than the original function. So pass autospec=True.
    """
    mock_query_db("foo", "bar")
    mock_query_db.assert_called_with("foo", "bar")
    mock_query_db.assert_called_once()


@patch("examples.testing.main.query_db", autospec=True)
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
