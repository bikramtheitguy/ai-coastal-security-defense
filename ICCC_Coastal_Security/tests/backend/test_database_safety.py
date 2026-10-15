"""Check the destructive test fixture's guard without connecting to a database."""

import pytest

from conftest import validate_test_database


def test_disposable_local_database_is_allowed():
    validate_test_database("postgresql+psycopg://test:test@127.0.0.1:5432/iccc_test_ci", "1")


@pytest.mark.parametrize("url,opt_in", [
    ("postgresql+psycopg://test:test@127.0.0.1/iccc_test_ci", None),
    ("postgresql+psycopg://test:test@127.0.0.1/retained_records", "1"),
    ("postgresql+psycopg://test:test@remote.example/iccc_test_ci", "1"),
    ("postgresql+psycopg://test:test@localhost/iccc_test_ci?host=remote.example", "1"),
    ("sqlite:///retained.db", "1"),
])
def test_retained_remote_or_unapproved_target_is_rejected(url, opt_in):
    with pytest.raises(ValueError, match="Schema reset requires"):
        validate_test_database(url, opt_in)
