"""Tests for tools/currency.py.

Most tests MOCK the API (fake replies): fast, offline, same result every time.
One test is marked 'live' and calls the real API.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.currency import get_exchange_rate


def fake_response(status=200, json_data=None):
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} error", response=resp)
    return resp


OK_REPLY = {"amount": 1.0, "base": "USD", "date": "2026-11-19", "rates": {"JPY": 159.97}}


@patch("tools.currency.requests.get")
def test_success_shape(mock_get):
    mock_get.return_value = fake_response(200, OK_REPLY)
    result = get_exchange_rate("USD", "JPY")
    assert result["from"] == "USD" and result["to"] == "JPY"
    assert result["rate"] == 159.97
    assert result["date"] == "2026-11-19"
    assert result["source"] == "Frankfurter (ECB)"
    assert "as_of" in result
    # Rule: every API call uses a timeout
    assert "timeout" in mock_get.call_args.kwargs


@patch("tools.currency.requests.get")
def test_lowercase_and_spaces_are_cleaned(mock_get):
    mock_get.return_value = fake_response(200, OK_REPLY)
    result = get_exchange_rate(" usd ", "jpy")
    assert result["from"] == "USD"
    assert mock_get.call_args.kwargs["params"] == {"from": "USD", "to": "JPY"}


@patch("tools.currency.requests.get")
def test_same_currency_needs_no_api(mock_get):
    result = get_exchange_rate("usd", "USD")
    assert result["rate"] == 1.0
    mock_get.assert_not_called()


@pytest.mark.parametrize("bad", ["US", "DOLLARS", "12A", "", None])
@patch("tools.currency.requests.get")
def test_invalid_code_needs_no_api(mock_get, bad):
    result = get_exchange_rate(bad, "JPY")
    assert "error" in result and "valid currency code" in result["error"]
    mock_get.assert_not_called()


@patch("tools.currency.requests.get")
def test_unknown_code_404(mock_get):
    mock_get.return_value = fake_response(404, {"message": "not found"})
    result = get_exchange_rate("XYZ", "EUR")
    assert "Unknown currency code" in result["error"]


@patch("tools.currency.requests.get", side_effect=requests.Timeout)
def test_timeout(mock_get):
    assert "too long" in get_exchange_rate("USD", "JPY")["error"]


@patch("tools.currency.requests.get", side_effect=requests.ConnectionError)
def test_no_connection(mock_get):
    assert "Could not reach" in get_exchange_rate("USD", "JPY")["error"]


@patch("tools.currency.requests.get")
def test_server_error_500(mock_get):
    mock_get.return_value = fake_response(500)
    assert "returned an error" in get_exchange_rate("USD", "JPY")["error"]


@patch("tools.currency.requests.get")
def test_weird_reply_shape(mock_get):
    mock_get.return_value = fake_response(200, {"unexpected": True})
    assert "unexpected answer" in get_exchange_rate("USD", "JPY")["error"]


@patch("tools.currency.requests.get")
def test_amount_is_converted_by_code(mock_get):
    mock_get.return_value = fake_response(200, OK_REPLY)
    result = get_exchange_rate("USD", "JPY", amount="220")  # LLMs may send a string
    assert result["amount"] == 220.0
    assert result["converted"] == round(220 * 159.97, 2)


@patch("tools.currency.requests.get")
def test_no_amount_means_no_converted_field(mock_get):
    mock_get.return_value = fake_response(200, OK_REPLY)
    assert "converted" not in get_exchange_rate("USD", "JPY")


@pytest.mark.parametrize("bad", [-5, 0, "lots"])
@patch("tools.currency.requests.get")
def test_bad_amount_needs_no_api(mock_get, bad):
    assert "positive number" in get_exchange_rate("USD", "JPY", amount=bad)["error"]
    mock_get.assert_not_called()


@pytest.mark.live
def test_live_usd_to_jpy():
    """Real API call. Skip offline with: pytest -m "not live" """
    result = get_exchange_rate("USD", "JPY")
    assert "error" not in result, result
    assert result["rate"] > 0
