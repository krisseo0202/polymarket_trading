"""Tests for fetch_market_odds() in src/utils/market_utils.py.

All HTTP calls are patched via unittest.mock — no live network needed.
"""

import json
import pytest
import requests

from unittest.mock import MagicMock, patch, call

from src.utils.market_utils import fetch_market_odds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    """Build a fake requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


def _gamma_resp(yes_id: str = "YES_TOKEN", no_id: str = "NO_TOKEN") -> MagicMock:
    """Gamma API response with two valid token IDs."""
    return _mock_response([{"clobTokenIds": json.dumps([yes_id, no_id])}])


def _book_resp(bid: float, ask: float) -> MagicMock:
    """CLOB /book response with a single bid and ask level."""
    return _mock_response({
        "bids": [{"price": str(bid), "size": "100"}],
        "asks": [{"price": str(ask), "size": "100"}],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestValidResponse:
    @patch("src.utils.market_utils.requests.get")
    def test_returns_correct_mid_prices(self, mock_get):
        # Gamma → YES book (bid=0.60, ask=0.62) → NO book (bid=0.38, ask=0.40)
        mock_get.side_effect = [
            _gamma_resp(),
            _book_resp(bid=0.60, ask=0.62),   # YES mid = 0.61
            _book_resp(bid=0.38, ask=0.40),    # NO  mid = 0.39
        ]
        up, down = fetch_market_odds("12345", backoff_base=0.0)

        assert up   == pytest.approx(0.61)
        assert down == pytest.approx(0.39)

    @patch("src.utils.market_utils.requests.get")
    def test_returns_tuple_of_two_floats(self, mock_get):
        mock_get.side_effect = [
            _gamma_resp(),
            _book_resp(bid=0.55, ask=0.57),
            _book_resp(bid=0.43, ask=0.45),
        ]
        result = fetch_market_odds("12345", backoff_base=0.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        up, down = result
        assert isinstance(up, float)
        assert isinstance(down, float)

    @patch("src.utils.market_utils.requests.get")
    def test_calls_gamma_then_two_clob_books(self, mock_get):
        mock_get.side_effect = [
            _gamma_resp(yes_id="YES_TID", no_id="NO_TID"),
            _book_resp(bid=0.55, ask=0.57),
            _book_resp(bid=0.43, ask=0.45),
        ]
        fetch_market_odds("42", backoff_base=0.0)

        assert mock_get.call_count == 3
        # second and third calls should use yes/no token IDs
        _, yes_call, no_call = mock_get.call_args_list
        assert yes_call.kwargs["params"]["token_id"] == "YES_TID"
        assert no_call.kwargs["params"]["token_id"] == "NO_TID"


class TestMarketNotFound:
    @patch("src.utils.market_utils.requests.get")
    def test_raises_value_error_when_gamma_returns_empty(self, mock_get):
        mock_get.return_value = _mock_response([])  # empty list

        with pytest.raises(ValueError, match="not found"):
            fetch_market_odds("99999", backoff_base=0.0)

    @patch("src.utils.market_utils.requests.get")
    def test_raises_value_error_when_gamma_returns_none(self, mock_get):
        mock_get.return_value = _mock_response(None)

        with pytest.raises(ValueError, match="not found"):
            fetch_market_odds("99999", backoff_base=0.0)


class TestMissingTokenIds:
    @patch("src.utils.market_utils.requests.get")
    def test_raises_value_error_when_only_one_token(self, mock_get):
        mock_get.return_value = _mock_response([
            {"clobTokenIds": json.dumps(["ONLY_ONE_TOKEN"])}
        ])

        with pytest.raises(ValueError, match="expected 2"):
            fetch_market_odds("12345", backoff_base=0.0)

    @patch("src.utils.market_utils.requests.get")
    def test_raises_value_error_when_no_tokens(self, mock_get):
        mock_get.return_value = _mock_response([
            {"clobTokenIds": "[]"}
        ])

        with pytest.raises(ValueError, match="expected 2"):
            fetch_market_odds("12345", backoff_base=0.0)


class TestTimeoutRetry:
    @patch("src.utils.market_utils.time.sleep")
    @patch("src.utils.market_utils.requests.get")
    def test_retries_and_succeeds_after_one_timeout(self, mock_get, mock_sleep):
        # First Gamma call times out; second succeeds
        timeout_exc = requests.exceptions.Timeout()
        mock_get.side_effect = [
            timeout_exc,
            _gamma_resp(),
            _book_resp(bid=0.55, ask=0.57),
            _book_resp(bid=0.43, ask=0.45),
        ]
        up, down = fetch_market_odds("12345", retries=3, backoff_base=0.01)

        assert up   == pytest.approx(0.56)
        assert down == pytest.approx(0.44)
        assert mock_get.call_count == 4  # 1 fail + 3 successful calls

    @patch("src.utils.market_utils.time.sleep")
    @patch("src.utils.market_utils.requests.get")
    def test_raises_runtime_error_after_all_timeouts_exhausted(self, mock_get, mock_sleep):
        timeout_exc = requests.exceptions.Timeout()
        mock_get.side_effect = [timeout_exc, timeout_exc, timeout_exc]  # 3 retries

        with pytest.raises(RuntimeError, match="All 3 attempts failed"):
            fetch_market_odds("12345", retries=3, backoff_base=0.0)


class TestHttp500Retry:
    @patch("src.utils.market_utils.time.sleep")
    @patch("src.utils.market_utils.requests.get")
    def test_retries_on_500_and_succeeds(self, mock_get, mock_sleep):
        server_error = _mock_response({}, status_code=500)
        mock_get.side_effect = [
            server_error,
            _gamma_resp(),
            _book_resp(bid=0.50, ask=0.52),
            _book_resp(bid=0.48, ask=0.50),
        ]
        up, down = fetch_market_odds("12345", retries=3, backoff_base=0.01)

        assert up   == pytest.approx(0.51)
        assert down == pytest.approx(0.49)


class TestHttp404NoRetry:
    @patch("src.utils.market_utils.time.sleep")
    @patch("src.utils.market_utils.requests.get")
    def test_raises_immediately_on_404_without_retrying(self, mock_get, mock_sleep):
        not_found = _mock_response({}, status_code=404)
        mock_get.return_value = not_found

        with pytest.raises(RuntimeError, match="HTTP 404"):
            fetch_market_odds("12345", retries=3, backoff_base=0.0)

        # Must not have retried — only one GET issued
        assert mock_get.call_count == 1
