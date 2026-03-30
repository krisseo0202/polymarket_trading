import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live Polymarket client tests (requires env vars).",
    )
    parser.addoption(
        "--yes-token",
        action="store",
        default=None,
        help="YES token id for live tests (overrides env).",
    )
    parser.addoption(
        "--no-token",
        action="store",
        default=None,
        help="NO token id for live tests (overrides env).",
    )


@pytest.fixture(scope="session")
def live_mode(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--live"))


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="session")
def live_env_enabled() -> bool:
    return _env_flag("POLY_LIVE_TEST")


@pytest.fixture(scope="session")
def yes_token_id(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--yes-token") or os.getenv("YES_TOKEN_ID")


@pytest.fixture(scope="session")
def no_token_id(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--no-token") or os.getenv("NO_TOKEN_ID")