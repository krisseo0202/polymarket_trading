"""One-shot read-only check that the CLOB V2 endpoint sees our pUSD balance.

Run this after wrapping USDC.e -> pUSD to confirm the migration landed before
flipping `paper_trading: false`. Does not place orders, does not modify state.

Usage:
    .venv/bin/python scripts/check_v2_balance.py
    .venv/bin/python scripts/check_v2_balance.py --host https://clob-v2.polymarket.com

Env vars required: PRIVATE_KEY, PROXY_FUNDER (same as bot.py).
"""

import argparse
import os
import sys

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://clob.polymarket.com",
                        help="CLOB endpoint. Use https://clob-v2.polymarket.com "
                             "for the V2 testing URL pre-cutover; prod URL post.")
    parser.add_argument("--chain-id", type=int, default=137)
    parser.add_argument("--signature-type", type=int, default=1)
    args = parser.parse_args()

    pk = os.getenv("PRIVATE_KEY")
    funder = os.getenv("PROXY_FUNDER")
    if not pk or not funder:
        print("ERROR: PRIVATE_KEY and PROXY_FUNDER must be set in env.", file=sys.stderr)
        return 2

    client = ClobClient(
        host=args.host, key=pk, chain_id=args.chain_id,
        signature_type=args.signature_type, funder=funder,
    )
    creds = client.create_or_derive_api_key()
    client.set_api_creds(creds)

    print(f"host        = {args.host}")
    print(f"address     = {client.get_address()}")
    print(f"funder      = {funder}")

    raw = client.get_balance_allowance(
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    print(f"raw         = {raw}")

    bal_wei = int(raw.get("balance", 0)) if raw else 0
    allow_wei = int(raw.get("allowance", 0)) if raw else 0
    bal_pUSD = bal_wei / 1e6
    allow_pUSD = allow_wei / 1e6

    print(f"balance     = {bal_pUSD:,.6f} pUSD")
    print(f"allowance   = {allow_pUSD:,.6f} pUSD")

    if bal_pUSD == 0:
        print("\nbalance is 0. Most likely cause: USDC.e not yet wrapped to pUSD.")
        print("Wrap via the Polymarket UI or the Collateral Onramp's wrap() before going live.")
        return 1
    if allow_pUSD < bal_pUSD:
        print(f"\nWARNING: allowance ({allow_pUSD}) < balance ({bal_pUSD}).")
        print("Call client.update_balance_allowance(...) to top it up before placing orders.")
        return 1

    print("\nOK — V2 endpoint sees wrapped pUSD with sufficient allowance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
