"""CycleRunner: one-cycle execution engine for the BTC Up/Down bot.

Replaces the large procedural loop in bot.py with a class that holds
per-cycle context (market_id, token IDs, captured slot) so intra-cycle
re-analysis doesn't need a 16-variable closure.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from enum import Enum
from typing import Optional, TYPE_CHECKING

from ..engine.cycle_snapshot import build_cycle_snapshot
from ..engine.inventory import (
    InventoryState,
    apply_fill_to_state,
    sync_inventories_to_state,
    sync_strategy_from_inventories,
)
from ..engine.slot_state import (
    SLOT_INTERVAL_S,
    SlotContext,
    apply_slot_settlement,
    fetch_slot_market,
    settle_expiring_positions,
)
from ..engine.state_store import record_realized_pnl, snapshot_chainlink_state, snapshot_strategy_state
from ..strategies.base import Signal
from ..utils.market_utils import find_updown_market

if TYPE_CHECKING:
    from ..utils.startup import Services


PENDING_SETTLEMENT_TTL_S = 3600  # drop pending settlements older than this


class CycleResult(Enum):
    OK = "ok"
    NO_MARKET = "no_market"
    CIRCUIT_BREAKER = "circuit_breaker"


class CycleRunner:
    """Executes one trading cycle and holds context for intra-cycle re-analysis."""

    def __init__(self, svc: "Services"):
        self.svc = svc
        # Per-cycle context — set at start of run_cycle, used by run_intra_cycle
        self._market_id: str = ""
        self._yes_tid: str = ""
        self._no_tid: str = ""
        self._question: str = ""
        self._captured_slot: int = 0
        self._slot_expiry_ts: float = 0.0

        # Observability: detect stuck model predictions
        self._no_prediction_streak: int = 0
        self._degenerate_streak: int = 0

        # Seed slot history in background so dashboard shows outcomes immediately.
        threading.Thread(target=self._seed_slot_history, daemon=True).start()
        # Sweep stale inventories from prior sessions (runs once, background).
        threading.Thread(target=self._sweep_stale_inventories, daemon=True).start()

    # ── Public properties ────────────────────────────────────────────────────

    def has_market(self) -> bool:
        return bool(self._yes_tid and self._no_tid)

    @property
    def yes_token_id(self) -> str:
        return self._yes_tid

    @property
    def no_token_id(self) -> str:
        return self._no_tid

    @property
    def current_market_id(self) -> str:
        return self._market_id

    # ── Main cycle ───────────────────────────────────────────────────────────

    def run_cycle(self) -> CycleResult:
        """Execute one full discover → analyze → execute → persist cycle."""
        svc = self.svc
        logger = svc.logger

        market_info = find_updown_market(
            svc.market_keywords, svc.min_volume, logger,
            slug_prefix=svc.slug_prefix,
        )
        if not market_info:
            logger.warning("Skipping cycle: market not found")
            return CycleResult.NO_MARKET

        new_market_id = market_info["market_id"]
        self._handle_rollover(new_market_id, market_info)
        self._retry_pending_settlements()

        self._market_id = new_market_id
        self._yes_tid = market_info["yes_token_id"]
        self._no_tid = market_info["no_token_id"]
        self._question = market_info.get("question", "")

        slot_ctx = svc.slot_mgr.update_from_chainlink(
            svc.chainlink_feed,
            fallback_question=self._question,
            btc_feed=svc.btc_feed,
        )
        # Log strike source and feed health every cycle for observability.
        cl_age_s = (svc.chainlink_feed.get_feed_age_ms() or 0) / 1000 if svc.chainlink_feed else None
        btc_age_s = (svc.btc_feed.get_feed_age_ms() or 0) / 1000 if svc.btc_feed else None
        logger.info(
            f"Strike from {slot_ctx.strike_source}: "
            + (f"${slot_ctx.strike_price:,.2f}" if slot_ctx.strike_price else "None")
            + (f" | chainlink age={cl_age_s:.1f}s" if cl_age_s is not None else "")
            + (f" | btc age={btc_age_s:.3f}s" if btc_age_s is not None else "")
        )
        # Chainlink sources, in order of preference:
        #   chainlink              — fresh tick in the current slot
        #   chainlink_carryforward — last-known Chainlink price (still the
        #                            settlement-correct value when BTC hasn't
        #                            moved 0.5% since slot open)
        #   regex / btc_feed       — genuinely degraded (venue drift risk)
        chainlink_sources = {"chainlink", "chainlink_carryforward"}
        if slot_ctx.strike_source not in chainlink_sources and svc.chainlink_feed is not None:
            cl_status = "connecting" if svc.chainlink_feed.is_connecting() else "stale"
            logger.warning(f"Chainlink {cl_status} — trading with degraded strike price source")

        balance = svc.client.get_balance()
        _check_daily_reset(svc.state, svc.risk_manager, logger)

        if svc.state.slot_realized_pnl <= -abs(svc.session_loss_cap):
            logger.warning(
                f"Slot loss cap reached ({svc.state.slot_realized_pnl:.2f} <= "
                f"-{svc.session_loss_cap:.2f}) — shutting down"
            )
            self.shutdown()

        if svc.risk_manager.check_circuit_breaker(balance):
            logger.warning(f"Circuit breaker active (balance={balance:.2f}) — skipping cycle")
            return CycleResult.CIRCUIT_BREAKER

        # Reconcile orders + positions
        closed_orders = svc.execution_tracker.reconcile(
            svc.client, [self._yes_tid, self._no_tid], svc.inventories
        )
        for order in closed_orders:
            tid = order.token_id
            if svc.state.active_order_ids.get(tid) == order.order_id:
                svc.state.active_order_ids[tid] = None
        # Live trading: apply PnL from fills computed before the API snapshot overwrote
        # inventory.  Paper trading handles fills immediately in execute_signals() so we
        # must NOT apply them again here (that would double-count).
        if not svc.paper_trading:
            record_realized_pnl(
                svc.state, svc.risk_manager,
                svc.execution_tracker.realized_pnl_from_fills,
            )
        sync_inventories_to_state(svc.state, svc.inventories)
        sync_strategy_from_inventories(
            svc.strategy, svc.inventories, (self._yes_tid, self._no_tid)
        )

        yes_book, no_book, positions, _bal = svc.client.fetch_market_data_parallel(
            self._yes_tid, self._no_tid
        )

        # Orphaned position cleanup
        by_token = {p.token_id: p for p in positions}
        yes_live = by_token.get(self._yes_tid)
        no_live = by_token.get(self._no_tid)
        if yes_live and yes_live.size > 0 and no_live and no_live.size > 0:
            logger.warning(
                f"Orphaned positions detected — YES: {yes_live.size:.2f}, "
                f"NO: {no_live.size:.2f}. Emitting cleanup SELLs."
            )
            from ..utils.market_utils import get_mid_price  # local import for clarity
            orphan_sigs = []
            for tok_id, outcome, live_pos, book in [
                (self._yes_tid, "YES", yes_live, yes_book),
                (self._no_tid, "NO", no_live, no_book),
            ]:
                mid = get_mid_price(book) if book else None
                best_bid = book.bids[0].price if book and book.bids else mid
                if best_bid:
                    orphan_sigs.append(Signal(
                        market_id=self._market_id,
                        outcome=outcome,
                        action="SELL",
                        confidence=1.0,
                        price=best_bid,
                        size=float(live_pos.size),
                        reason="orphan_cleanup",
                    ))
            if orphan_sigs:
                self._execute(orphan_sigs, balance, positions,
                              book_summary=_book_summary(yes_book, no_book))
                snapshot_chainlink_state(svc.state, svc.chainlink_feed, slot_mgr=svc.slot_mgr)
                svc.state_store.save(svc.state)
                positions = svc.client.get_positions()

        # Build snapshot — single assembly point for strategy + dashboard
        btc_now = svc.btc_feed.get_latest_mid() if svc.btc_feed else None
        cycle_snap = build_cycle_snapshot(
            market_id=self._market_id,
            question=self._question,
            yes_token_id=self._yes_tid,
            no_token_id=self._no_tid,
            slot_ctx=svc.slot_mgr.get(),
            btc_now=btc_now,
            yes_book=yes_book,
            no_book=no_book,
            inventories=svc.inventories,
            execution_tracker=svc.execution_tracker,
            risk_manager=svc.risk_manager,
            state=svc.state,
            paper_trading=svc.paper_trading,
        )
        market_data = cycle_snap.to_market_data(yes_book, no_book, positions, balance)

        # Online-retrain hot-swap: if the retrainer thread has a new model
        # ready AND the strategy is currently flat, swap it in before the
        # next analyze() call. Gated on is_flat so we never change models
        # mid-slot while carrying a position.
        retrainer = getattr(svc, "retrainer", None)
        if retrainer is not None and retrainer.has_ready_model():
            if svc.strategy.is_flat(by_token) and hasattr(svc.strategy, "reload_model"):
                new_dir = retrainer.consume_ready_model()
                if new_dir:
                    svc.strategy.reload_model(new_dir)
            else:
                logger.debug("Retrainer has candidate but strategy not flat — deferring swap")

        svc.strategy.set_tokens(self._market_id, self._yes_tid, self._no_tid)

        # Seed price history from REST API so features like up_mid_ret_30s
        # work immediately without waiting for 30s of ticker data.
        _seed_price_history(svc.strategy, self._yes_tid, logger)

        # Store intra-cycle context before analyzing
        self._captured_slot = svc.slot_mgr.current_slot_ts()
        self._slot_expiry_ts = self._captured_slot + SLOT_INTERVAL_S

        signals = svc.strategy.analyze(market_data)

        # Log the strategy's calculation breakdown
        _log_strategy_calc(svc.strategy, signals, logger, self)
        _log_decision(svc.strategy, signals, market_data, cycle_type="main")

        # Suppress BUY entries if insufficient time remains in the slot.
        # min_entry_window_s is independent of max_hold_seconds (which is the
        # exit cap, not an entry gate).
        time_remaining = (cycle_snap.slot_end_ts or 0) - time.time()
        min_entry = getattr(svc.strategy, "min_entry_window_s", 10)
        if time_remaining < min_entry:
            original = len(signals)
            signals = [s for s in signals if s.action == "SELL"]
            if len(signals) < original:
                logger.info(
                    f"Entry suppressed at cycle start: {time_remaining:.0f}s < "
                    f"{min_entry}s required window"
                )
                svc.strategy._reset_position_state()

        self._execute(signals, balance, positions, book_summary=_book_summary(yes_book, no_book))

        svc.state.cycle_count += 1
        cycle_snap.update_from_strategy(svc.strategy)
        cycle_snap.cycle_count = svc.state.cycle_count
        snapshot_strategy_state(svc.strategy, svc.state)
        snapshot_chainlink_state(svc.state, svc.chainlink_feed, slot_mgr=svc.slot_mgr)
        svc.state_store.save(svc.state)
        try:
            svc.snapshot_store.save(cycle_snap)
        except Exception:
            pass

        logger.info(
            f"Cycle {svc.state.cycle_count} done | market={self._market_id} "
            f"| balance={balance:.2f} | daily_pnl={svc.state.daily_realized_pnl:.4f} "
            f"| signals={len(signals)}"
        )
        return CycleResult.OK

    # ── Intra-cycle re-analysis ──────────────────────────────────────────────

    def run_intra_cycle(self) -> None:
        """Lightweight re-analyze pass called every 30s between cycle boundaries."""
        svc = self.svc
        logger = svc.logger
        try:
            now_wall = time.time()
            if SlotContext.slot_for(now_wall) != self._captured_slot:
                logger.debug("Intra-cycle skipped: market slot rolled over")
                return

            time_remaining = self._slot_expiry_ts - now_wall
            min_entry_window = getattr(svc.strategy, "min_entry_window_s", 10)

            if svc.execution_tracker and svc.inventories is not None:
                closed_orders = svc.execution_tracker.reconcile(
                    svc.client, [self._yes_tid, self._no_tid], svc.inventories
                )
                for order in closed_orders:
                    tid = order.token_id
                    if svc.state.active_order_ids.get(tid) == order.order_id:
                        svc.state.active_order_ids[tid] = None
                if not svc.paper_trading:
                    record_realized_pnl(
                        svc.state, svc.risk_manager,
                        svc.execution_tracker.realized_pnl_from_fills,
                    )
                sync_inventories_to_state(svc.state, svc.inventories)
                sync_strategy_from_inventories(
                    svc.strategy, svc.inventories, (self._yes_tid, self._no_tid)
                )

            if svc.slot_mgr is not None:
                svc.slot_mgr.update_from_chainlink(
                    svc.chainlink_feed,
                    fallback_question=self._question,
                    btc_feed=svc.btc_feed,
                )

            try:
                yes_book, no_book, positions, balance = svc.client.fetch_market_data_parallel(
                    self._yes_tid, self._no_tid
                )
            except Exception as fetch_err:
                logger.warning(f"Intra-cycle: API fetch failed ({fetch_err!r}) — skipping")
                return
            btc_now = svc.btc_feed.get_latest_mid() if svc.btc_feed else None
            intra_snap = build_cycle_snapshot(
                market_id=self._market_id,
                question=self._question,
                yes_token_id=self._yes_tid,
                no_token_id=self._no_tid,
                slot_ctx=svc.slot_mgr.get() if svc.slot_mgr else None,
                btc_now=btc_now,
                yes_book=yes_book,
                no_book=no_book,
                inventories=svc.inventories,
                execution_tracker=svc.execution_tracker,
                risk_manager=svc.risk_manager,
                state=svc.state,
                paper_trading=svc.paper_trading,
            )
            market_data = intra_snap.to_market_data(yes_book, no_book, positions, balance)
            signals = svc.strategy.analyze(market_data)
            _log_decision(svc.strategy, signals, market_data, cycle_type="intra")

            # Mirror the main-cycle Calc/Decision lines so intra-cycle
            # decisions aren't silent. Without this, skip reasons (edge_low,
            # confidence_low, tte, spread_wide, etc.) were only written to
            # the JSONL decision log via _log_decision above — not visible
            # in the live terminal output. The prefix [intra] distinguishes
            # these from main-cycle logs.
            logger.info(f"  [intra @ tte={time_remaining:.0f}s]")
            _log_strategy_calc(svc.strategy, signals, logger, self)

            if time_remaining < min_entry_window:
                buy_count = sum(1 for s in signals if s.action == "BUY")
                signals = [s for s in signals if s.action == "SELL"]
                if buy_count:
                    logger.info(
                        f"Intra-cycle: BUY suppressed ({time_remaining:.0f}s < "
                        f"{min_entry_window}s entry window) — exits only"
                    )
                    svc.strategy._reset_position_state()

            if signals:
                logger.info(f"Intra-cycle: {len(signals)} signal(s)")
                self._execute(signals, balance, positions,
                              book_summary=_book_summary(yes_book, no_book))

            # Persist every tick so dashboard sees fresh TTE/skip reason/edges
            # during watching phases, not just on signal events.
            snapshot_strategy_state(svc.strategy, svc.state)
            snapshot_chainlink_state(svc.state, svc.chainlink_feed, slot_mgr=svc.slot_mgr)
            svc.state_store.save(svc.state)
        except Exception as e:
            logger.error(f"Intra-cycle analyze error: {e}", exc_info=True)

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Cancel resting orders, close positions, save state, exit."""
        svc = self.svc
        logger = svc.logger
        logger.info("Graceful shutdown initiated")

        # Stop the retrainer daemon first so it doesn't write a new
        # candidate while we're tearing down state.
        retrainer = getattr(svc, "retrainer", None)
        if retrainer is not None:
            try:
                retrainer.stop()
            except Exception as exc:
                logger.warning(f"Retrainer stop raised: {exc}")

        for token_id, order_id in list(svc.state.active_order_ids.items()):
            if order_id:
                logger.info(f"Cancelling resting order {order_id} for token {token_id[:12]}...")
                from ..utils.market_utils import cancel_if_exists
                cancel_if_exists(svc.client, order_id, dry_run=svc.paper_trading)

        self._close_open_positions()

        if svc.btc_feed is not None:
            svc.btc_feed.stop()
        if svc.chainlink_feed is not None:
            svc.chainlink_feed.stop()

        snapshot_chainlink_state(svc.state, svc.chainlink_feed)
        svc.state_store.save(svc.state)

        trades = svc.state.session_wins + svc.state.session_losses
        win_rate = svc.state.session_wins / trades if trades > 0 else 0.0
        logger.info(
            f"Shutdown complete | cycles={svc.state.cycle_count} "
            f"| daily_pnl={svc.state.daily_realized_pnl:+.4f} "
            f"| session trades={trades} wins={svc.state.session_wins} "
            f"({win_rate:.0%} win rate)"
        )
        if svc.perf_store is not None:
            try:
                svc.perf_store.record_session(
                    svc.state,
                    strategy_name=svc.strategy_name or svc.state.strategy_name,
                    start_ts=svc.bot_start_ts,
                    end_ts=time.time(),
                    paper_trading=svc.paper_trading,
                )
            except Exception as exc:
                logger.warning(f"Failed to record session to perf.db: {exc}")
        sys.exit(0)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _handle_rollover(self, new_market_id: str, market_info: dict) -> None:
        """Detect market rollover and settle/reset slot state."""
        if not self._market_id or new_market_id == self._market_id:
            return

        svc = self.svc
        logger = svc.logger
        ended_slot_ts = SlotContext.slot_for(svc.slot_mgr._clock()) - SLOT_INTERVAL_S
        resolved_outcome = settle_expiring_positions(
            yes_token_id=self._yes_tid,
            no_token_id=self._no_tid,
            slot_ts=ended_slot_ts,
            inventories=svc.inventories,
            state=svc.state,
            risk_manager=svc.risk_manager,
            paper_trading=svc.paper_trading,
            logger=logger,
        )
        # Store resolved outcome so the dashboard never needs to call Gamma API.
        # settle_expiring_positions only calls Gamma when there are open positions;
        # fetch directly for no-position slots so history is always populated.
        if resolved_outcome is None:
            from ..engine.slot_state import fetch_slot_outcome
            resolved_outcome = fetch_slot_outcome(ended_slot_ts, logger)
        if resolved_outcome is not None:
            svc.state.slot_outcomes[ended_slot_ts] = resolved_outcome
            # Keep only the last 48h of outcomes (~576 slots)
            cutoff = ended_slot_ts - 48 * 3600
            svc.state.slot_outcomes = {
                k: v for k, v in svc.state.slot_outcomes.items() if k >= cutoff
            }
        else:
            # Outcome not available yet — queue for retry on next cycle.
            # Only queue if there were actual open positions to settle.
            has_open = any(
                svc.inventories.get(tid) is not None
                and svc.inventories[tid].position > 0
                for tid in (self._yes_tid, self._no_tid)
            )
            if has_open:
                svc.state.pending_settlements.append({
                    "slot_ts": ended_slot_ts,
                    "yes_token_id": self._yes_tid,
                    "no_token_id": self._no_tid,
                })
                logger.info(
                    f"Queued pending settlement for slot {ended_slot_ts} "
                    f"(yes={self._yes_tid[:8]}, no={self._no_tid[:8]})"
                )
        sync_inventories_to_state(svc.state, svc.inventories)

        # Freeze the settled slot PnL + outcome for the dashboard before
        # resetting the accumulator. Without this, the dashboard always
        # sees slot_realized_pnl = 0.0 on the first cycle of a new slot
        # because the reset happens before state_store.save().
        svc.state.last_slot_pnl = svc.state.slot_realized_pnl
        svc.state.last_slot_outcome = resolved_outcome or ""

        logger.info(
            f"Market rolled over: {self._market_id} → {new_market_id} | "
            f"Slot PnL for {self._market_id[:8]}: {svc.state.slot_realized_pnl:+.4f}"
        )
        svc.state.slot_realized_pnl = 0.0

        if hasattr(svc.strategy, "reset_slot_state"):
            svc.strategy.reset_slot_state()

        for old_tid in (self._yes_tid, self._no_tid):
            svc.state.active_order_ids.pop(old_tid, None)
            svc.execution_tracker.active_orders = {
                oid: o for oid, o in svc.execution_tracker.active_orders.items()
                if o.token_id != old_tid
            }
            svc.execution_tracker._last_positions_by_token.pop(old_tid, None)

    def _retry_pending_settlements(self) -> None:
        """Retry settling positions from slots where the outcome wasn't available at rollover."""
        svc = self.svc
        logger = svc.logger
        if not svc.state.pending_settlements:
            return

        still_pending = []
        for entry in svc.state.pending_settlements:
            slot_ts = entry["slot_ts"]
            yes_tid = entry["yes_token_id"]
            no_tid = entry["no_token_id"]

            # Drop entries past TTL — outcome will never arrive
            if time.time() - slot_ts > PENDING_SETTLEMENT_TTL_S:
                logger.warning(
                    f"Dropping stale pending settlement for slot {slot_ts} (>1h old)"
                )
                continue

            outcome = settle_expiring_positions(
                yes_token_id=yes_tid,
                no_token_id=no_tid,
                slot_ts=slot_ts,
                inventories=svc.inventories,
                state=svc.state,
                risk_manager=svc.risk_manager,
                paper_trading=svc.paper_trading,
                logger=logger,
            )
            if outcome is not None:
                svc.state.slot_outcomes[slot_ts] = outcome
                sync_inventories_to_state(svc.state, svc.inventories)
                logger.info(
                    f"Settled pending slot {slot_ts} → {outcome} "
                    f"(daily_pnl={svc.state.daily_realized_pnl:+.4f})"
                )
            else:
                still_pending.append(entry)

        svc.state.pending_settlements = still_pending

    def _seed_slot_history(self, n_slots: int = 12) -> None:
        """Background: fetch outcomes for the last n_slots and seed state.slot_outcomes.

        Runs once at startup so the dashboard has historical data immediately,
        even before the bot has rolled over a single slot.
        """
        svc = self.svc
        logger = svc.logger
        from ..engine.slot_state import fetch_slot_outcome

        now = time.time()
        current_slot = int(math.floor(now / SLOT_INTERVAL_S) * SLOT_INTERVAL_S)
        cutoff = current_slot - 48 * 3600

        changed = False
        # Fetch the last n_slots resolved slots (skip current, it hasn't closed yet)
        for i in range(1, n_slots + 1):
            slot_ts = current_slot - i * SLOT_INTERVAL_S
            if slot_ts < cutoff:
                break
            if slot_ts in svc.state.slot_outcomes:
                continue  # already known
            outcome = fetch_slot_outcome(slot_ts, logger)
            if outcome is not None:
                svc.state.slot_outcomes[slot_ts] = outcome
                changed = True

        if changed:
            svc.state.slot_outcomes = {
                k: v for k, v in svc.state.slot_outcomes.items() if k >= cutoff
            }
            svc.state_store.save(svc.state)
            logger.info(
                f"Seeded {sum(1 for k in svc.state.slot_outcomes if k < current_slot)} "
                "historical slot outcomes into state"
            )

    def _sweep_stale_inventories(self) -> None:
        """One-time startup sweep: settle stale inventory positions from resolved markets.

        Scans recent slots via Gamma API, matches token IDs against inventory,
        and applies synthetic settlement fills for any positions that were never settled
        (e.g., because the outcome wasn't available at rollover time).
        """
        svc = self.svc
        logger = svc.logger

        # Find stale inventory tokens (non-zero position, not the current market).
        # Hydrate svc.inventories from persisted state for any missing entries so
        # the shared settlement helper can find them.
        stale_tids: set = set()
        for tid, inv_data in svc.state.inventories.items():
            if not isinstance(inv_data, dict):
                continue
            pos = inv_data.get("position", 0)
            if pos > 0 and tid not in (self._yes_tid, self._no_tid):
                stale_tids.add(tid)
                if svc.inventories.get(tid) is None:
                    svc.inventories[tid] = InventoryState(
                        position=pos, avg_cost=inv_data.get("avg_cost", 0),
                    )

        if not stale_tids:
            return

        logger.info(f"Startup sweep: {len(stale_tids)} stale inventory positions to settle")

        current_slot = int(math.floor(time.time() / SLOT_INTERVAL_S) * SLOT_INTERVAL_S)
        settled_count = 0

        # Scan backwards through recent slots (up to 24h = 288 slots)
        for i in range(1, 289):
            if not stale_tids:
                break
            slot_ts = current_slot - i * SLOT_INTERVAL_S
            info = fetch_slot_market(slot_ts, logger)
            if info is None:
                continue

            outcome = info["outcome"]
            yes_tid = info["yes_token_id"]
            no_tid = info["no_token_id"]

            # Cache outcome opportunistically for the dashboard slot history
            svc.state.slot_outcomes.setdefault(slot_ts, outcome)

            matched_tids = stale_tids & {yes_tid, no_tid}
            if not matched_tids:
                continue

            settled_count += apply_slot_settlement(
                yes_tid, no_tid, outcome,
                svc.inventories, svc.state, svc.risk_manager,
                svc.paper_trading, logger,
                label_prefix=f"Sweep settled (slot {slot_ts})",
            )
            stale_tids -= matched_tids

        # Zero out any remaining unmatched stale positions (too old to look up)
        for tid in stale_tids:
            inv = svc.inventories.get(tid)
            if inv is not None:
                inv.position = 0
                inv.avg_cost = 0
            svc.state.inventories[tid] = {"position": 0, "avg_cost": 0}

        sync_inventories_to_state(svc.state, svc.inventories)
        svc.state_store.save(svc.state)
        logger.info(
            f"Startup sweep complete: settled {settled_count} positions, "
            f"{len(stale_tids)} zeroed (older than 24h or API miss), "
            f"daily_pnl={svc.state.daily_realized_pnl:+.4f}"
        )

    def _close_open_positions(self) -> None:
        """SELL any open positions at shutdown."""
        svc = self.svc
        logger = svc.logger
        if not self._market_id:
            logger.warning("Shutdown: no market context — cannot close open positions")
            return

        token_outcome_map = {self._yes_tid: "YES", self._no_tid: "NO"}
        for token_id, inv in svc.state.inventories.items():
            size = inv.get("position", 0) if isinstance(inv, dict) else getattr(inv, "position", 0)
            if size <= 0:
                continue
            outcome = token_outcome_map.get(token_id, "")
            if not outcome:
                continue
            logger.info(f"Shutdown: closing {outcome} position  {size:.4f}sh @ market-sell")
            try:
                svc.client.place_order(
                    market_id=self._market_id,
                    token_id=token_id,
                    outcome=outcome,
                    side="SELL",
                    price=0.01,
                    size=round(size, 4),
                )
            except Exception as exc:
                logger.error(f"Shutdown: failed to close {outcome} position: {exc}")

    def _execute(
        self,
        signals: list,
        balance: float,
        positions: list,
        book_summary: Optional[dict] = None,
    ) -> None:
        """Delegate to standalone execute_signals using this cycle's context."""
        svc = self.svc
        execute_signals(
            signals=signals,
            client=svc.client,
            strategy=svc.strategy,
            risk_manager=svc.risk_manager,
            state=svc.state,
            current_market_id=self._market_id,
            yes_token_id=self._yes_tid,
            no_token_id=self._no_tid,
            balance=balance,
            positions=positions,
            paper_trading=svc.paper_trading,
            logger=svc.logger,
            inventories=svc.inventories,
            book_summary=book_summary,
            trade_log_path=svc.trade_log_path,
        )


def execute_signals(
    signals: list,
    client,
    strategy,
    risk_manager,
    state,
    current_market_id: str,
    yes_token_id: str,
    no_token_id: str,
    balance: float,
    positions: list,
    paper_trading: bool,
    logger,
    inventories=None,
    book_summary: Optional[dict] = None,
    trade_log_path: Optional[str] = None,
) -> None:
    """Risk-gate BUY signals and place orders for all passing signals."""
    from ..utils.market_utils import cancel_if_exists

    for sig in signals:
        if sig.action == "BUY":
            if not strategy.should_enter(sig):
                logger.debug(f"BUY rejected by strategy: {sig.reason}")
                continue
            # Clip to risk limits instead of rejecting outright. The
            # strategy's Kelly sizing is a suggestion; the risk manager
            # caps it to the binding constraint (position size, total
            # exposure, or per-market exposure).
            clipped = risk_manager.calculate_position_size(
                sig, balance, positions, sig.price,
            )
            if clipped <= 0:
                logger.info(
                    f"BUY rejected by risk manager: clipped size=0 "
                    f"(original={sig.size:.2f})"
                )
                continue
            if clipped < sig.size:
                logger.info(
                    f"BUY size clipped by risk manager: "
                    f"{sig.size:.2f} → {clipped:.2f}"
                )
                sig.size = clipped

        token_id = yes_token_id if sig.outcome == "YES" else no_token_id
        existing_order_id = state.active_order_ids.get(token_id)
        if existing_order_id:
            logger.info(f"Cancelling stale order {existing_order_id}")
            cancel_if_exists(client, existing_order_id, dry_run=paper_trading)
            state.active_order_ids[token_id] = None

        try:
            order = client.place_order(
                market_id=current_market_id,
                token_id=token_id,
                outcome=sig.outcome,
                side=sig.action,
                price=sig.price,
                size=sig.size,
            )
            state.active_order_ids[token_id] = order.order_id
            state.strategy_last_signal = (
                f"{sig.action} {sig.outcome} @ {sig.price:.4f} | {sig.reason[:60]}"
            )
            state.strategy_last_signal_ts = time.time()

            # Apply fill first so realized_pnl_delta (return value of
            # apply_fill_to_state) can be attached to the trade_log entry.
            # On paper trades this is the FIFO/avg-cost realized delta; 0.0
            # for opens and nonzero only when the fill closes inventory.
            realized_pnl_delta = 0.0
            if paper_trading and inventories is not None:
                inv = inventories.get(token_id)
                if inv is None:
                    inv = InventoryState(token_id=token_id)
                    inventories[token_id] = inv
                realized_pnl_delta = apply_fill_to_state(
                    inv, sig.action, sig.price, sig.size, state, risk_manager
                )
                sync_inventories_to_state(state, inventories)

            now_ts = time.time()
            slot_expiry_ts = SlotContext.slot_for(now_ts) + SLOT_INTERVAL_S
            # Pick the edge for the side we actually took; state fields are
            # Optional[float] and may be None before the first cycle populates them.
            edge_for_side = (
                state.strategy_edge_yes if sig.outcome == "YES" else state.strategy_edge_no
            )
            state.trade_log.append({
                "ts": now_ts,
                "action": sig.action,
                "outcome": sig.outcome,
                "price": sig.price,
                "size": sig.size,
                # Enrichment for the redesigned dashboard Trades panel.
                "strategy_name": state.strategy_name,
                "slot_expiry_ts": slot_expiry_ts,
                "seconds_to_expiry": max(0.0, slot_expiry_ts - now_ts),
                "token_id": token_id,
                "confidence": sig.confidence,
                "reason": sig.reason,
                "edge": edge_for_side,
                "realized_pnl_delta": realized_pnl_delta,
            })
            state.trade_log = state.trade_log[-20:]

            if trade_log_path:
                _write_trade_log(
                    trade_log_path, state, sig, token_id, order, balance, book_summary, paper_trading
                )

            if sig.action == "BUY":
                state.strategy_status = "POSITION_OPEN"
            elif sig.action == "SELL":
                state.strategy_status = "EXITED"

            if sig.action == "BUY":
                logger.info(
                    f"BUY  {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                    f"  (cost ${sig.price * sig.size:.2f})  [{sig.reason}]"
                )
            else:
                entry_price = getattr(strategy, "_pending_exit_entry_price", None)
                if entry_price and entry_price > 0 and sig.size:
                    realized = (sig.price - entry_price) * sig.size
                    pct = (sig.price - entry_price) / entry_price * 100
                    logger.info(
                        f"SELL {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                        f"  PnL: {realized:+.2f} ({pct:+.1f}%)  [{sig.reason}]"
                    )
                else:
                    logger.info(
                        f"SELL {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                        f"  [{sig.reason}]"
                    )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")


def _seed_price_history(strategy, yes_token_id: str, logger) -> None:
    """Fetch recent price history from Polymarket REST API and seed the strategy.

    This ensures features like up_mid_ret_30s have data from second 1,
    without waiting for the ticker to accumulate 30s of observations.
    """
    if not yes_token_id or not hasattr(strategy, "seed_price_history"):
        return
    # Skip if strategy already has sufficient history
    buf = strategy._price_history.get(yes_token_id)
    if buf and len(buf) >= 5:
        return
    try:
        import requests
        resp = requests.get(
            "https://clob.polymarket.com/prices-history",
            params={"market": yes_token_id, "interval": "max"},
            timeout=5,
        )
        resp.raise_for_status()
        raw = resp.json().get("history", [])
        if not raw:
            return
        history = [(int(entry["t"]), float(entry["p"])) for entry in raw]
        history.sort()
        # Keep only last 5 minutes
        cutoff = time.time() - 300
        history = [(t, p) for t, p in history if t >= cutoff]
        if history:
            strategy.seed_price_history(yes_token_id, history)
            logger.debug(f"Seeded {len(history)} price ticks for YES token")
    except Exception as e:
        logger.debug(f"Price history seed failed: {e}")


def _check_daily_reset(state, risk_manager, logger) -> None:
    """Reset daily PnL tracking when the calendar date rolls over."""
    from datetime import date
    today = str(date.today())
    if state.daily_reset_date != today:
        logger.info(
            f"New day {today} — resetting daily PnL "
            f"(previous: {state.daily_realized_pnl:.2f})"
        )
        state.daily_realized_pnl = 0.0
        state.daily_reset_date = today
        state.session_wins = 0
        state.session_losses = 0
        risk_manager.reset_daily()


def _log_strategy_calc(strategy, signals, logger, runner=None) -> None:
    """Log the strategy's model output and edge calculations after analyze()."""
    s = strategy
    prob_yes = getattr(s, "last_prob_yes", None)
    skip_reason = getattr(s, "last_skip_reason", "")
    if prob_yes is None:
        if runner is not None:
            runner._no_prediction_streak += 1
            if runner._no_prediction_streak >= 6:
                logger.warning(
                    f"  No model prediction for {runner._no_prediction_streak} consecutive "
                    f"cycles (skip_reason={skip_reason})"
                )
                return
        logger.info("  Calc: no model prediction this cycle")
        return

    if runner is not None:
        runner._no_prediction_streak = 0
        if prob_yes < 0.01 or prob_yes > 0.99:
            runner._degenerate_streak += 1
            if runner._degenerate_streak >= 3:
                logger.warning(
                    f"  Model prediction degenerate for {runner._degenerate_streak} "
                    f"consecutive cycles (p_yes={prob_yes:.4f}) — possible feature issue"
                )
        else:
            runner._degenerate_streak = 0

    prob_no = getattr(s, "last_prob_no", None) or (1.0 - prob_yes)
    edge_yes = getattr(s, "last_edge_yes", None)
    edge_no = getattr(s, "last_edge_no", None)
    net_yes = getattr(s, "last_net_edge_yes", None)
    net_no = getattr(s, "last_net_edge_no", None)
    tte = getattr(s, "last_tte_seconds", None)
    req_edge = getattr(s, "last_required_edge", None)
    dist_bps = getattr(s, "last_distance_to_strike_bps", None)
    feat = getattr(s, "last_feature_status", "")

    def _f(v, fmt="+.3f"):
        return f"{v:{fmt}}" if v is not None else "---"

    parts = [
        f"p_yes={prob_yes:.3f} p_no={prob_no:.3f}",
        f"edge_yes={_f(edge_yes)} edge_no={_f(edge_no)}",
        f"net_yes={_f(net_yes)} net_no={_f(net_no)}",
    ]
    if req_edge is not None:
        parts.append(f"req_edge={req_edge:.3f}")
    if tte is not None:
        parts.append(f"tte={tte:.0f}s")
    if dist_bps is not None:
        parts.append(f"dist={dist_bps:+.0f}bps")
    if feat:
        parts.append(f"feat={feat}")

    decision = "NO TRADE"
    if signals:
        sig = signals[0]
        decision = f"{sig.action} {sig.outcome} {sig.size}sh @{sig.price}"
    elif skip_reason:
        decision = f"NO TRADE ({skip_reason})"

    logger.info(f"  Calc: {' | '.join(parts)}")
    logger.info(f"  Decision: {decision}")


# ── Structured decision log ──────────────────────────────────────────────
from datetime import datetime, timezone as _tz

_now_utc = datetime.now(_tz.utc)
_DECISION_LOG_DIR = os.path.join("data", _now_utc.strftime("%Y-%m-%d"))
os.makedirs(_DECISION_LOG_DIR, exist_ok=True)
_DECISION_LOG_PATH = os.path.join(
    _DECISION_LOG_DIR,
    f"decision_log_{_now_utc.strftime('%Y%m%dT%H%M%SZ')}.jsonl",
)


def _log_decision(strategy, signals, market_data, cycle_type="main") -> None:
    """Append a structured JSONL record for every prediction cycle.

    Captures: timestamp, slot, model output, features, orderbook state,
    edge, skip reason, and trade action — everything needed for post-hoc analysis.
    """
    now = time.time()
    s = strategy

    prob_yes = getattr(s, "last_prob_yes", None)
    skip_reason = getattr(s, "last_skip_reason", "")

    # Orderbook state
    q_t = getattr(s, "last_q_t", None)
    c_t = getattr(s, "last_c_t", None)

    # Edges
    edge_yes = getattr(s, "last_edge_yes", None)
    edge_no = getattr(s, "last_edge_no", None)

    # Model features (only available for logreg strategy)
    model_svc = getattr(s, "model_service", None)
    features = {}
    if model_svc is not None:
        features = getattr(model_svc, "last_features", {})

    # Trade action
    action = None
    if signals:
        sig = signals[0]
        action = {
            "side": sig.outcome,
            "action": sig.action,
            "price": sig.price,
            "size": sig.size,
            "confidence": sig.confidence,
            "reason": sig.reason,
        }

    # Orderbook from market_data
    order_books = market_data.get("order_books", {})
    yes_tid = getattr(s, "_yes_token_id", None)
    no_tid = getattr(s, "_no_token_id", None)
    ob = {}
    for tid, prefix in [(yes_tid, "yes"), (no_tid, "no")]:
        book = order_books.get(tid)
        if book:
            bids = book.bids
            asks = book.asks
            ob[f"{prefix}_bid"] = float(bids[0].price) if bids else None
            ob[f"{prefix}_ask"] = float(asks[0].price) if asks else None
            ob[f"{prefix}_bid_size"] = float(bids[0].size) if bids else None
            ob[f"{prefix}_ask_size"] = float(asks[0].size) if asks else None
            bid_d3 = sum(float(e.size) for e in bids[:3]) if bids else 0.0
            ask_d3 = sum(float(e.size) for e in asks[:3]) if asks else 0.0
            ob[f"{prefix}_bid_depth_3"] = bid_d3
            ob[f"{prefix}_ask_depth_3"] = ask_d3
            # Intra-book depth imbalance features (per-side, namespaced to
            # avoid collision with the trained model's cross-market
            # `depth_ratio` in src/models/logreg_model.py).
            _hi = max(bid_d3, ask_d3)
            _lo = max(min(bid_d3, ask_d3), 1e-9)
            ob[f"{prefix}_depth_ratio"] = _hi / _lo
            ob[f"{prefix}_depth_skew"] = math.log((bid_d3 + 1e-9) / (ask_d3 + 1e-9))

    record = {
        "ts": now,
        "slot_ts": market_data.get("slot_start_ts") or SlotContext.slot_for(now),
        "tte": getattr(s, "last_tte_seconds", None),
        "cycle_type": cycle_type,
        # Model
        "prob_yes": prob_yes,
        "model_version": getattr(s, "last_model_version", ""),
        "feature_status": getattr(s, "last_feature_status", ""),
        # Edges
        "q_t": q_t,
        "c_t": c_t,
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        # Polymarket-derived features (computed in logreg_edge._check_entry)
        "microprice_yes": getattr(s, "last_microprice_yes", None),
        "micro_delta_yes": getattr(s, "last_micro_delta_yes", None),
        "f_ret_30s": getattr(s, "last_f_ret_30s", None),
        "f_delta": getattr(s, "last_f_delta", None),
        # Decision
        "skip_reason": skip_reason,
        "action": action,
        # BTC
        "strike": market_data.get("strike_price"),
        # Features (flat dict for easy CSV conversion)
        **{f"f_{k}": v for k, v in features.items()},
        # Orderbook
        **ob,
    }

    # Remove None values to keep lines compact
    record = {k: v for k, v in record.items() if v is not None}

    os.makedirs(os.path.dirname(_DECISION_LOG_PATH), exist_ok=True)
    with open(_DECISION_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def _book_summary(yes_book, no_book) -> dict:
    return {
        "yes_bid": yes_book.bids[0].price if yes_book and yes_book.bids else None,
        "yes_ask": yes_book.asks[0].price if yes_book and yes_book.asks else None,
        "no_bid": no_book.bids[0].price if no_book and no_book.bids else None,
        "no_ask": no_book.asks[0].price if no_book and no_book.asks else None,
    }


def _write_trade_log(
    path: str, state, sig, token_id: str, order, balance: float, book_summary, paper_trading: bool
) -> None:
    """Write one structured JSONL trade record."""
    import json
    slot_ts = SlotContext.slot_for(time.time()) + SLOT_INTERVAL_S
    record = {
        "ts": time.time(), "cycle": state.cycle_count,
        "slot_expiry_ts": slot_ts, "seconds_to_expiry": slot_ts - time.time(),
        "action": sig.action, "outcome": sig.outcome,
        "price": sig.price, "size": sig.size,
        "confidence": sig.confidence, "reason": sig.reason,
        "token_id": token_id, "order_id": order.order_id,
        **(book_summary or {}), "balance": balance,
        "daily_pnl": state.daily_realized_pnl, "slot_pnl": state.slot_realized_pnl,
        "strategy_name": state.strategy_name,
        "prob_yes": state.strategy_prob_yes,
        "edge_yes": state.strategy_edge_yes, "edge_no": state.strategy_edge_no,
        "bias": state.strategy_bias,
        "model_version": state.strategy_model_version or None,
        "feature_status": state.strategy_feature_status or None,
        "chainlink_ref_price": state.chainlink_ref_price,
        "chainlink_healthy": state.chainlink_healthy or None,
        "paper_trading": paper_trading,
    }
    line = json.dumps({k: v for k, v in record.items() if v is not None}, default=str)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
