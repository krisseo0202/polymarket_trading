"""
cycle_scheduler.py — 5-minute cycle timing and late-window strategy trigger.

Public API
----------
detect_current_cycle(start_timestamp, cycle_len=300) -> float
    Returns seconds remaining until the end of the current cycle window.

run_last_second_strategy(start_timestamp, prob_callback, ...)
    Blocking loop that fires prob_callback once per cycle when
    ≤ trigger_window_s seconds remain in the current cycle.

Typical usage
-------------
    from src.utils.cycle_scheduler import detect_current_cycle, run_last_second_strategy
    from src.utils.market_utils import fetch_market_odds
    import threading, time

    # Anchor to the start of the current wall-clock 5-minute grid
    anchor = time.time() - (time.time() % 300)

    # Quick one-shot check
    secs_left = detect_current_cycle(anchor)
    print(f"{secs_left:.1f}s until cycle end")

    # Blocking scheduler — run in a daemon thread
    stop = threading.Event()
    def on_window():
        up, down = fetch_market_odds("12345")
        print(f"UP={up:.2%}  DOWN={down:.2%}")

    t = threading.Thread(
        target=run_last_second_strategy,
        args=(anchor, on_window),
        kwargs={"stop_event": stop},
        daemon=True,
    )
    t.start()
    time.sleep(600)
    stop.set()
"""

import logging
import threading
import time
from typing import Callable, Optional

_log = logging.getLogger(__name__)


# ── detect_current_cycle ──────────────────────────────────────────────────────

def detect_current_cycle(
    start_timestamp: float,
    cycle_len: int = 300,
) -> float:
    """
    Return the number of seconds remaining until the end of the current cycle.

    The cycle grid is anchored at ``start_timestamp`` and repeats every
    ``cycle_len`` seconds.  Given any wall-clock instant ``now``, the function
    answers: "how many seconds until the next cycle boundary?"

    Args:
        start_timestamp: Unix timestamp (``time.time()``) that anchors the
                         cycle grid.  Use the market-open time for exact
                         Polymarket cycle alignment, or ``time.time()`` at
                         bot start for a relative grid.
        cycle_len:       Cycle duration in seconds.  Default 300 (5 minutes).

    Returns:
        Seconds remaining in the current cycle, a float in ``(0.0, cycle_len]``.
        Returns ``float(cycle_len)`` when called exactly on a boundary.

    Raises:
        ValueError: If ``cycle_len`` is not a positive integer.

    Examples::

        # Anchor to the most-recent wall-clock 5-minute mark
        anchor = time.time() - (time.time() % 300)
        remaining = detect_current_cycle(anchor)
        # → e.g. 187.4  (3 min 7 s left in the current window)

        # Shorter cycles for testing
        remaining = detect_current_cycle(time.time() - 8, cycle_len=10)
        # → 2.0  (8 s elapsed in a 10-s cycle → 2 s remain)
    """
    if cycle_len <= 0:
        raise ValueError(f"cycle_len must be positive, got {cycle_len}")

    now = time.time()
    # Clamp negative elapsed (start_timestamp slightly in the future) to 0
    # so we're treated as being at the very start of cycle 0.
    elapsed = max(now - start_timestamp, 0.0)
    position_in_cycle = elapsed % cycle_len          # seconds into current cycle
    remaining = cycle_len - position_in_cycle        # seconds left
    # Guard: floating-point residual can yield remaining ≈ 0 at exact boundary
    return remaining if remaining > 1e-9 else float(cycle_len)


def cycle_index(start_timestamp: float, cycle_len: int = 300) -> int:
    """
    Return the zero-based index of the cycle that contains ``now``.

    Useful for detecting cycle rollovers::

        idx = cycle_index(anchor)
        if idx != last_idx:
            # new cycle just started

    Args:
        start_timestamp: Same anchor as :func:`detect_current_cycle`.
        cycle_len:       Cycle duration in seconds.

    Returns:
        Non-negative integer cycle index.
    """
    elapsed = max(time.time() - start_timestamp, 0.0)
    return int(elapsed // cycle_len)


# ── run_last_second_strategy ──────────────────────────────────────────────────

def run_last_second_strategy(
    start_timestamp: float,
    prob_callback: Callable[[], None],
    *,
    cycle_len: int = 300,
    trigger_window_s: float = 30.0,
    poll_interval_s: float = 0.5,
    stop_event: Optional[threading.Event] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Blocking loop that fires ``prob_callback`` once per cycle when
    ≤ ``trigger_window_s`` seconds remain in the current cycle window.

    State machine per cycle
    -----------------------
    WAITING  →  remaining > trigger_window_s
                 Log: periodic heartbeat every 60 s
    ACTIVE   →  remaining ≤ trigger_window_s, callback not yet fired
                 Log: "entering trigger window"
                 Action: call prob_callback(), record fired=True
    FIRED    →  callback already fired this cycle
                 Log: remaining countdown every 5 s (for visibility)
    [cycle boundary crossed] → reset fired=False, log "new cycle"

    Threading
    ---------
    This function **blocks** the calling thread.  Run it inside a daemon
    thread if the main thread must stay free::

        t = threading.Thread(target=run_last_second_strategy,
                             args=(anchor, my_callback),
                             kwargs={"stop_event": stop},
                             daemon=True)
        t.start()

    Args:
        start_timestamp:  Cycle-grid anchor (see :func:`detect_current_cycle`).
        prob_callback:    Callable with no arguments, called once per cycle
                          when the trigger window opens.  Any exception it
                          raises is caught, logged, and does NOT stop the loop.
        cycle_len:        Cycle duration in seconds (default 300).
        trigger_window_s: Fire the callback when this many seconds remain
                          (default 30).
        poll_interval_s:  Sleep duration between checks in seconds (default 0.5).
                          Lower values give tighter trigger latency at the cost
                          of more CPU; 0.5 s gives ≤ 0.5 s trigger jitter.
        stop_event:       ``threading.Event`` — set it to stop the loop cleanly.
                          If None, the loop runs until the process exits.
        logger:           Optional logger.  Falls back to the module logger.
    """
    log = logger or _log
    _stop = stop_event or threading.Event()  # sentinel that is never set if None

    log.info(
        f"cycle_scheduler started | cycle_len={cycle_len}s "
        f"trigger_window={trigger_window_s}s  poll={poll_interval_s}s"
    )

    fired_cycle_idx: int    = -1   # index of the last cycle we fired in
    announced_cycle_idx: int = -1  # index of the last cycle we logged "new cycle" for
    last_heartbeat_ts: float = 0.0  # wall time of last WAITING heartbeat
    last_countdown_ts: float = 0.0  # wall time of last FIRED countdown log

    _HEARTBEAT_INTERVAL  = 60.0   # log a "still waiting" message this often
    _COUNTDOWN_INTERVAL  = 5.0    # log remaining seconds this often after firing

    while not _stop.is_set():
        now        = time.time()
        remaining  = detect_current_cycle(start_timestamp, cycle_len)
        idx        = cycle_index(start_timestamp, cycle_len)
        already_fired = (fired_cycle_idx == idx)

        # ── Cycle boundary: log once when we first see a new cycle index ──
        if idx > announced_cycle_idx:
            if announced_cycle_idx >= 0:   # skip the very first iteration
                log.info(
                    f"[cycle {idx}] New cycle started "
                    f"({cycle_len}s window, "
                    f"trigger opens at T-{trigger_window_s:.0f}s)"
                )
            announced_cycle_idx = idx

        # ── WAITING: more than trigger_window_s remain ────────────────────
        if remaining > trigger_window_s:
            if now - last_heartbeat_ts >= _HEARTBEAT_INTERVAL:
                log.debug(
                    f"[cycle {idx}] Waiting — "
                    f"{remaining:.1f}s until cycle end "
                    f"(trigger window opens in "
                    f"{remaining - trigger_window_s:.1f}s)"
                )
                last_heartbeat_ts = now

            _stop.wait(timeout=poll_interval_s)
            continue

        # ── ACTIVE: inside the trigger window, not yet fired ─────────────
        if not already_fired:
            log.info(
                f"[cycle {idx}] *** Trigger window open — "
                f"{remaining:.1f}s remaining (≤ {trigger_window_s:.0f}s threshold) ***"
            )
            log.info(f"[cycle {idx}] Calling probability callback …")
            try:
                prob_callback()
                log.info(f"[cycle {idx}] Probability callback completed successfully")
            except Exception as exc:
                log.error(
                    f"[cycle {idx}] prob_callback raised {type(exc).__name__}: {exc}",
                    exc_info=True,
                )
            fired_cycle_idx = idx
            last_countdown_ts = now

        # ── FIRED: callback done, just log the countdown ──────────────────
        else:
            if now - last_countdown_ts >= _COUNTDOWN_INTERVAL:
                log.info(
                    f"[cycle {idx}] Countdown — {remaining:.1f}s remaining "
                    f"in cycle (callback already fired)"
                )
                last_countdown_ts = now

        _stop.wait(timeout=poll_interval_s)

    log.info("cycle_scheduler stopped (stop_event set)")


# ── Convenience: wall-clock 5-minute grid anchor ──────────────────────────────

def aligned_cycle_anchor(cycle_len: int = 300) -> float:
    """
    Return the Unix timestamp of the most-recent wall-clock cycle boundary.

    E.g. with cycle_len=300 and current time 14:07:42, this returns the
    timestamp for 14:05:00 — so the scheduler aligns to the natural
    5-minute grid rather than the bot's start time.

    Args:
        cycle_len: Cycle duration in seconds (default 300).

    Returns:
        Unix timestamp of the most-recent cycle boundary.
    """
    now = time.time()
    return now - (now % cycle_len)


# ── Demo / smoke test ─────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test demo.  Runs two short 10-second cycles with a 3-second trigger
    window, printing behaviour to stdout.

    Run with:
        python src/utils/cycle_scheduler.py
        python -m src.utils.cycle_scheduler
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("cycle_scheduler.demo")

    CYCLE_LEN      = 10    # short cycles for demo
    TRIGGER_WINDOW = 3     # fire when ≤ 3 s remain
    NUM_CYCLES     = 2

    anchor = time.time()
    stop   = threading.Event()

    call_count = [0]  # mutable for closure

    def fake_prob_callback() -> None:
        call_count[0] += 1
        remaining = detect_current_cycle(anchor, CYCLE_LEN)
        up, down = 0.62, 0.38   # simulated
        log.info(
            f"[callback #{call_count[0]}] "
            f"Simulated odds → UP={up:.2%}  DOWN={down:.2%}  "
            f"({remaining:.2f}s left in cycle)"
        )

    # Stop after NUM_CYCLES complete
    def _stopper() -> None:
        time.sleep(CYCLE_LEN * NUM_CYCLES + 1)
        stop.set()

    threading.Thread(target=_stopper, daemon=True).start()

    log.info(f"Demo: {NUM_CYCLES}×{CYCLE_LEN}s cycles, trigger at T-{TRIGGER_WINDOW}s")
    log.info(f"detect_current_cycle → {detect_current_cycle(anchor, CYCLE_LEN):.2f}s remaining at start")

    run_last_second_strategy(
        start_timestamp=anchor,
        prob_callback=fake_prob_callback,
        cycle_len=CYCLE_LEN,
        trigger_window_s=TRIGGER_WINDOW,
        poll_interval_s=0.1,
        stop_event=stop,
        logger=log,
    )

    log.info(f"Demo complete. Callback fired {call_count[0]} time(s).")


if __name__ == "__main__":
    _demo()
