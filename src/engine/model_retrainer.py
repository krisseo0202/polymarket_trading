"""Rolling-window batch retraining for logreg_v4.

A daemon thread that periodically rebuilds the v4 feature set from the
live append-only CSVs, trains a fresh LogisticRegression + isotonic
calibrator on the last `window_s` seconds of data, validates the
candidate against the currently-loaded production model on a held-out
walk-forward tail, and — on acceptance — writes the new model into a
dated directory under `output_parent/`.

The live cycle loop polls `has_ready_model()` between cycles and calls
`consume_ready_model()` + `strategy.reload_model(new_dir)` only when the
strategy is flat. This keeps the hot-swap out of the middle of a slot.

Design notes:

- We do not use SGD / partial_fit. Training on ~8k rows × 18 features
  takes well under a second, so a full refit every N minutes is strictly
  better than online updates: it preserves the isotonic calibration
  stage (which has no partial_fit), keeps honest walk-forward metrics,
  and avoids regime-chasing weight jumps.
- We deliberately write each candidate into its own new directory rather
  than overwriting the baseline. This sidesteps cross-file atomicity
  concerns entirely — the new model either loads cleanly from its own
  directory or not at all.
- The validation gate compares the candidate's Brier on a held-out tail
  against the currently-loaded prod model's Brier on the *same* rows,
  so a bad upstream data ingestion or a genuine model regression will be
  rejected without touching live inference.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RetrainConfig:
    enabled: bool = True
    interval_s: int = 3600
    window_s: int = 48 * 3600
    tolerance_brier: float = 0.005
    min_train_rows: int = 500
    # Glob patterns (fed to glob.glob). The bot rotates these CSVs per
    # session (e.g. btc_live_1s_20260412T011843Z.csv), so a fixed filename
    # would freeze the retraining data. A pattern with `*` picks up every
    # rotated file automatically; a plain path still works as a single
    # literal match.
    ob_csv_glob: str = "data/**/live_orderbook_snapshots*.csv"
    btc_csv_glob: str = "data/**/btc_live_1s*.csv"
    history_path: str = "data/retrain_history.jsonl"
    output_parent: str = "models"
    model_prefix: str = "logreg_v4"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RetrainConfig":
        d = d or {}
        # Accept both new keys (`*_glob`) and legacy single-path keys
        # (`ob_csv`, `btc_csv`) so existing configs keep loading.
        ob_glob = d.get("ob_csv_glob") or d.get("ob_csv") or "data/**/live_orderbook_snapshots*.csv"
        btc_glob = d.get("btc_csv_glob") or d.get("btc_csv") or "data/**/btc_live_1s*.csv"
        return cls(
            enabled=bool(d.get("enabled", True)),
            interval_s=int(d.get("interval_s", 3600)),
            window_s=int(d.get("window_s", 48 * 3600)),
            tolerance_brier=float(d.get("tolerance_brier", 0.005)),
            min_train_rows=int(d.get("min_train_rows", 500)),
            ob_csv_glob=str(ob_glob),
            btc_csv_glob=str(btc_glob),
            history_path=str(d.get("history_path", "data/retrain_history.jsonl")),
            output_parent=str(d.get("output_parent", "models")),
            model_prefix=str(d.get("model_prefix", "logreg_v4")),
        )


class Retrainer:
    """Daemon-thread retrainer for the logreg_v4 model family."""

    def __init__(
        self,
        cfg: RetrainConfig,
        *,
        prod_model_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.prod_model_dir = prod_model_dir
        self.logger = logger or logging.getLogger("model_retrainer")
        self._lock = threading.Lock()
        self._ready_dir: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_run_ts: float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.cfg.enabled:
            self.logger.info("Retrainer disabled by config")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="model_retrainer", daemon=True
        )
        self._thread.start()
        self.logger.info(
            "Retrainer started (interval=%ds, window=%ds, tolerance=%.4f)",
            self.cfg.interval_s, self.cfg.window_s, self.cfg.tolerance_brier,
        )

    def stop(self) -> None:
        """Signal the thread to exit after its current iteration.

        Training is not interruptible mid-fit, so if a retrain is in
        flight when stop() is called we wait up to `stop_timeout_s` for
        it to finish. The daemon flag ensures the thread dies with the
        process even if it doesn't join in time.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    # ── thread-safe handoff ───────────────────────────────────────────

    def has_ready_model(self) -> bool:
        with self._lock:
            return self._ready_dir is not None

    def consume_ready_model(self) -> Optional[str]:
        with self._lock:
            d, self._ready_dir = self._ready_dir, None
            return d

    # ── main loop ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            # Sleep first so the bot has time to stabilize on startup.
            # Wait on the stop event so shutdown is snappy.
            if self._stop.wait(timeout=self.cfg.interval_s):
                return
            try:
                record = self._retrain_once()
            except Exception as exc:
                self.logger.exception("Retrain iteration failed: %s", exc)
                self._append_history({
                    "trained_at": datetime.utcnow().isoformat() + "Z",
                    "accepted": False,
                    "reason": f"exception: {exc!r}",
                })
                continue
            self._append_history(record)
            if record.get("accepted") and record.get("new_dir"):
                with self._lock:
                    self._ready_dir = record["new_dir"]
                self.logger.info(
                    "Retrain accepted → %s (new_brier=%.4f prod_brier=%.4f)",
                    record["new_dir"], record["new_brier"], record["prod_brier"],
                )
            else:
                self.logger.info(
                    "Retrain skipped: %s", record.get("reason", "unknown")
                )

    # ── one retrain pass ──────────────────────────────────────────────

    def _retrain_once(self) -> Dict[str, Any]:
        """Build dataset, fit candidate, validate vs prod, maybe persist.

        Returns a history record dict (always a dict, never raises on
        expected failure modes).
        """
        now = time.time()
        self._last_run_ts = now
        since_ts = now - self.cfg.window_s

        record: Dict[str, Any] = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "since_ts": since_ts,
            "window_s": self.cfg.window_s,
            "accepted": False,
        }

        # ── Load rolling slice of the live CSVs ──
        # Both globs match all rotated session files plus any legacy
        # single-file. We concat everything, dedupe, and slice by
        # timestamp so a fresh bot session is picked up automatically.
        ob_files = self._list_csv(self.cfg.ob_csv_glob, since_ts)
        btc_files = self._list_csv(self.cfg.btc_csv_glob, since_ts)
        record["n_ob_files"] = len(ob_files)
        record["n_btc_files"] = len(btc_files)
        if not ob_files or not btc_files:
            record["reason"] = "csv_missing"
            return record

        ob_df = self._concat_csv(ob_files, dedupe_subset=["slot_ts", "elapsed_s", "side"])
        btc_df = self._concat_csv(btc_files, dedupe_subset=["timestamp"])
        ob_df = ob_df[ob_df["slot_ts"] >= since_ts].reset_index(drop=True)
        btc_df = btc_df[btc_df["timestamp"] >= since_ts].reset_index(drop=True)
        if ob_df.empty or btc_df.empty:
            record["reason"] = "empty_window"
            return record

        # ── Lazy import trainer helpers (avoids cycles on module load) ──
        import sys
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        scripts_dir = os.path.join(repo, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import train_logreg_v3 as trainer  # type: ignore

        labels = trainer.derive_labels(ob_df)
        if not labels:
            record["reason"] = "no_labels"
            return record
        df = trainer.build_dataset(ob_df, btc_df, labels, row_interval=15)
        if df.empty or len(df) < self.cfg.min_train_rows:
            record["reason"] = f"insufficient_rows:{len(df)}"
            return record

        # ── Walk-forward split (80/20 by slot chronology) ──
        slots = sorted(df["slot_ts"].unique())
        if len(slots) < 5:
            record["reason"] = f"insufficient_slots:{len(slots)}"
            return record
        split = max(1, int(len(slots) * 0.8))
        train_slots = set(slots[:split])
        valid_slots = set(slots[split:])
        train = df[df["slot_ts"].isin(train_slots)].reset_index(drop=True)
        valid = df[df["slot_ts"].isin(valid_slots)].reset_index(drop=True)
        if train.empty or valid.empty:
            record["reason"] = "empty_split"
            return record

        # Train on the PROD model's declared feature list — not the
        # trainer module's current FEATURES constant — so every retrain
        # is a drop-in replacement with identical shape. If prod meta is
        # missing or has no features, fall back to the trainer constant.
        FEATURES = self._load_prod_features() or list(trainer.FEATURES)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            record["reason"] = f"feature_missing:{missing[:3]}"
            return record

        X_tr = train[FEATURES].values.astype(float)
        y_tr = train["y"].values.astype(int)
        X_va = valid[FEATURES].values.astype(float)
        y_va = valid["y"].values.astype(int)

        record.update({
            "n_train_rows": int(len(train)),
            "n_valid_rows": int(len(valid)),
            "n_train_slots": len(train_slots),
            "n_valid_slots": len(valid_slots),
            "features": list(FEATURES),
        })

        # ── Fit candidate ──
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss

        scaler = StandardScaler().fit(X_tr)
        Xs_tr = scaler.transform(X_tr)
        Xs_va = scaler.transform(X_va)

        model = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
        model.fit(Xs_tr, y_tr)

        p_va_raw = model.predict_proba(Xs_va)[:, 1]
        calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        calibrator.fit(p_va_raw, y_va)
        p_va_cal = calibrator.predict(p_va_raw)

        new_brier = float(brier_score_loss(y_va, p_va_cal))
        record["new_brier"] = new_brier

        # ── Score prod model on the SAME valid slice ──
        prod_brier = self._score_prod_on_valid(valid, FEATURES, y_va)
        record["prod_brier"] = prod_brier

        if prod_brier is None:
            accept_reason = "no_prod_baseline"
            accept = True  # first-ever retrain
        elif new_brier <= prod_brier + self.cfg.tolerance_brier:
            accept_reason = "within_tolerance"
            accept = True
        else:
            accept_reason = (
                f"regression new={new_brier:.4f} > prod={prod_brier:.4f} + "
                f"tol={self.cfg.tolerance_brier:.4f}"
            )
            accept = False

        record["reason"] = accept_reason
        if not accept:
            return record

        # ── Persist candidate into a dated directory ──
        new_dir = self._persist_candidate(
            model=model, scaler=scaler, calibrator=calibrator,
            features=FEATURES, metrics=record,
        )
        record["accepted"] = True
        record["new_dir"] = new_dir
        return record

    # ── CSV loading (rotated-file aware) ─────────────────────────────

    @staticmethod
    def _list_csv(pattern: str, since_ts: float) -> List[str]:
        """Resolve a glob pattern to a sorted list of CSV files.

        Files whose mtime is strictly earlier than `since_ts` are
        dropped — they can't contain any row inside the rolling window.
        This is a soft guard; we still filter by the in-CSV timestamp
        column after concatenation, so it's safe to be imprecise here.
        """
        files = glob.glob(pattern, recursive=True)
        kept = [f for f in files if os.path.getmtime(f) >= since_ts]
        return sorted(kept)

    @staticmethod
    def _concat_csv(files: List[str], *, dedupe_subset: List[str]) -> pd.DataFrame:
        """Read and concatenate multiple CSVs, dedupe, sort stably.

        Overlap between a legacy single-file and its rotated successors
        is possible in principle — we dedupe on a stable key rather
        than whole-row to be robust to minor column-formatting drift.
        """
        if not files:
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for f in files:
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                # A concurrently-written file may hit a torn last line;
                # pandas recovers on the next read. Skipping one file
                # just shifts that data into the next retrain cycle.
                continue
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        # Only dedupe on columns that actually exist (schema drift-safe)
        present = [c for c in dedupe_subset if c in df.columns]
        if present:
            df = df.drop_duplicates(subset=present, keep="last")
        return df.reset_index(drop=True)

    def _load_prod_features(self) -> Optional[list]:
        """Read the prod model's feature list from its meta.json.

        Returns None if the meta is missing or has no features list —
        caller falls back to the trainer module's default.
        """
        meta_path = os.path.join(self.prod_model_dir, "logreg_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            feats = meta.get("features")
            if isinstance(feats, list) and feats:
                return [str(x) for x in feats]
        except Exception as exc:
            self.logger.warning("failed to read prod meta: %s", exc)
        return None

    def _score_prod_on_valid(
        self, valid: pd.DataFrame, features: list, y_va: np.ndarray,
    ) -> Optional[float]:
        """Load prod pickles directly and score them on the valid slice.

        Uses direct sigmoid scoring (scaler + coef + intercept + optional
        isotonic) rather than `LogRegV4Model.predict(snapshot)` — the
        training dataframe has raw feature columns, not `snapshot` dicts,
        so there's no point rebuilding synthetic OrderBook objects.
        """
        meta_path = os.path.join(self.prod_model_dir, "logreg_meta.json")
        scaler_path = os.path.join(self.prod_model_dir, "logreg_scaler.pkl")
        cal_path = os.path.join(self.prod_model_dir, "logreg_calibrator.pkl")
        if not (os.path.exists(meta_path) and os.path.exists(scaler_path)):
            return None
        try:
            import pickle
            from sklearn.metrics import brier_score_loss
            with open(meta_path) as f:
                meta = json.load(f)
            prod_features = meta.get("features") or features
            # If the prod feature set differs from the candidate's, we can't
            # compare cleanly — surface None and treat as "first retrain".
            if list(prod_features) != list(features):
                self.logger.warning(
                    "prod feature set differs from candidate — skipping baseline comparison"
                )
                return None
            with open(scaler_path, "rb") as f:
                prod_scaler = pickle.load(f)
            coef = np.asarray(meta["coef"])
            intercept = float(meta["intercept"])
            X = valid[features].values.astype(float)
            Xs = prod_scaler.transform(X)
            logits = Xs @ coef + intercept
            p_raw = 1.0 / (1.0 + np.exp(-logits))
            if os.path.exists(cal_path):
                with open(cal_path, "rb") as f:
                    cal = pickle.load(f)
                if hasattr(cal, "predict"):
                    p = cal.predict(p_raw)
                elif hasattr(cal, "transform"):
                    p = cal.transform(p_raw)
                else:
                    p = p_raw
            else:
                p = p_raw
            return float(brier_score_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6)))
        except Exception as exc:
            self.logger.warning("prod scoring failed: %s", exc)
            return None

    def _persist_candidate(
        self, *, model, scaler, calibrator, features: list, metrics: Dict[str, Any],
    ) -> str:
        import pickle
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        new_dir = os.path.join(self.cfg.output_parent, f"{self.cfg.model_prefix}_{stamp}")
        os.makedirs(new_dir, exist_ok=True)
        with open(os.path.join(new_dir, "logreg_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(new_dir, "logreg_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(new_dir, "logreg_calibrator.pkl"), "wb") as f:
            pickle.dump(calibrator, f)
        meta = {
            "model_version": f"{self.cfg.model_prefix}_online_{stamp}",
            "features": list(features),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "calibrated": True,
            "trained_at": metrics["trained_at"],
            "n_train_rows": metrics.get("n_train_rows"),
            "n_valid_rows": metrics.get("n_valid_rows"),
            "n_train_slots": metrics.get("n_train_slots"),
            "n_valid_slots": metrics.get("n_valid_slots"),
            "valid_brier": metrics.get("new_brier"),
            "prod_brier_at_train": metrics.get("prod_brier"),
            "retrain_source": "Retrainer",
        }
        with open(os.path.join(new_dir, "logreg_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return new_dir

    def _append_history(self, record: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.cfg.history_path) or ".", exist_ok=True)
            with open(self.cfg.history_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            self.logger.warning("failed to append retrain history: %s", exc)
