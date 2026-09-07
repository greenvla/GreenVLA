"""Opt-in runtime wrapper for the S0/S1 GMM state guard."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .state_gmm_guard import GMMStateGuard, GuardDiagnostics


FINGER_RUNTIME_INDICES = np.asarray(
    list(range(34, 40)) + list(range(42, 48)), dtype=np.int64
)


@dataclass
class RuntimeGMMGuard:
    guard: GMMStateGuard | None
    mode: str = "off"
    log_every: int = 20
    ramp_rows: int = 8
    hard_z_limit: float = 0.0
    correct_fingers: bool = False
    logger: logging.Logger = logging.getLogger(__name__)
    last_summary: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "monitor", "correct"}:
            raise ValueError(f"Unsupported GMM guard mode: {self.mode}")
        if self.mode != "off" and self.guard is None:
            raise ValueError("GMM guard artifact is required in monitor/correct mode")
        self._calls = 0
        self._ood_calls = 0
        self._correction_feature_mask = None
        if self.guard is not None:
            self._correction_feature_mask = np.ones(
                len(self.guard.feature_indices), dtype=bool
            )
            if not self.correct_fingers:
                self._correction_feature_mask &= ~np.isin(
                    self.guard.feature_indices, FINGER_RUNTIME_INDICES
                )
        self.reset()

    @classmethod
    def from_env(cls, logger: logging.Logger | None = None) -> "RuntimeGMMGuard":
        mode = os.environ.get("S0S1_GMM_GUARD_MODE", "off").strip().lower()
        path = os.environ.get("S0S1_GMM_GUARD_PATH", "").strip()
        target_logger = logger or logging.getLogger(__name__)
        if mode == "off":
            return cls(None, mode="off", logger=target_logger)
        if not path:
            raise ValueError(
                "S0S1_GMM_GUARD_PATH is required when S0S1_GMM_GUARD_MODE "
                f"is {mode!r}"
            )
        guard = GMMStateGuard.load(
            path,
            gradient_step=float(os.environ.get("S0S1_GMM_GRADIENT_STEP", "0.05")),
            max_correction=float(os.environ.get("S0S1_GMM_MAX_CORRECTION", "0.01")),
        )
        correct_fingers = os.environ.get(
            "S0S1_GMM_CORRECT_FINGERS", "0"
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        instance = cls(
            guard,
            mode=mode,
            log_every=max(1, int(os.environ.get("S0S1_GMM_LOG_EVERY", "20"))),
            ramp_rows=max(0, int(os.environ.get("S0S1_GMM_RAMP_ROWS", "8"))),
            hard_z_limit=max(
                0.0, float(os.environ.get("S0S1_GMM_HARD_Z_LIMIT", "0"))
            ),
            correct_fingers=correct_fingers,
            logger=target_logger,
        )

        target_logger.info(
            "S0S1 GMM guard loaded mode=%s artifact=%s threshold=%.4f "
            "step=%.4f cap=%.4f hard_z=%.2f fingers=%s ramp_rows=%d",
            mode,
            Path(path).name,
            guard.threshold,
            guard.gradient_step,
            guard.max_correction,
            instance.hard_z_limit,
            "correct" if correct_fingers else "monitor-only",
            instance.ramp_rows,
        )
        return instance

    def reset(self) -> None:
        self.last_summary = {
            "mode": self.mode,
            "status": "OFF" if self.mode == "off" else "WAITING",
        }

    def _feature_name(self, feature_position: int) -> str:
        if self.guard is None:
            return "-"
        names = self.guard.metadata.get("feature_names", [])
        if feature_position < len(names):
            return str(names[feature_position])
        return f"runtime_{int(self.guard.feature_indices[feature_position])}"

    def apply(
        self,
        absolute_actions: np.ndarray,
        *,
        current_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, GuardDiagnostics | None]:
        """Score a chunk and optionally correct all non-finger position channels."""
        if self.mode == "off":
            self.last_summary = {"mode": "off", "status": "OFF"}
            return absolute_actions, None
        assert self.guard is not None
        proposed, diagnostics = self.guard.correct(
            absolute_actions,
            correction_feature_mask=self._correction_feature_mask,
            ramp_rows=self.ramp_rows,
            hard_z_limit=self.hard_z_limit,
        )
        original = np.asarray(absolute_actions)
        delta = np.asarray(proposed) - original
        selected_delta = delta[..., self.guard.feature_indices].reshape(
            -1, len(self.guard.feature_indices)
        )
        abs_by_feature = np.abs(selected_delta).max(axis=0)
        top_feature = int(np.argmax(abs_by_feature))
        scores = np.asarray(diagnostics.log_density)
        ood = np.asarray(diagnostics.ood_mask)
        selected = original[..., self.guard.feature_indices].reshape(
            -1, len(self.guard.feature_indices)
        )
        standardized = (selected - self.guard.center) / self.guard.scale
        hard_clipped_count = (
            int((np.abs(standardized) > self.hard_z_limit).sum())
            if self.hard_z_limit > 0.0
            else 0
        )
        worst_row, worst_feature = np.unravel_index(
            int(np.argmax(np.abs(standardized))), standardized.shape
        )

        current_score = None
        current_ood = None
        current_top_feature = None
        current_max_abs_z = None
        if current_state is not None:
            state_array = np.asarray(current_state)
            current_score = float(
                np.asarray(self.guard.score_samples(state_array)).reshape(-1)[0]
            )
            current_ood = bool(current_score < self.guard.threshold)
            state_selected, _ = self.guard._select(state_array)
            state_z = (state_selected[0] - self.guard.center) / self.guard.scale
            current_top_feature = int(np.argmax(np.abs(state_z)))
            current_max_abs_z = float(abs(state_z[current_top_feature]))

        self._calls += 1
        has_ood = bool(ood.any())
        self._ood_calls += int(has_ood)
        max_abs = float(abs_by_feature[top_feature])
        self.last_summary = {
            "mode": self.mode,
            "status": "OOD" if has_ood or current_ood else "ID",
            "threshold": float(self.guard.threshold),
            "current_log_density": current_score,
            "current_ood": current_ood,
            "current_max_abs_z": current_max_abs_z,
            "current_top_deviation_joint": (
                None
                if current_top_feature is None
                else self._feature_name(current_top_feature)
            ),
            "chunk_min_log_density": float(scores.min()),
            "chunk_median_log_density": float(np.median(scores)),
            "chunk_ood_count": int(ood.sum()),
            "chunk_size": int(ood.size),
            "chunk_max_abs_z": float(abs(standardized[worst_row, worst_feature])),
            "chunk_top_deviation_joint": self._feature_name(int(worst_feature)),
            "chunk_top_deviation_row": int(worst_row),
            "max_strength": float(np.asarray(diagnostics.strength).max()),
            "max_correction_norm": float(
                np.asarray(diagnostics.correction_norm).max()
            ),
            "max_correction_abs": max_abs,
            "hard_z_limit": float(self.hard_z_limit),
            "hard_clipped_count": hard_clipped_count,
            "top_correction_joint": self._feature_name(top_feature),
            "correct_fingers": bool(self.correct_fingers),
            "call": self._calls,
        }
        if has_ood or current_ood or self._calls % self.log_every == 0:
            self.logger.info(
                "S0S1 GMM mode=%s call=%d state=%s state_logp=%s "
                "chunk_ood=%d/%d min_logp=%.3f threshold=%.3f "
                "max_delta=%.5f top_joint=%s state_top=%s/%.2fz "
                "chunk_top=%s/%.2fz@%d",
                self.mode,
                self._calls,
                "OOD" if current_ood else "ID",
                "-" if current_score is None else f"{current_score:.3f}",
                int(ood.sum()),
                int(ood.size),
                float(scores.min()),
                self.guard.threshold,
                max_abs,
                self._feature_name(top_feature),
                "-"
                if current_top_feature is None
                else self._feature_name(current_top_feature),
                0.0 if current_max_abs_z is None else current_max_abs_z,
                self._feature_name(int(worst_feature)),
                float(abs(standardized[worst_row, worst_feature])),
                int(worst_row),
            )
        if self.mode == "monitor":
            return absolute_actions, diagnostics
        return proposed, diagnostics
