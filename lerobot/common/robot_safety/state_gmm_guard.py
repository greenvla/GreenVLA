"""Runtime GMM guard for S0/S1 humanoid joint-position trajectories.

The guard models the density of training poses in a standardized PCA space.
It never changes velocity channels and only corrects samples whose log density
falls below a threshold calibrated on training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.exp(values - maximum).sum(axis=axis)
    )


@dataclass(frozen=True)
class GuardDiagnostics:
    log_density: np.ndarray
    ood_mask: np.ndarray
    strength: np.ndarray
    correction_norm: np.ndarray


class GMMStateGuard:
    """Score and gently correct absolute S0/S1 state/action vectors."""

    def __init__(
        self,
        *,
        feature_indices: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        pca_components: np.ndarray,
        residual_components: np.ndarray | None = None,
        residual_variances: np.ndarray | None = None,
        weights: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
        threshold: float,
        score_softness: float,
        gradient_step: float = 0.2,
        max_correction: float = 0.08,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.pca_components = np.asarray(pca_components, dtype=np.float64)
        self.residual_components = np.asarray(
            residual_components
            if residual_components is not None
            else np.empty((0, len(self.feature_indices))),
            dtype=np.float64,
        )
        self.residual_variances = np.asarray(
            residual_variances
            if residual_variances is not None
            else np.empty((0,)),
            dtype=np.float64,
        )
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.variances = np.asarray(variances, dtype=np.float64)
        self.threshold = float(threshold)
        self.score_softness = max(float(score_softness), 1e-6)
        self.gradient_step = float(gradient_step)
        self.max_correction = float(max_correction)
        self.metadata = dict(metadata or {})
        self._validate()
        latent_dim = self.means.shape[1]
        self._component_log_norm = (
            np.log(np.maximum(self.weights, 1e-300))
            - 0.5
            * (
                latent_dim * np.log(2.0 * np.pi)
                + np.log(self.variances).sum(axis=1)
            )
        )
        self._residual_log_norm = -0.5 * (
            len(self.residual_variances) * np.log(2.0 * np.pi)
            + np.log(self.residual_variances).sum()
        )

    def _validate(self) -> None:
        feature_dim = len(self.feature_indices)
        if self.center.shape != (feature_dim,) or self.scale.shape != (feature_dim,):
            raise ValueError("center/scale do not match feature_indices")
        if self.pca_components.shape[1] != feature_dim:
            raise ValueError("PCA components do not match feature dimension")
        if self.residual_components.shape[1] != feature_dim:
            raise ValueError("Residual PCA components do not match feature dimension")
        if self.residual_components.shape[0] != len(self.residual_variances):
            raise ValueError("Residual PCA components/variances do not match")
        if self.means.shape != self.variances.shape:
            raise ValueError("GMM means and variances must have identical shapes")
        if self.means.shape[0] != len(self.weights):
            raise ValueError("GMM weights do not match component count")
        if self.means.shape[1] != self.pca_components.shape[0]:
            raise ValueError("GMM latent dimension does not match PCA")
        if (
            np.any(self.scale <= 0)
            or np.any(self.variances <= 0)
            or np.any(self.residual_variances <= 0)
        ):
            raise ValueError("scale and variances must be positive")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        gradient_step: float | None = None,
        max_correction: float | None = None,
    ) -> "GMMStateGuard":
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            residual_components = (
                artifact["residual_components"]
                if "residual_components" in artifact.files
                else None
            )
            residual_variances = (
                artifact["residual_variances"]
                if "residual_variances" in artifact.files
                else None
            )
            kwargs = dict(
                feature_indices=artifact["feature_indices"],
                center=artifact["center"],
                scale=artifact["scale"],
                pca_components=artifact["pca_components"],
                residual_components=residual_components,
                residual_variances=residual_variances,
                weights=artifact["weights"],
                means=artifact["means"],
                variances=artifact["variances"],
                threshold=float(artifact["threshold"].item()),
                score_softness=float(artifact["score_softness"].item()),
                gradient_step=(
                    float(metadata.get("gradient_step", 0.2))
                    if gradient_step is None
                    else gradient_step
                ),
                max_correction=(
                    float(metadata.get("max_correction", 0.08))
                    if max_correction is None
                    else max_correction
                ),
                metadata=metadata,
            )
        return cls(**kwargs)

    def _select(self, states: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        states = np.asarray(states)
        if states.shape[-1] <= int(self.feature_indices.max()):
            raise ValueError(
                f"Expected state/action dim > {self.feature_indices.max()}, got {states.shape}"
            )
        leading_shape = states.shape[:-1]
        selected = states[..., self.feature_indices].reshape(-1, len(self.feature_indices))
        if not np.isfinite(selected).all():
            raise ValueError("GMM guard received NaN/Inf joint positions")
        return selected.astype(np.float64, copy=False), leading_shape

    def _latent(
        self, selected: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        standardized = (selected - self.center) / self.scale
        latent = standardized @ self.pca_components.T
        residual = standardized @ self.residual_components.T
        return standardized, latent, residual

    def _residual_log_density(self, residual: np.ndarray) -> np.ndarray:
        if residual.shape[1] == 0:
            return np.zeros(len(residual), dtype=np.float64)
        return self._residual_log_norm - 0.5 * (
            np.square(residual) / self.residual_variances[None, :]
        ).sum(axis=1)

    def _log_components(
        self, latent: np.ndarray, residual: np.ndarray
    ) -> np.ndarray:
        diff = latent[:, None, :] - self.means[None, :, :]
        mahalanobis = np.square(diff) / self.variances[None, :, :]
        return (
            self._component_log_norm[None, :]
            - 0.5 * mahalanobis.sum(axis=2)
            + self._residual_log_density(residual)[:, None]
        )

    def score_samples(self, states: np.ndarray) -> np.ndarray:
        selected, leading_shape = self._select(states)
        _, latent, residual = self._latent(selected)
        scores = _logsumexp(self._log_components(latent, residual), axis=1)
        return scores.reshape(leading_shape)

    def log_density_gradient(self, states: np.ndarray) -> np.ndarray:
        """Return gradient of log density in original selected-joint units."""
        selected, leading_shape = self._select(states)
        _, latent, residual = self._latent(selected)
        log_components = self._log_components(latent, residual)
        log_density = _logsumexp(log_components, axis=1)
        responsibilities = np.exp(log_components - log_density[:, None])
        grad_latent = (
            responsibilities[:, :, None]
            * (self.means[None, :, :] - latent[:, None, :])
            / self.variances[None, :, :]
        ).sum(axis=1)
        grad_standardized = grad_latent @ self.pca_components
        if residual.shape[1]:
            grad_residual = -residual / self.residual_variances[None, :]
            grad_standardized += grad_residual @ self.residual_components
        grad_original = grad_standardized / self.scale
        return grad_original.reshape(*leading_shape, len(self.feature_indices))

    def correct(
        self,
        states: np.ndarray,
        *,
        correction_feature_mask: np.ndarray | None = None,
        ramp_rows: int = 0,
        hard_z_limit: float = 0.0,
    ) -> tuple[np.ndarray, GuardDiagnostics]:
        """Correct OOD absolute states/actions, preserving disabled channels.

        ``correction_feature_mask`` uses the modeled feature order rather than
        the full action layout. Scoring always uses every modeled channel; the
        mask only controls which commands may change. ``ramp_rows`` introduces
        correction gradually at the beginning of an action chunk.
        """
        original = np.asarray(states)
        selected, leading_shape = self._select(original)
        standardized, latent, residual = self._latent(selected)
        log_components = self._log_components(latent, residual)
        scores = _logsumexp(log_components, axis=1)
        responsibilities = np.exp(log_components - scores[:, None])
        grad_latent = (
            responsibilities[:, :, None]
            * (self.means[None, :, :] - latent[:, None, :])
            / self.variances[None, :, :]
        ).sum(axis=1)
        grad_standardized = grad_latent @ self.pca_components
        if residual.shape[1]:
            grad_residual = -residual / self.residual_variances[None, :]
            grad_standardized += grad_residual @ self.residual_components

        deficit = np.maximum(self.threshold - scores, 0.0)
        strength = np.clip(deficit / self.score_softness, 0.0, 1.0)
        grad_norm = np.linalg.norm(grad_standardized, axis=1, keepdims=True)
        direction = grad_standardized / np.maximum(grad_norm, 1.0)
        delta_standardized = self.gradient_step * strength[:, None] * direction
        delta_original = delta_standardized * self.scale
        delta_original = np.clip(
            delta_original, -self.max_correction, self.max_correction
        )
        enabled = np.ones(len(self.feature_indices), dtype=bool)
        if correction_feature_mask is not None:
            enabled = np.asarray(correction_feature_mask, dtype=bool)
            if enabled.shape != (len(self.feature_indices),):
                raise ValueError(
                    "correction_feature_mask must match the modeled feature count"
                )
            delta_original[:, ~enabled] = 0.0
        if ramp_rows > 0 and len(delta_original) > 0:
            count = min(int(ramp_rows), len(delta_original))
            ramp = np.linspace(0.2, 1.0, count, dtype=np.float64)
            delta_original[:count] *= ramp[:, None]

        # A local gradient step cannot recover a target that is already far
        # outside train support. The optional envelope is a final fail-safe
        # around the learned per-feature distribution.
        if hard_z_limit > 0.0:
            lower = self.center - float(hard_z_limit) * self.scale
            upper = self.center + float(hard_z_limit) * self.scale
            bounded = np.clip(selected + delta_original, lower, upper)
            bounded[:, ~enabled] = selected[:, ~enabled]
            delta_original = bounded - selected

        corrected = original.astype(np.result_type(original.dtype, np.float32), copy=True)
        flat = corrected.reshape(-1, corrected.shape[-1])
        flat[:, self.feature_indices] = selected + delta_original
        diagnostics = GuardDiagnostics(
            log_density=scores.reshape(leading_shape),
            ood_mask=(scores < self.threshold).reshape(leading_shape),
            strength=strength.reshape(leading_shape),
            correction_norm=np.linalg.norm(delta_original, axis=1).reshape(leading_shape),
        )
        return corrected, diagnostics
