#!/usr/bin/env python3
"""Shared Conformal Subscription Analytics & Contextual CTA/Retention Bandit Module.

Used by:
- aarynfans-mvp (/Users/aaryn/Projects/aarynfans-mvp)
- stellar-subs (/Users/aaryn/Projects/stellar-subs)
- deal-ghost (/Users/aaryn/Projects/deal-ghost)
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class SplitConformalLGBMRegressor:
    """Locally-weighted Split Conformal Regressor using LightGBM point + log-MAD scale models."""

    def __init__(self, alpha: float = 0.10, random_state: int = 42) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self.point_model: Any = None
        self.scale_model: Any = None
        self.q_hat: float = 1.0
        self.feature_names: List[str] = []

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.point_model.predict(X)
        sigma = np.exp(self.scale_model.predict(X))
        half_width = self.q_hat * sigma
        lower = np.maximum(0.0, mu - half_width)
        upper = mu + half_width
        return mu, lower, upper


@dataclass(frozen=True)
class SubscriberRiskEvaluation:
    subscriber_id: str
    calibrated_churn_prob: float
    churn_upper_bound_prob: float
    high_risk_churn: bool
    ltv_90d_point_usd: float
    ltv_90d_lower_usd: float
    ltv_90d_upper_usd: float
    recommended_action: str
    expected_net_retention_lift_usd: float


class _ConformalUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "SplitConformalLGBMRegressor":
            return SplitConformalLGBMRegressor
        return super().find_class(module, name)


class SubscriptionConformalPredictor:
    """Loads calibrated churn + SplitConformal 90-day LTV bundle and evaluates subscriber risk."""

    def __init__(self, bundle_path: str | Path, high_risk_upper_threshold: float = 0.65) -> None:
        self.bundle_path = Path(bundle_path)
        self.high_risk_upper_threshold = float(high_risk_upper_threshold)
        with self.bundle_path.open("rb") as f:
            self.bundle: Dict[str, Any] = _ConformalUnpickler(f).load()
        self.feature_names: List[str] = list(self.bundle["feature_names"])
        self.churn_clf = self.bundle["churn_classifier"]
        self.churn_iso = self.bundle["churn_calibrator"]
        self.conformal_ltv: SplitConformalLGBMRegressor = self.bundle["conformal_ltv_regressor"]
        self.metrics: Dict[str, Any] = dict(self.bundle.get("metrics", {}))

    def predict_batch(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        raw_probs = self.churn_clf.predict_proba(X_arr)[:, 1]
        cal_probs = np.asarray(self.churn_iso.transform(raw_probs), dtype=np.float64)
        # Finite-sample conformal binomial/calibration upper confidence bound (alpha=0.10)
        n_cal = 2000.0
        se_bound = np.sqrt(np.clip(cal_probs * (1.0 - cal_probs), 1e-4, 0.25) / (n_cal / 15.0))
        churn_upper = np.clip(cal_probs + 1.64485 * se_bound, 0.0, 1.0)
        high_risk = churn_upper > self.high_risk_upper_threshold

        X_aug = np.column_stack([X_arr, cal_probs])
        mu_ltv, lower_ltv, upper_ltv = self.conformal_ltv.predict_interval(X_aug)
        return {
            "calibrated_churn_prob": cal_probs,
            "churn_upper_bound_prob": churn_upper,
            "high_risk_churn": high_risk,
            "ltv_90d_point_usd": mu_ltv,
            "ltv_90d_lower_usd": lower_ltv,
            "ltv_90d_upper_usd": upper_ltv,
        }


class CTABanditManager:
    """Contextual LinUCB + Thompson Sampling Bandit for CTA & Subscriber Retention Campaigns."""

    DEFAULT_ARMS: Tuple[str, ...] = (
        "no_intervention_holdout",
        "vip_dm_personal_checkin",
        "renewal_discount_20pct_30d",
        "free_ppv_vault_unlock_bundle",
        "annual_tier_upgrade_25pct",
    )

    def __init__(
        self,
        arms: Sequence[str] = DEFAULT_ARMS,
        context_dim: int = 8,
        alpha_ucb: float = 0.65,
        ridge_lambda: float = 1.0,
    ) -> None:
        self.arms = list(arms)
        self.context_dim = int(context_dim)
        self.alpha_ucb = float(alpha_ucb)
        self.ridge_lambda = float(ridge_lambda)
        self.A: Dict[str, np.ndarray] = {
            arm: np.eye(self.context_dim, dtype=np.float64) * self.ridge_lambda for arm in self.arms
        }
        self.b: Dict[str, np.ndarray] = {
            arm: np.zeros(self.context_dim, dtype=np.float64) for arm in self.arms
        }
        self.counts: Dict[str, int] = {arm: 0 for arm in self.arms}
        self.reward_sums: Dict[str, float] = {arm: 0.0 for arm in self.arms}

    def _theta(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        return theta, A_inv

    def score_arms(self, context: np.ndarray) -> Dict[str, float]:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        scores: Dict[str, float] = {}
        for arm in self.arms:
            theta, A_inv = self._theta(arm)
            mean_est = float(theta @ x)
            ucb = self.alpha_ucb * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            scores[arm] = mean_est + ucb
        return scores

    def select_arm(self, context: np.ndarray, *, high_risk_churn: bool = True) -> Tuple[str, Dict[str, float]]:
        scores = self.score_arms(context)
        if not high_risk_churn:
            return "no_intervention_holdout", scores
        # Among active retention arms, pick highest UCB score
        active_arms = [a for a in self.arms if a != "no_intervention_holdout"]
        best_arm = max(active_arms, key=lambda a: scores[a])
        return best_arm, scores

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x
        self.counts[arm] += 1
        self.reward_sums[arm] += float(reward)

    def predict_expected_reward(self, arm: str, context: np.ndarray) -> float:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        theta, _ = self._theta(arm)
        return float(theta @ x)

    @staticmethod
    def doubly_robust_ope(
        contexts: np.ndarray,
        logged_arms: Sequence[str],
        logged_propensities: np.ndarray,
        logged_rewards: np.ndarray,
        target_arms: Sequence[str],
        reward_predictor: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute Doubly-Robust Off-Policy Evaluation (DR-OPE) value and uplift vs logging policy."""
        n = len(logged_rewards)
        dr_values = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x = contexts[i]
            t_arm = target_arms[i]
            l_arm = logged_arms[i]
            r_hat_target = float(reward_predictor[t_arm] @ x)
            r_hat_logged = float(reward_predictor[l_arm] @ x)
            if t_arm == l_arm:
                w = 1.0 / max(float(logged_propensities[i]), 0.05)
                dr_values[i] = r_hat_target + w * (float(logged_rewards[i]) - r_hat_logged)
            else:
                dr_values[i] = r_hat_target
        logging_mean = float(np.mean(logged_rewards))
        dr_mean = float(np.mean(dr_values))
        dr_se = float(np.std(dr_values, ddof=1) / np.sqrt(n))
        return {
            "n_episodes": int(n),
            "logging_policy_value_usd": round(logging_mean, 4),
            "doubly_robust_policy_value_usd": round(dr_mean, 4),
            "doubly_robust_se_usd": round(dr_se, 4),
            "doubly_robust_lb95_usd": round(dr_mean - 1.96 * dr_se, 4),
            "relative_uplift_pct": round((dr_mean - logging_mean) / max(abs(logging_mean), 1e-6) * 100.0, 2),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "arms": self.arms,
            "context_dim": self.context_dim,
            "alpha_ucb": self.alpha_ucb,
            "counts": self.counts,
            "mean_rewards_usd": {
                arm: round(self.reward_sums[arm] / max(self.counts[arm], 1), 4) for arm in self.arms
            },
            "theta_weights": {arm: self._theta(arm)[0].round(5).tolist() for arm in self.arms},
        }

