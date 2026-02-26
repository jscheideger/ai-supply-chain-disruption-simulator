from __future__ import annotations

from typing import Dict
import os

# pandas supports dataset manipulation and feature selection.
import pandas as pd

# scikit-learn provides baseline models and evaluation utilities.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# joblib supports efficient model serialization.
import joblib


class MachineLearningModel:
    """
    Trains and serves predictive models based on simulation outputs.

    This baseline implementation trains:
    1 A classifier for stockout occurrence
    2 A regressor for percent cost increase
    """
    def __init__(self) -> None:
        """
        Initialize model references.

        Models are assigned after training or loading.
        """
        self.stockout_classifier = None
        self.cost_regressor = None

    def trainModel(self, df: pd.DataFrame, out_dir: str = "artifacts") -> Dict[str, float]:
        """
        Train both classifier and regressor models and persist them to disk.

        The training dataset is expected to include baseline and scenario metrics plus scenario parameters.
        """
        df = df.copy()

        # Classification target: whether any stockouts occur during the scenario.
        df["stockout_flag"] = (df["scenario_stockout_steps"] > 0).astype(int)

        # Regression target: percent cost increase relative to baseline.
        df["cost_increase_pct"] = (
            100.0
            * (df["scenario_total_cost"] - df["baseline_total_cost"])
            / df["baseline_total_cost"].clip(lower=1e-6)
        )

        # Feature set combines network topology features and scenario attributes.
        feature_cols = [
            "n_nodes",
            "n_edges",
            "avg_out_degree",
            "avg_in_degree",
            "avg_lead_time",
            "avg_lane_capacity",
            "kind",
            "severity",
            "duration",
            "start_step",
        ]

        X = df[feature_cols]
        y_cls = df["stockout_flag"]
        y_reg = df["cost_increase_pct"]

        # kind is categorical and must be encoded to numeric form.
        cat_cols = ["kind"]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        # Random forests are chosen as a strong baseline:
        # they model nonlinear relationships and need minimal tuning.
        clf = Pipeline(
            steps=[
                ("pre", pre),
                ("model", RandomForestClassifier(n_estimators=250, random_state=7)),
            ]
        )

        reg = Pipeline(
            steps=[
                ("pre", pre),
                ("model", RandomForestRegressor(n_estimators=350, random_state=7)),
            ]
        )

        # Train and evaluate the classifier with stratification to preserve class balance.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cls, test_size=0.25, random_state=7, stratify=y_cls
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        cls_report = classification_report(y_test, pred, output_dict=True, zero_division=0)

        # Train and evaluate the regressor.
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y_reg, test_size=0.25, random_state=7
        )
        reg.fit(Xr_train, yr_train)
        pred_r = reg.predict(Xr_test)
        mae = float(mean_absolute_error(yr_test, pred_r))

        # Persist artifacts for later use without retraining.
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(clf, f"{out_dir}/stockout_classifier.joblib")
        joblib.dump(reg, f"{out_dir}/cost_regressor.joblib")

        self.stockout_classifier = clf
        self.cost_regressor = reg

        # Return a small metric set suitable for reporting.
        metrics = {
            "cls_accuracy": float(cls_report["accuracy"]),
            "cls_precision_1": float(cls_report["1"]["precision"]) if "1" in cls_report else 0.0,
            "cls_recall_1": float(cls_report["1"]["recall"]) if "1" in cls_report else 0.0,
            "reg_mae_cost_increase_pct": mae,
        }
        return metrics

    def predictImpact(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict disruption impact for a single feature row.

        Returns:
        predicted probability of stockout and predicted percent cost increase.
        """
        if self.stockout_classifier is None or self.cost_regressor is None:
            raise ValueError("Models are not trained or loaded")

        stockout_prob = float(self.stockout_classifier.predict_proba(features_df)[0][1])
        cost_increase_pct = float(self.cost_regressor.predict(features_df)[0])

        return {
            "pred_stockout_probability": stockout_prob,
            "pred_cost_increase_pct": cost_increase_pct,
        }

    def load(self, out_dir: str = "artifacts") -> None:
        """
        Load previously trained models from disk.

        This supports fast startup for demos and repeated evaluation.
        """
        self.stockout_classifier = joblib.load(f"{out_dir}/stockout_classifier.joblib")
        self.cost_regressor = joblib.load(f"{out_dir}/cost_regressor.joblib")