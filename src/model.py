"""
model.py
--------
Responsible for preprocessing, training, evaluating, and predicting
luxury watch prices using an XGBoost pipeline.

Usage:
    from src.model import WatchPriceModel

    # Training
    model = WatchPriceModel()
    model.fit(df)
    model.save("models/watch_price_model.joblib")

    # Prediction (e.g. in Streamlit)
    model = WatchPriceModel()
    model.load("models/watch_price_model.joblib")
    price = model.predict(brand="Rolex", model_name="Daytona", ...)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

import category_encoders as ce
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class WatchPriceModel:
    """
    Full modelling pipeline for luxury watch price prediction.

    The pipeline handles:
        - Log transformation of the target variable (price)
        - Filling missing categoricals with 'Unknown'
        - Native XGBoost categorical handling for low-cardinality columns
        - Target encoding for high-cardinality columns (brand, model),
          fitted inside cross-validation folds to prevent data leakage
        - Hyperparameter tuning via RandomizedSearchCV
        - Model persistence via joblib
    """

    # Low-cardinality categoricals handled natively by XGBoost
    _LOW_CARD_COLS = [
        "movement", "case_material", "bracelet_material", "condition", "sex"
    ]

    # High-cardinality categoricals encoded via TargetEncoder
    _HIGH_CARD_COLS = ["brand", "model"]

    # Categorical columns that receive 'Unknown' for missing values
    _CATEGORICAL_COLS = _LOW_CARD_COLS + _HIGH_CARD_COLS

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None
        self._watch_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, test_size: float = 0.2,
            n_iter: int = 5, random_state: int = 42) -> None:
        """
        Preprocess the data and train the XGBoost pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Cleaned DataFrame from WatchDataCleaner.clean().
        test_size : float
            Fraction of data held out for final evaluation. Default 0.2.
        n_iter : int
            Number of hyperparameter combinations to try. Default 5.
        random_state : int
            Random seed for reproducibility. Default 42.
        """
        self._watch_data = data.copy()
        prepared = self._preprocess(data)
        self._split(prepared, test_size, random_state)
        self._train(n_iter, random_state)

    def evaluate(self) -> dict:
        """
        Evaluate the trained model against both baselines on the test set.

        Returns
        -------
        dict
            Dictionary containing MAE, MAPE, and RMSE for all three models.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        y_dollars = np.exp(self._y_test)

        # --- Dummy baseline ---
        dummy = DummyRegressor(strategy="median")
        dummy.fit(self._X_train, self._y_train)
        dummy_preds = np.exp(dummy.predict(self._X_test))

        # --- Ridge baseline ---
        ridge_preds = np.exp(self._ridge_pipeline.predict(self._X_test))

        # --- XGBoost ---
        xgb_preds = np.exp(self._pipeline.predict(self._X_test))

        results = {
            "dummy": {
                "MAE":  mean_absolute_error(y_dollars, dummy_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, dummy_preds) * 100,
            },
            "ridge": {
                "MAE":  mean_absolute_error(y_dollars, ridge_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, ridge_preds) * 100,
            },
            "xgboost": {
                "MAE":  mean_absolute_error(y_dollars, xgb_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, xgb_preds) * 100,
                "RMSE": np.sqrt(mean_squared_error(y_dollars, xgb_preds)),
            },
        }

        self._print_evaluation(results)
        return results

    def plot_model_comparison(self, results: dict) -> None:
        """
        Bar chart comparing MAPE across Dummy, Ridge, and XGBoost.

        Parameters
        ----------
        results : dict
            Output from evaluate().
        """
        model_names = [
            "Dummy Baseline\n(Guess Median)",
            "Ridge Baseline\n(Linear)",
            "Final Tuned\nXGBoost Model",
        ]
        mapes = [
            results["dummy"]["MAPE"],
            results["ridge"]["MAPE"],
            results["xgboost"]["MAPE"],
        ]
        colors = ["#a6cee3", "#5cb8d1", "#1f78b4"]

        plt.figure(figsize=(10, 5))
        bars = plt.barh(model_names, mapes, color=colors)
        plt.xlabel("Mean Absolute Percentage Error (%) - Lower is Better",
                   fontsize=12)
        plt.title("Model Performance Comparison (MAPE)", fontsize=14)
        plt.gca().invert_yaxis()

        for bar in bars:
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%",
                va="center", ha="left", fontsize=12, fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    def plot_shap(self, instance_index: int = 0) -> None:
        """
        SHAP waterfall plot for a single test set observation.

        Parameters
        ----------
        instance_index : int
            Index of the observation in X_test to explain. Default 0.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        fitted_encoder = self._pipeline.named_steps["target_encoder"]
        fitted_xgb = self._pipeline.named_steps["xgb"]

        X_encoded = fitted_encoder.transform(self._X_test)
        for col in self._LOW_CARD_COLS:
            X_encoded[col] = X_encoded[col].astype("category")

        explainer = shap.TreeExplainer(fitted_xgb)
        shap_values = explainer(X_encoded)

        shap.plots.waterfall(shap_values[instance_index], max_display=10)

        actual = np.exp(self._y_test.iloc[instance_index])
        predicted = np.exp(self._pipeline.predict(
            self._X_test.iloc[[instance_index]]
        )[0])

        print(f"\nActual Price:    ${actual:,.2f}")
        print(f"Predicted Price: ${predicted:,.2f}")

    def plot_feature_importance(self) -> None:
        """Bar chart of XGBoost feature importances."""
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        fitted_xgb = self._pipeline.named_steps["xgb"]
        importance_df = (
            pd.DataFrame({
                "Feature": self._X_train.columns,
                "Importance": fitted_xgb.feature_importances_,
            })
            .sort_values("Importance", ascending=True)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"],
                 color="#ff77b4")
        plt.xlabel("Importance Score (Relative Weight)")
        plt.title("XGBoost Feature Importances: What Drives the Price?")
        plt.tight_layout()
        plt.show()

    def predict(self, brand: str, model_name: str, case_material: str,
                condition: str, movement: str, bracelet_material: str,
                sex: str, size: float | None, yop: int | None) -> float:
        """
        Predict the market price for a single watch.

        Parameters
        ----------
        brand : str
        model_name : str
        case_material : str
        condition : str
        movement : str
        bracelet_material : str
        sex : str
        size : float or None
        yop : int or None

        Returns
        -------
        float
            Estimated market price in USD.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained or loaded. "
                               "Call fit() or load() first.")

        user_data = pd.DataFrame({
            "brand":             [brand],
            "model":             [model_name],
            "case_material":     [case_material],
            "condition":         [condition],
            "size":              [size],
            "movement":          [movement],
            "yop":               pd.array([yop], dtype="Int64"),
            "bracelet_material": [bracelet_material],
            "sex":               [sex],
        })

        for col in self._LOW_CARD_COLS:
            user_data[col] = user_data[col].astype("category")

        user_data = user_data[self._X_train.columns]
        log_price = self._pipeline.predict(user_data)[0]
        return float(np.exp(log_price))

    def get_valid_options(self) -> dict:
        """
        Return valid input options for the Streamlit UI, derived from
        the training data.

        Returns
        -------
        dict
            Keys are column names, values are sorted lists of valid options.
        """
        if self._watch_data is None:
            raise RuntimeError("Model not trained or loaded. "
                               "Call fit() or load() first.")

        return {
            "brands":    self._watch_data["brand"].value_counts().index.tolist(),
            "models":    self._watch_data["model"].value_counts().index.tolist(),
            "cases":     self._watch_data["case_material"].value_counts().index.tolist(),
            "conditions": self._watch_data["condition"].value_counts().index.tolist(),
            "movements": self._watch_data["movement"].value_counts().index.tolist(),
            "bracelets": self._watch_data["bracelet_material"].value_counts().index.tolist(),
            "sexes":     self._watch_data["sex"].value_counts().index.tolist(),
            "size_range": (
                int(self._watch_data["size"].min()),
                int(self._watch_data["size"].max()),
            ),
            "yop_range": (
                int(self._watch_data["yop"].min()),
                int(self._watch_data["yop"].max()),
            ),
        }

    def save(self, filepath: str) -> None:
        """
        Save the trained pipeline and training data to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model, e.g. 'models/watch_price_model.joblib'.
        """
        if self._pipeline is None:
            raise RuntimeError("Nothing to save. Call fit() first.")

        joblib.dump({
            "pipeline":   self._pipeline,
            "X_train":    self._X_train,
            "watch_data": self._watch_data,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load a previously saved pipeline from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        """
        checkpoint = joblib.load(filepath)
        self._pipeline   = checkpoint["pipeline"]
        self._X_train    = checkpoint["X_train"]
        self._watch_data = checkpoint["watch_data"]
        print(f"Model loaded from {filepath}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation and fill missing categoricals."""
        df = data.copy()
        df["log_price"] = np.log(df["price"])
        df.drop(columns=["price"], inplace=True)
        df[self._CATEGORICAL_COLS] = df[self._CATEGORICAL_COLS].fillna("Unknown")
        return df

    def _split(self, data: pd.DataFrame, test_size: float,
               random_state: int) -> None:
        """Split into train/test and apply category dtypes."""
        X = data.drop(columns=["log_price"])
        y = data["log_price"]

        self._X_train, self._X_test, self._y_train, self._y_test = (
            train_test_split(X, y, test_size=test_size,
                             random_state=random_state)
        )

        for col in self._LOW_CARD_COLS:
            self._X_train[col] = self._X_train[col].astype("category")
            self._X_test[col]  = self._X_test[col].astype("category")

        # Fit Ridge baseline here so it shares the same train/test split
        self._fit_ridge_baseline()

    def _fit_ridge_baseline(self) -> None:
        """Fit the Ridge baseline pipeline on the training data."""
        preprocessor = ColumnTransformer(transformers=[
            ("num", SimpleImputer(strategy="median"), ["size", "yop"]),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot",  OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False)),
            ]), self._CATEGORICAL_COLS),
        ])

        self._ridge_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("ridge", Ridge()),
        ])
        self._ridge_pipeline.fit(self._X_train, self._y_train)

    def _train(self, n_iter: int, random_state: int) -> None:
        """Build and tune the XGBoost pipeline via RandomizedSearchCV."""
        xgb_model = xgb.XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            random_state=random_state,
            objective="reg:squarederror",
        )

        pipeline = Pipeline([
            ("target_encoder", ce.TargetEncoder(
                cols=self._HIGH_CARD_COLS,
                min_samples_leaf=10,
                smoothing=10,
            )),
            ("xgb", xgb_model),
        ])

        param_distributions = {
            "xgb__learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
            "xgb__max_depth":        [4, 5, 6, 7, 8, 9, 10],
            "xgb__n_estimators":     [100, 300, 500, 750, 1000],
            "xgb__subsample":        [0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=5,
            verbose=2,
            random_state=random_state,
            n_jobs=1,
        )

        search.fit(self._X_train, self._y_train)
        self._pipeline = search.best_estimator_
        print("Best parameters found:")
        print(search.best_params_)

    @staticmethod
    def _print_evaluation(results: dict) -> None:
        """Print a formatted evaluation summary."""
        print("--- Baseline Benchmarks ---")
        print(f"Dummy  MAE: ${results['dummy']['MAE']:,.2f}  | "
              f"MAPE: {results['dummy']['MAPE']:.1f}%")
        print(f"Ridge  MAE: ${results['ridge']['MAE']:,.2f}  | "
              f"MAPE: {results['ridge']['MAPE']:.1f}%")
        print("\n" + "=" * 30)
        print("--- Final XGBoost Evaluation ---")
        print("=" * 30)
        print(f"MAE:  ${results['xgboost']['MAE']:,.2f}")
        print(f"MAPE: {results['xgboost']['MAPE']:.1f}%")
        print(f"RMSE: ${results['xgboost']['RMSE']:,.2f}")