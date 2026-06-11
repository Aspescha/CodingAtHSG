"""
eda.py
------
Responsible for all exploratory data analysis visualizations.

Usage:
    from src.eda import WatchEDA
    eda = WatchEDA(df)
    eda.plot_price_distribution()
    eda.plot_brand_distribution()
    ...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class WatchEDA:
    """
    Produces exploratory visualizations for the cleaned luxury watch dataset.

    Each method is self-contained and can be called independently from the
    notebook. No method modifies the underlying DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Cleaned DataFrame produced by WatchDataCleaner.clean().
        """
        self._data = data.copy()

        # Add log_price for all visualizations that need it
        self._data["log_price"] = np.log(self._data["price"])

    # ------------------------------------------------------------------
    # Price distribution
    # ------------------------------------------------------------------

    def plot_price_distribution(self) -> None:
        """Plot the raw price distribution and its log-transformed version."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(self._data["price"], bins=50, kde=True, ax=axes[0])
        axes[0].set_title("Distribution of Price")
        axes[0].set_xlabel("Price (USD)")
        axes[0].set_ylabel("Frequency")

        sns.histplot(self._data["log_price"], bins=50, kde=True,
                     color="blue", ax=axes[1])
        axes[1].set_title("Distribution of Log Price")
        axes[1].set_xlabel("Log Price")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_price_categories(self) -> None:
        """Plot the number of observations per price range category."""
        low = (self._data["price"] < 5000).sum()
        medium = ((self._data["price"] >= 5000) &
                  (self._data["price"] < 20000)).sum()
        high = (self._data["price"] >= 20000).sum()

        categories = ["Low < $5,000", "Medium < $20,000", "High >= $20,000"]
        counts = [low, medium, high]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=categories, y=counts, hue=categories,
                    palette="viridis", legend=False)
        plt.title("Number of Observations in Each Price Range Category")
        plt.xlabel("Price Range Category")
        plt.ylabel("Number of Observations")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Categorical distributions
    # ------------------------------------------------------------------

    def plot_brand_distribution(self) -> None:
        """Plot the number of observations per brand."""
        unique_brands = self._data["brand"].value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=unique_brands.index, y=unique_brands.values,
                    hue=unique_brands.index, palette="viridis", legend=False)
        plt.title("Number of Observations for Each Brand")
        plt.xlabel("Brand")
        plt.ylabel("Number of Observations")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_model_distribution(self, top_n: int = 20) -> None:
        """
        Plot the most common watch models.

        Parameters
        ----------
        top_n : int
            Number of top models to display. Default is 20.
        """
        models_df = (
            self._data["model"]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        models_df.columns = ["model", "count"]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=models_df, x="model", y="count",
                    hue="model", palette="viridis", legend=False)
        plt.title("Number of Observations for Each Watch Model")
        plt.xlabel("Watch Model")
        plt.ylabel("Number of Observations")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Correlations with price
    # ------------------------------------------------------------------

    def plot_size_vs_price(self) -> None:
        """Scatter plot of case size vs log price with a trend line."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self._data, x="size", y="log_price",
                        color="violet", alpha=0.9, s=50)
        sns.regplot(data=self._data, x="size", y="log_price",
                    scatter=False, color="red", label="Trend")
        plt.title("Relationship between Size and Price", fontsize=14)
        plt.xlabel("Size (mm)")
        plt.ylabel("Log Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_yop_vs_price(self) -> None:
        """Scatter plot of year of production vs log price with a trend line."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self._data, x="yop", y="log_price",
                        color="violet", alpha=0.9, s=50)
        sns.regplot(data=self._data, x="yop", y="log_price",
                    scatter=False, color="red", label="Trend")
        plt.title("Relationship between Year of Production and Price", fontsize=14)
        plt.xlabel("Year of Production")
        plt.ylabel("Log Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_brand_vs_price(self, top_n: int = 10) -> None:
        """
        Boxplot of log price by brand.

        Parameters
        ----------
        top_n : int
            Number of top brands to display. Default is 10.
        """
        top_brands = self._data["brand"].value_counts().head(top_n).index
        plot_data = self._data[self._data["brand"].isin(top_brands)].copy()

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=plot_data, x="brand", y="log_price",
                    order=top_brands, color="violet")
        plt.xticks(rotation=45)
        plt.title(f"Price Distribution by Brand (Top {top_n})", fontsize=14)
        plt.xlabel("Brand")
        plt.ylabel("Log Price (USD)")
        plt.tight_layout()
        plt.show()

    def plot_case_material_vs_price(self, top_n: int = 10) -> None:
        """
        Boxplot of log price by case material.

        Parameters
        ----------
        top_n : int
            Number of top case materials to display. Default is 10.
        """
        top_materials = (self._data["case_material"]
                         .value_counts().head(top_n).index)
        plot_data = self._data[
            self._data["case_material"].isin(top_materials)].copy()

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=plot_data, x="case_material", y="log_price",
                    order=top_materials, color="violet")
        plt.xticks(rotation=45)
        plt.title(f"Price Distribution by Case Material (Top {top_n})",
                  fontsize=14)
        plt.xlabel("Case Material")
        plt.ylabel("Log Price (USD)")
        plt.tight_layout()
        plt.show()

    def plot_bracelet_material_vs_price(self, top_n: int = 10) -> None:
        """
        Boxplot of log price by bracelet material.

        Parameters
        ----------
        top_n : int
            Number of top bracelet materials to display. Default is 10.
        """
        top_materials = (self._data["bracelet_material"]
                         .value_counts().head(top_n).index)
        plot_data = self._data[
            self._data["bracelet_material"].isin(top_materials)].copy()

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=plot_data, x="bracelet_material", y="log_price",
                    order=top_materials, color="violet")
        plt.xticks(rotation=45)
        plt.title(f"Price Distribution by Bracelet Material (Top {top_n})",
                  fontsize=14)
        plt.xlabel("Bracelet Material")
        plt.ylabel("Log Price (USD)")
        plt.tight_layout()
        plt.show()

    def plot_condition_vs_price(self) -> None:
        """Boxplot of log price by watch condition."""
        condition_order = self._data["condition"].value_counts().index

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=self._data, x="condition", y="log_price",
                    order=condition_order, color="violet")
        plt.xticks(rotation=45)
        plt.title("Price Distribution by Condition", fontsize=14)
        plt.xlabel("Condition")
        plt.ylabel("Log Price (USD)")
        plt.tight_layout()
        plt.show()

    def plot_missing_values(self) -> None:
        """Bar plot showing the percentage of missing values per column."""
        missing = self._data.isna().mean() * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.index, y=missing.values, color="violet")
        plt.xticks(rotation=90)
        plt.ylabel("Percentage of Missing Values")
        plt.title("Percentage of Missing Values by Column")
        plt.tight_layout()
        plt.show()