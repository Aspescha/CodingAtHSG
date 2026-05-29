"""
cleaner.py
----------
Responsible for loading and cleaning the raw luxury watch dataset.

Usage:
    from src.cleaner import WatchDataCleaner
    cleaner = WatchDataCleaner("data/Watches.csv")
    df = cleaner.clean()
"""

import datetime
import re

import numpy as np
import pandas as pd


class WatchDataCleaner:
    """
    Loads the raw Watches CSV and returns a fully cleaned DataFrame.

    Steps performed:
        1. Drop uninformative columns (index, name, reference number)
        2. Merge the duplicate condition columns
        3. Parse and validate the size column (mm)
        4. Parse and validate the year of production column
        5. Parse and validate the price column
        6. Drop rows with missing price (target variable)
        7. Rename columns to human-readable names
    """

    # Columns to drop immediately — carry no modelling information
    _COLUMNS_TO_DROP = ["Unnamed: 0", "name", "ref"]

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Path to the raw Watches.csv file.
        """
        self.filepath = filepath
        self._data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Run the full cleaning pipeline and return the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset ready for EDA and modelling.
        """
        self._data = pd.read_csv(self.filepath, low_memory=False)
        self._drop_uninformative_columns()
        self._merge_condition_columns()
        self._clean_size()
        self._clean_yop()
        self._clean_price()
        self._drop_missing_price()
        self._rename_columns()
        return self._data.copy()

    # ------------------------------------------------------------------
    # Private cleaning steps
    # ------------------------------------------------------------------

    def _drop_uninformative_columns(self) -> None:
        """Drop index, name, and reference columns."""
        cols = [c for c in self._COLUMNS_TO_DROP if c in self._data.columns]
        self._data.drop(columns=cols, inplace=True)

    def _merge_condition_columns(self) -> None:
        """
        The raw dataset contains two overlapping condition columns: 'cond'
        and 'condition'. Neither is complete on its own, but together they
        cover all rows without any conflicting values. We consolidate them
        into 'cond' and drop the redundant 'condition' column.
        """
        if "condition" in self._data.columns:
            self._data["cond"] = self._data["cond"].fillna(self._data["condition"])
            self._data.drop(columns=["condition"], inplace=True)

    def _clean_size(self) -> None:
        """
        Parse raw size strings into a single plausible case diameter (mm).
        Entries outside the range 20–60 mm are treated as missing.
        """
        self._data["size"] = (
            self._data["size"].apply(self._parse_size).astype("Int64")
        )

    def _clean_yop(self) -> None:
        """
        Extract the four-digit year from the raw year-of-production strings.
        Values below 1500 or above the current year are treated as missing.
        """
        current_year = datetime.date.today().year
        yop_numeric = (
            self._data["yop"]
            .astype(str)
            .str.strip()
            .str[:4]
            .pipe(pd.to_numeric, errors="coerce")
        )
        self._data["yop"] = (
            yop_numeric
            .where(yop_numeric.between(1500, current_year), other=pd.NA)
            .astype("Int64")
        )

    def _clean_price(self) -> None:
        """
        Strip dollar signs and commas from the price column and coerce to
        float. 'Price on request' entries and non-positive values become NaN.
        """
        self._data["price"] = (
            self._data["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
            .where(lambda s: s > 0, other=pd.NA)
            .astype("Float64")
        )

    def _drop_missing_price(self) -> None:
        """
        Drop all rows where price is missing. Price is the target variable
        and cannot be imputed.

        Note on selection bias: this step — combined with the removal of
        'Price on request' entries — systematically excludes rare, illiquid,
        or highly collectible watches whose sellers withhold the price.
        The resulting dataset skews towards well-documented, liquid,
        mid-range watches. Model predictions should be interpreted with
        this limitation in mind.
        """
        self._data.dropna(subset=["price"], inplace=True)

    def _rename_columns(self) -> None:
        """Rename abbreviated column names to human-readable equivalents."""
        self._data.rename(
            columns={
                "mvmt": "movement",
                "bracem": "bracelet_material",
                "casem": "case_material",
                "cond": "condition",
            },
            inplace=True,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size(value) -> int | None:
        """
        Extract a single plausible case diameter from a raw size string.

        Parameters
        ----------
        value : any
            Raw value from the size column.

        Returns
        -------
        int or None
            Rounded case diameter in mm, or None if no plausible value found.
        """
        if pd.isna(value):
            return pd.NA

        text = str(value).lower().strip().replace(",", ".")

        if text in ("", "nan", "none", "<na>", "unknown"):
            return pd.NA

        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
        plausible = [n for n in numbers if 20 <= n <= 60]

        if not plausible:
            return pd.NA

        return int(round(plausible[0]))