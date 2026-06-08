# CodingAtHSG
Project of Jan, Daniel, Timon, Julian, Philipp and Yannick for Skills: Introduction to Programming at HSG. We create an app to estimate the value of your watch!

# Luxury Watch Price Estimator 

A data-driven tool to estimate the fair market value of any luxury watch built as the group project for Skills: Introduction to Programming at the University of St. Gallen (HSG).

**Authors:** Jan Nemeth, Daniel Kullmann, Timon, Julian, Philipp, Yannick Allmann

------------------------------------------------------------------------

## What We Are Doing

This project builds a machine learning model that predicts the resale price of a luxury watch based on its core specifications: brand, model, year of production, case material, bracelet material, movement, size, condition, and intended wearer.

We work with a Kaggle dataset of luxury watch listings, clean it, explore it, and train an XGBoost regression model on the log-transformed price. The end product is an interactive estimator that lets a user enter the specs of any watch and instantly receive an estimated market value.

## Business Case

The luxury watch market is **opaque and fragmented**. Identical watches list at different prices across dealers, and a private seller has almost no objective way to know what their watch is worth.

Concretely, this matters for:

-   **Private sellers** who inherited or want to sell a watch and have no easy way to benchmark it against the market.
-   **Buyers** who want a check before paying a dealer's asking price.
-   **Insurance** appraisals where a defensible, data driven valuation is needed.
-   **Dealers** who need a quick reality check on incoming trade ins.

------------------------------------------------------------------------

## How to Use This Notebook

### Requirements

-   A working Jupyter environment (Jupyter Lab, Jupyter Notebook, or VS Code's notebook interface)
-   The dataset file `Watches.csv` (provided in this repository also available zipped as `Watches.csv.zip`)

### Required Python Packages

Install the required packages from the requirements file:

``` bash
pip install -r requirements.txt
```

If you already have these installed locally, you can skip that cell.

### Running the Notebook

1.  Clone this repository and open the folder in VS Code (File → Open Folder).
2.  Open `LuxuryWatches.ipynb` in Jupyter.
3.  Make sure `Watches.csv` is in the same directory as the notebook (unzip `Watches.csv.zip` if needed).
4.  Run the cells top-to-bottom.

**Runtime:** The hyperparameter tuning step (Part 6) uses 5-fold cross-validation over 25 random parameter combinations and takes roughly **10 minutes** depending on your machine. Everything else runs in seconds.

### Running the Streamlit App

The fastest way is directly from the notebook:

1.  Run all cells top-to-bottom (allow ~10 minutes for Part 6).
2.  In Part 8, run the **Save** cell — this generates `watch_price_model.pkl`.
3.  Run the **Launch** cell — the app opens automatically in your browser at `http://localhost:8501`.

Alternatively, from the terminal:

``` bash
streamlit run app.py
```

> **Note:** `watch_price_model.pkl` is included in this repository so you can launch the app directly without re-training.

------------------------------------------------------------------------

## Notebook Structure

### Part 1: Dataset Loading

We load the Kaggle dataset of luxury watch listings into a pandas DataFrame and inspect the first few rows. We also briefly explain the column names (`ref`, `mvmt`, `casem`, `bracem`, `yop`, `cond`) so the reader can follow what each field represents.

### Part 2: Data Cleaning: Missing Values

We quantify missing data per column, drop columns that carry no signal (`Unnamed: 0`, `name`, `ref`), and consolidate the two redundant condition columns (`cond` and `condition`) into one without losing any rows. After this step, only \~1.5% of condition rows remain missing.

### Part 3: Data Cleaning: Datatypes

The raw dataset stores everything as strings (`"42mm"`, `"$5,000"`, `"1980 (Approximation)"`). In this section we parse and validate these into proper numeric types:

-   **`size`**: extract the case diameter from messy multi-dimension strings, with sanity checks for thickness and lug-to-lug values.
-   **`yop`**: extract a clean 4-digit production year and reject obviously invalid years.
-   **`price`**: strip currency symbols and commas, reject zero or negative prices.

We then drop rows with missing prices (the target variable cannot be missing) and rename abbreviated columns to readable names.

### Part 4: Exploratory Data Analysis (EDA)

We explore the dataset visually to understand distributions and relationships:

-   **Price**: heavily right-skewed; a log transformation brings it close to normal, motivating `log_price` as our model target.
-   **Brand & model**: Rolex dominates the dataset, with the Datejust and Daytona being the most common models.
-   **Size, year of production, brand, materials, condition**: each plotted against log-price to reveal which features carry predictive signal.

The EDA confirms that brand and material are strong price drivers, while size and year of production matter less in isolation.

### Part 5: Data Preprocessing

We apply the log transformation to the price target and prepare categorical variables for modeling. Missing categorical values are filled with `"Unknown"` so the model can treat the absence of information as its own signal. Numeric missing values are left as `NaN` because XGBoost handles them natively.

### Part 6: Machine Learning

Core modeling section, structured in three steps:

1.  **Baselines**: A `DummyRegressor` (predicts the median) and a Ridge regression (basic linear model). These establish what "trivial" performance looks like.
2.  **Encoding**: We split train/test *before* encoding to prevent data leakage. Low-cardinality columns (movement, condition, etc.) become pandas categoricals for XGBoost's native handling. High-cardinality columns (brand, model) are target-encoded — each category is replaced by its average historical price within the training set.
3.  **XGBoost tuning & training**: We run `RandomizedSearchCV` with 5-fold cross-validation to find good hyperparameters, then train the final model on the full training set.

The section ends with three diagnostic visualizations: a bar chart comparing MAE across baselines and the final model, a SHAP waterfall plot deconstructing one specific prediction, and a feature importance bar chart.

The final tuned XGBoost model reduces the average dollar error from \~\$15,000 (naive baseline) to under \$6,000.

### Part 7: Interactive Price Estimator

A user-facing prompt that walks the user through entering a watch's specs and returns an estimated price. The prompt suggests typical options based on the user's earlier selections, but also accepts rare valid combinations not currently in the dataset.

### Part 8: Streamlit

Bridges the trained model to a browser-based interface. Two cells:

1.  **Save** — exports the complete trained pipeline (TargetEncoder + XGBoost) as `watch_price_model.pkl` using joblib. Only needs to be run once.
2.  **Launch** — starts the Streamlit app and opens it automatically in your browser at `http://localhost:8501`. Safe to run multiple times — checks whether the app is already running before launching.

------------------------------------------------------------------------

## Files in This Repository

| File                    | Purpose                                                     |
|-------------------------|-------------------------------------------------------------|
| `LuxuryWatches.ipynb`   | The main notebook — open this.                             |
| `app.py`                | Streamlit web app — launched from Part 8 of the notebook.  |
| `model_utils.py`        | Helper class required to load the trained pipeline.        |
| `watch_price_model.pkl` | Pre-trained pipeline (TargetEncoder + XGBoost).            |
| `Watches.csv`           | Raw dataset (uncompressed, \~40 MB).                       |
| `Watches.csv.zip`       | Same dataset, zipped (\~7 MB).                             |
| `requirements.txt`      | Python packages needed to run the notebook and app.        |
| `README.md`             | This file.                                                 |
| `AI_reflection.md`      | Reflection on how AI tools were used during the project.   |

------------------------------------------------------------------------

## Notes & Limitations

-   The model's accuracy depends heavily on how well-represented a watch is in the training data. Predictions for very rare brands or one-of-a-kind pieces will be less reliable than predictions for common references like a Rolex Submariner.

-   Prices in the dataset reflect listed asking prices, not transaction prices, so the estimator predicts a fair *listing* price rather than a guaranteed sale price.