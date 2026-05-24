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

Existing platforms (Chrono24, WatchCharts) give you list prices for exactly that watch, but if your specific configuration isn't currently listed, you're guessing. A model trained on tens of thousands of historical listings can interpolate: it knows what a 2015 Submariner with a steel bracelet in good condition tends to fetch, even if no such listing exists today.

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

1.  Clone this repository.
2.  Open `LuxuryWatches.ipynb` in Jupyter.
3.  Make sure `Watches.csv` is in the same directory as the notebook (unzip `Watches.csv.zip` if needed).
4.  Run the cells top-to-bottom.

**Runtime:** The hyperparameter tuning step (Part 6) uses 5-fold cross-validation over 25 random parameter combinations and takes roughly **10 minutes** depending on your machine. Everything else runs in seconds.

### Using the Interactive Estimator

The final section (Part 7) launches an interactive prompt. You will be asked, one at a time, for:

-   Brand, then model (the model options narrow based on your brand choice)
-   Case material, condition, movement, bracelet material, sex/gender
-   Size in millimeters
-   Year of production

For each field, the notebook shows you typical options found in the data. If you don't know a value, just press Enter — the model handles missing values natively. After all inputs, the model prints an estimated market value.

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
2.  **Encoding**: We split train/test *before* encoding to prevent data leakage. Low-cardinality columns (movement, condition, etc.) become pandas categoricals for XGBoost's native handling. High-cardinality columns (brand, model) are target-encoded each category is replaced by its average historical price within the training set.
3.  **XGBoost tuning & training**: We run `RandomizedSearchCV` with 5-fold cross-validation to find good hyperparameters, then train the final model on the full training set.

The section ends with three diagnostic visualizations:

-   A bar chart comparing MAE across baselines and the final model.
-   A SHAP waterfall plot deconstructing one specific prediction.
-   A feature importance bar chart showing which features the model relies on most globally.

The final tuned XGBoost model reduces the average dollar error from \~\$15,000 (naive baseline) to under \$6,000 a meaningful improvement that proves the model has learned non-linear pricing patterns.

### Part 7: Interactive Price Estimator

A user-facing prompt that walks the user through entering a watch's specs and returns an estimated price. The prompt suggests typical options based on the user's earlier selections (e.g., once you pick "Rolex" as the brand, only Rolex models are proposed for the next field), but it also accepts rare valid combinations not currently in the dataset.

### Part 8: Streamlit

Reserved for a future Streamlit web app that wraps the trained model in a browser interface, so users can run the estimator without opening Jupyter.

------------------------------------------------------------------------

## Files in This Repository

| File                  | Purpose                                     |
|-----------------------|---------------------------------------------|
| `LuxuryWatches.ipynb` | The main notebook — open this.              |
| `Watches.csv`         | Raw dataset (uncompressed, \~40 MB).        |
| `Watches.csv.zip`     | Same dataset, zipped (\~7 MB).              |
| `README.md`           | This file.                                  |
| `requirements.txt`    | Python packages needed to run the notebook. |
| `AI_reflection.md`    | Reflection on how AI tools were used during the project. |

------------------------------------------------------------------------

## Notes & Limitations

-   The model's accuracy depends heavily on how well-represented a watch is in the training data. Predictions for very rare brands or one-of-a-kind pieces will be less reliable than predictions for common references like a Rolex Submariner.

-   Prices in the dataset reflect listed asking prices, not transaction prices, so the estimator predicts a fair *listing* price rather than a guaranteed sale price.
