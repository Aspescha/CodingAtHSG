---

editor_options: 
  markdown: 
    wrap: 72
---

# Luxury Watch Price Estimator

A data-driven tool to estimate the fair market value of any luxury watch, built as the group project for Skills: Introduction to Programming at the University of St. Gallen (HSG).

**Authors:** Jan Nemeth, Daniel Kullmann, Timon, Julian, Philipp Sathy, Yannick Allmann

------------------------------------------------------------------------

## What We Are Doing

The luxury watch market is opaque and fragmented. Identical watches list at different prices across dealers, and a private seller has almost no objective way to know what their watch is worth.

Therefore this project builds a machine learning model that predicts the resale price of a luxury watch based on its core specifications: brand, model, year of production, case material, bracelet material, movement, size, condition, and intended wearer.

We work with a Kaggle dataset of luxury watch listings, clean it, explore it, and train an XGBoost regression model on the log-transformed price. The end product is an interactive Streamlit web app that lets a user enter the specs of any watch and instantly receive an estimated market value.

------------------------------------------------------------------------

## Project Structure — Why Everything Exists Twice

You will notice that the code appears in two places. This is intentional:

**The notebook (`LuxuryWatches.ipynb`)** is the story. It walks through every step of the project — data loading, cleaning, exploration, modelling, and evaluation — with full explanations and visualizations at each stage. It is meant to be read and graded.

**The source files (`src/`)** are the engine. The same logic is implemented as clean, documented Python classes that power the Streamlit app and the training script. This separation means the app can run without anyone ever opening Jupyter, and it demonstrates good software design: one source of truth, separate from the presentation layer.

In short: the notebook is for understanding, the classes are for deployment.

------------------------------------------------------------------------

## How to Run the Project

### 1. Clone the repository and install dependencies

``` bash
git clone https://github.com/yannickallmann/CodingAtHSG.git
cd CodingAtHSG
pip install -r requirements.txt
```

### 2. Train and save the model

This step trains the XGBoost model and saves it to disk. It takes roughly 2–5 minutes depending on your machine.

``` bash
python train.py
```

The trained model is saved to `models/watch_price_model.joblib`. You only need to do this once.

### 3. Launch the Streamlit app

``` bash
streamlit run app.py
```

This opens the interactive price estimator in your browser. Select a brand, model, and watch specifications, then click "Estimate Price".

### 4. (Optional) Run the notebook

Open `LuxuryWatches.ipynb` in Jupyter or VS Code and run all cells top to bottom. The hyperparameter tuning step (Part 6) takes roughly 5–10 minutes.

------------------------------------------------------------------------

## About the Notebook

### Part 1: Dataset Loading

We load the Kaggle dataset of \~284,000 luxury watch listings and inspect the raw data.

### Part 2: Data Cleaning — Missing Values

We quantify missing data, drop uninformative columns, and consolidate the two redundant condition columns without losing rows. We also document the selection bias introduced by dropping "Price on request" entries and rows with missing prices — the cleaned dataset skews towards liquid, well-documented, mid-range watches.

### Part 3: Data Cleaning — Data Types

We parse raw strings into proper numeric types: case size (mm), year of production, and price. All cleaning is vectorized for performance.

### Part 4: Exploratory Data Analysis

We explore distributions and relationships visually. Key findings: price is heavily right-skewed (log transformation needed), brand and material are strong price drivers, size and year of production matter less in isolation.

### Part 5: Data Preprocessing

We apply the log transformation to the price target and prepare categorical variables. Missing categoricals are filled with "Unknown" so the model treats the absence of information as its own signal. The target encoder for high-cardinality variables (brand, model) is placed inside the cross-validation pipeline to prevent data leakage.

### Part 6: Machine Learning

Three-step process: baselines (Dummy + Ridge), XGBoost pipeline with target encoding and hyperparameter tuning via RandomizedSearchCV, and final evaluation using MAPE. The tuned model achieves a MAPE of 30.3%, compared to 216.7% for the naive baseline and 40.3% for Ridge.

Visualizations include a MAPE comparison chart, a SHAP waterfall plot for a single prediction, and a feature importance chart. These are interpreted carefully — brand and model dominate importance scores partly because they carry price information through target encoding, not necessarily because of independent causal effects.

### Part 7: Interactive Price Estimator

A notebook-based prompt that walks the user through entering watch specs and returns an estimated market value. Useful for quick testing without launching the full Streamlit app.

------------------------------------------------------------------------

## Files in This Repository

| File/Folder | Purpose |
|----------------------|--------------------------------------------------|
| `LuxuryWatches.ipynb` | Main notebook — the full story with explanations. |
| `src/cleaner.py` | `WatchDataCleaner` class — loads and cleans raw data. |
| `src/eda.py` | `WatchEDA` class — all exploratory visualizations. |
| `src/model.py` | `WatchPriceModel` class — train, evaluate, save, predict. |
| `train.py` | One-time script to train and save the model. |
| `app.py` | Streamlit web app — the interactive price estimator. |
| `data/Watches.csv` | Raw dataset (\~40 MB). |
| `data/Watches.csv.zip` | Same dataset, zipped (\~7 MB). |
| `models/` | Saved trained model (generated by `train.py`). |
| `requirements.txt` | Python packages needed to run the project. |
| `AI_reflection.md` | Reflection on how AI tools were used during the project. |
| `README.md` | This file. |

------------------------------------------------------------------------

## Notes & Limitations

- **Selection bias**: the cleaned dataset excludes "Price on request" listings and watches with missing prices. These tend to be rare, illiquid, or ultra-high-end pieces. The model is most reliable for mainstream luxury watches in the \$2,000–\$50,000 range.

- **Condition paradox**: changing condition alone (e.g. Good → Poor) may not always move the price in the expected direction. This is because brand and model dominate the model's predictions, and condition's effect is relative — a Poor Rolex Daytona still fetches more than a Mint Seiko.

- **Target encoding**: brand and model are encoded using their average historical price. Their high feature importance is partly a reflection of this encoding, not evidence of independent causal effects.

- **Asking prices vs. transaction prices**: the dataset reflects listed prices, not confirmed sales. The estimator predicts a fair listing price rather than a guaranteed sale price.

## How Well Does It Work in Practice?

Tested against real secondary market listings, the model performs well for standard, liquid watches. Performance degrades predictably for edge cases — unusual material combinations, very rare conditions like "Unworn" or "Poor", and made-up brand/model combinations that were never in the training data. This is expected and honest: it is fundamentally impossible to price something that has never been sold before. The model can only interpolate within what it has seen - it cannot extrapolate into uncharted territory.
