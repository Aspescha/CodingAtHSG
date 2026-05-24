---
editor_options: 
  markdown: 
    wrap: 72
---

# Critical Evaluation of `LuxuryWatches.ipynb`

There's solid effort here, but the project has several substantive
issues — some are bugs, some are methodology problems that would fail a
critical reviewer, and some are simply missed opportunities. Below is a
deep, honest critique organized by severity.

------------------------------------------------------------------------

## 1. Methodology: Critical Issues

### 1.1 The "MAE in dollars" headline number is statistically misleading

You train on `log_price` (MSE objective), then `np.exp()` predictions
and report MAE in dollars. This is more wrong than it looks:

-   Minimizing MSE on `log_price` finds the conditional **median** in
    dollar-space, not the mean. So your `np.exp(prediction)` is a
    **systematically biased** dollar estimate (it underpredicts the mean
    of a log-normal).
-   The honest fix is **Duan's smearing estimator**: multiply
    `np.exp(pred)` by `mean(exp(residuals_train))`. Without it, your
    \$5,751 MAE understates true error on expensive watches.
-   A "\$5,751 average error" headline also hides that on a \$200k Patek
    the model can be off by tens of thousands and on a \$2k Seiko it can
    be off by hundreds — same MAE, very different business meaning.
    Report **MAPE** and **MdAPE** alongside MAE, and bucket errors by
    price band.

### 1.2 You do NOT have an apples-to-apples baseline

-   Ridge: median imputation + One-Hot Encoding
-   XGBoost: native NaN handling + target encoding for high-cardinality

So when XGBoost beats Ridge, you don't know whether **the algorithm** or
**the encoding** won. Run Ridge with the *same* target-encoded features.
I'd bet a target-encoded Ridge closes most of the gap, which would
change the story you're telling.

### 1.3 Duplicate listings are never checked

The dataset is scraped from Chrono24-style listings. The same physical
watch is frequently listed by multiple dealers and re-listed over time.
If duplicates straddle your 80/20 split, **train/test contamination
inflates your reported MAE**. Add:
`watch_data.duplicated(subset=[...all features...]).sum()` before
splitting, and either drop or split by reference number.

### 1.4 Single train/test split — no confidence interval

A heavy-tailed target like watch prices makes a single 80/20 split
high-variance. The reported \$5,751 could realistically be
\$5,200–\$6,400 depending on the seed. Either: - Report 5-fold CV MAE on
the full dataset (with refit), or - Run 5 different `random_state` seeds
and report mean ± std.

### 1.5 Unseen-category leakage in target encoder is unmeasured

After the split, brands/models that only appear in the test set get
encoded with the global prior. You never report **how many test rows hit
this fallback** — that's the actual cliff for novel watches. Add a
coverage check.

------------------------------------------------------------------------

## 2. Feature Engineering: Major Misses

### 2.1 You dropped the `ref` column. This is a serious mistake.

Reference numbers (e.g., Rolex `116610LN` vs `116610LV`) are arguably
the **single most informative feature** for watch valuation — they
identify the exact variant, often more predictive than brand+model. You
dismissed it in one line ("does not provide any information"). At
minimum: - Use the reference's brand prefix as a feature -
Frequency-encode the full ref - Or target-encode it like brand/model

### 2.2 `condition` is ordinal, you treat it as nominal

"Poor \< Fair \< Good \< Very Good \< Excellent \< New" is an ordering,
but `astype('category')` discards it. Use
`pd.Categorical(..., ordered=True)` or map to integers. XGBoost can
exploit monotonicity here.

### 2.3 `size` cleaning is brittle heuristic spaghetti

The 25mm/50mm/100mm thresholds work for typical wristwatches but
silently mishandle: - Pocket watches (50–60mm diameter — your code
treats \>50mm as "lug-to-lug") - Cushion/rectangular cases (`28x40mm` —
picks 40, but 28 might be the cased width) - A regex like
`r'(\d{2}(?:\.\d+)?)\s*mm'` finding the *first* mm-suffixed number is
more honest. Or extract diameter and thickness separately as two
features.

### 2.4 `yop` cleaning loses recoverable data

-   `[:4]` truncation drops anything that doesn't start with a 4-digit
    year. "ca. 1980", "circa 1965", "1980s", "1980-1985" → all become
    None.
-   Use `re.search(r'(1[5-9]\d{2}|20[0-2]\d)', str(val))` to extract the
    year regardless of position.
-   Also: you cap at 2026, but the `(Approximation)` strings often refer
    to current production — verify nothing legitimate is being chopped.

### 2.5 No engineered features

You never derived: - `watch_age = current_year - yop` (XGBoost can find
it, but it's a strong, monotonic signal) - `is_precious_metal`
(gold/platinum vs steel) - Brand-tier groupings (Tier 1:
Patek/AP/Vacheron; Tier 2: Rolex/JLC; etc.) - Interaction features like
`brand × condition`

These are exactly the features a watch dealer uses mentally — your model
is rediscovering them inefficiently.

------------------------------------------------------------------------

## 3. Model Tuning Issues

### 3.1 The hyperparameter search is undersized and miscoded

-   `n_iter=25` over a 2,800-combination grid with 5-fold CV samples
    \<1% of the space.
-   **No regularization params searched**: `reg_alpha`, `reg_lambda`,
    `min_child_weight`, `gamma` are all left at defaults. These are
    exactly the knobs that prevent overfitting on a heavy-tailed target.
-   **You're not using early stopping**. Searching
    `n_estimators ∈ {100, ..., 1000}` while specifying it explicitly
    wastes compute. Use `early_stopping_rounds=50` with a validation
    fold and let XGBoost pick the tree count.
-   Hardcoding `best_params = {...}` after the search is fragile — the
    next reader who re-runs the search gets different numbers but uses
    the old hardcoded set.

### 3.2 `objective='reg:squarederror'` may not be optimal

With heavy-tailed (even after log) residuals, `reg:absoluteerror` (MAE)
or `reg:pseudohubererror` aligns better with your reporting metric.
Compare them.

------------------------------------------------------------------------

## 4. Concrete Bugs

### 4.1 Era-binning bug in "old code storage"

``` python
bins=[0, 1980, 2000, 2010, 2020, 2030],
labels=['Vintage', '1980s', '2000s', '2010s', '2020s']
```

The bin `(1980, 2000]` is labeled `'1980s'` — but it actually contains
1981–2000. You're labeling 90s watches as "1980s". The label `'1990s'`
is missing entirely. (Yes, this is in dead code, but it's wrong and
someone will copy it.)

### 4.2 Currency is contradictory across the notebook

-   Plot axes: `Log Price (CHF)`
-   Bucket labels: `< 5000 CHF`, `< 20000 CHF`
-   Print statements: `Off by $X` and `real dollars`
-   Dataset is from Kaggle/Chrono24 — original currency is almost
    certainly **EUR**, not USD or CHF.

For an HSG project, this is the kind of thing a TA will circle in red.
Pick one currency, label it consistently, and document the assumption.

### 4.3 `palette=` without `hue=` will break in newer seaborn

Multiple `sns.barplot(x=..., y=..., palette='viridis')` calls trigger
`FutureWarning` in seaborn ≥0.13 and will error in future versions. Pass
`hue=x_var, legend=False` instead.

### 4.4 Standard-scaling `size` in the dead code block is pointless

The active pipeline is XGBoost (scale-invariant). The "old code" applies
`StandardScaler` and drops original `size`. If anyone re-uses that block
they'll silently break the active pipeline. Delete the dead code or move
to a separate `archive.ipynb`.

### 4.5 Interactive UI lies about itself

``` python
print("The input boxes will appear at the top of your screen.")
```

But you call `input()`, which is a terminal prompt at the bottom. Either
implement actual `ipywidgets` or fix the message.

### 4.6 Author list incomplete

`#### Yannick Allmann, Jan Nemeth (add names)` — but the README lists
six people. Fill it in.

------------------------------------------------------------------------

## 5. Validation & Diagnostics Missing

You build a model and report a single number. A reviewer will want:

1.  **Residual plot** (predicted vs actual, log-space and dollar-space)
    — reveals heteroscedasticity, systematic bias by price band.
2.  **Q-Q plot of residuals** — confirms (or denies) the log-normal
    assumption that justifies the log transform.
3.  **Error decomposition by brand and price tier** — does the model
    fail on cheap Seikos or expensive Pateks? This is the most
    actionable diagnostic and you have the data for it.
4.  **Learning curve** — train vs validation MAE as a function of
    training size, to know whether more data would help.
5.  **SHAP global summary plot** (beeswarm) — you only show a waterfall
    for one watch. The beeswarm reveals which features matter and *in
    which direction* across the whole dataset. Two extra lines of code.
6.  **Calibration plot** — bin predictions into deciles and plot
    predicted vs observed mean. Tells you if the model is
    well-calibrated or systematically biased somewhere.

------------------------------------------------------------------------

## 6. Code Quality

-   **`watch_data` vs `watch_data_cleaned`** — inconsistent. The
    cleaning happens in-place on `watch_data`, then renamed at the end.
    Half the EDA uses the wrong name.
-   **40MB CSV committed to git** alongside a 7MB zip of the same. Keep
    the zip, gitignore the csv, load with
    `pd.read_csv("Watches.csv.zip")`.
-   **No `requirements.txt` or `environment.yml`** — `!pip install` in a
    cell is not reproducibility.
-   **No random seed for the target encoder** (it's deterministic for
    the basic case, but not if you use `category_encoders` with
    leave-one-out variants later).
-   **Magic numbers everywhere**: `min_samples_leaf=10`, `smoothing=10`,
    `0.5` threshold for imputation, `25/50/100` for size. Lift them into
    named constants at the top.
-   **Typos in markdown** are pervasive: "descirbe", "menas", "boolen",
    "freuqent", "ctaegories", "imporve". Run a spell-checker before
    submission.

------------------------------------------------------------------------

## 7. The Streamlit Section is Empty

`## Part 8: Streamlit` has no implementation. Either build the app or
remove the heading. If you build it: serialize the trained model +
target encoder with `joblib`, load in a `streamlit_app.py`, and provide
a one-line `streamlit run` command in the README.

------------------------------------------------------------------------

## Top 5 Things to Fix Before Submission

If you only have time for five things, do these:

1.  **Fix the dollars/CHF/EUR currency mess** — one currency, labeled
    everywhere consistently.
2.  **Bring back `ref` as a feature** (frequency- or target-encoded).
    This will likely move the MAE meaningfully.
3.  **Apply Duan's smearing factor** when exponentiating log
    predictions, and report MdAPE alongside MAE.
4.  **Add early stopping + regularization params** to the hyperparameter
    search; remove the hardcoded `best_params` block.
5.  **Add residual diagnostics + error-by-price-tier + SHAP beeswarm** —
    three plots that turn this from "we got a number" into "we
    understand our model".

If I can pick a sixth: **dedupe the dataset before splitting** and
report how many duplicates you found.
