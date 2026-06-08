from sklearn.base import BaseEstimator, TransformerMixin

class BrandModelEncoder(BaseEstimator, TransformerMixin):
    """Wraps a TargetEncoder fitted on [brand, model] only into a full-DataFrame transform."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.cols = ['brand', 'model']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[['brand', 'model']] = self.encoder.transform(X[['brand', 'model']])
        return X