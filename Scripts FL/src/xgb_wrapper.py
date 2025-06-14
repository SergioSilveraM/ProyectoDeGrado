from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
import xgboost as xgb

class XGBWrapper:
    def __init__(self, params, numeric_features, categorical_features):
        self.params = params.copy()
        self.num_boost_round = self.params.pop("num_boost_round")
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", None)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = self._build_preprocessor()
        self.model = None
        self.evals_result = None
        self.best_iteration = None
        self.best_score = None

    def _build_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), self.numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Sin Dato')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), self.categorical_features)
            ],
            verbose_feature_names_out=False,
            force_int_remainder_cols=False
        )

    def fit(self, X, y, eval_set=None, sample_weight=None):
        self.preprocessor = self._build_preprocessor().fit(X)
        X_proc = self.preprocessor.transform(X)

        if sample_weight is not None:
            dtrain = xgb.DMatrix(X_proc, label=y, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(X_proc, label=y)

        evals = [(dtrain, "train")]
        evals_result = {}

        if eval_set is not None:
            X_val, y_val = eval_set[0]
            X_val_proc = self.preprocessor.transform(X_val)
            dval = xgb.DMatrix(X_val_proc, label=y_val)
            evals.append((dval, "validation"))

        train_params = {
            'params': self.params,
            'dtrain': dtrain,
            'num_boost_round': self.num_boost_round,
            'evals': evals,
            'evals_result': evals_result,
            'verbose_eval': False
        }

        if self.early_stopping_rounds is not None and len(evals) > 1:
            train_params['early_stopping_rounds'] = self.early_stopping_rounds

        self.model = xgb.train(**train_params)
        self.evals_result = evals_result
        self.best_iteration = getattr(self.model, "best_iteration", None)
        self.best_score = getattr(self.model, "best_score", None)                           

        return self

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        dmatrix = xgb.DMatrix(X_proc)
        y_proba = self.model.predict(dmatrix)
        return np.argmax(y_proba, axis=1)

    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        dmatrix = xgb.DMatrix(X_proc)
        return self.model.predict(dmatrix)

    def save(self, path):
        joblib.dump({
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessor": self.preprocessor,
            "booster": self.model,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        params = {**data["params"], "num_boost_round": data["num_boost_round"]}

        if "early_stopping_rounds" in data:
            params["early_stopping_rounds"] = data["early_stopping_rounds"]

        wrapper = cls(
            params,
            data["numeric_features"],
            data["categorical_features"]
        )
        wrapper.preprocessor = data["preprocessor"]
        wrapper.model = data["booster"]
        wrapper.best_iteration = data.get("best_iteration")
        wrapper.best_score = data.get("best_score")
        return wrapper
