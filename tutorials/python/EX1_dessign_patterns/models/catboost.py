from typing import Any, Dict

from catboost import CatBoostRegressor, Pool, cv


class CatBoost:
    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.model = None

    def train(self, X_train, y_train):
        # model = CatBoostRegressor(self.configs.train_params)
        pool = Pool(X_train, y_train)
        scores = cv(pool, self.configs.train_params)
        self.model = ...

    def predict(self): ...

    def save_model(self): ...
