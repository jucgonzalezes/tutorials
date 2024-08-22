from category_encoders import CatBoostEncoder

class CatBoostRegressor:
    def __init__(self, **kwargs):
        self.model = self.set_model(**kwargs)

    @staticmethod
    def set_model(**kwargs):
        ...


