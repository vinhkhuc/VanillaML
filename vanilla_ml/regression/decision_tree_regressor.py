from vanilla_ml.base.decision_tree import DecisionTreeBase
from vanilla_ml.regression.abstract_regressor import AbstractRegressor


class DecisionTreeRegressor(AbstractRegressor, DecisionTreeBase):

    def __init__(self,
                 max_depth=3,
                 criterion='mse',
                 min_leaf_samples=1,
                 rand_features_ratio=None,
                 rand_state=42,
                 verbose=False):

        assert criterion == 'mse', "The criterion '%s' is not supported by DecisionTreeRegressor." % criterion

        super(DecisionTreeRegressor, self).__init__(max_depth, criterion, min_leaf_samples,
                                                    rand_features_ratio, rand_state, verbose)

    def fit(self, X, y, sample_weights=None):
        DecisionTreeBase.fit(self, X, y, sample_weights)

    def predict(self, X):
        return DecisionTreeBase.predict(self, X)
