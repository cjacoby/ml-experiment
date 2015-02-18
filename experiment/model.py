

class Model(object):
    "model mixin, based on sklearn."

    def __init__(self, sample_factory, hyperparameters):
        self.sample_factory = sample_factory
        self.hyperparams = hyperparameters

    def fit(self, X, Y):
        pass
