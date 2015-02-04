""" This module defines an experiment class, and the classes related to the parts of running an experiment.
"""

import feature_extractor
import model
import path_mgr
import sample_factory

class Experiment(object):
    def __init__(self, input, feature_path, feature_extractor, data_split_mode, run_type, model_class, hyperparameters):
        """ A master class for running experiments.

        Parameters
        ---------
        input : str or array-like
            If str:
                If folder, is the path to a folder containing the input files.
                If file, is the path to a json or yaml file containing the list of input files.
            If array-like, is the list of input files.
        feature_path : str
            Path to the output folder which will contain the features, or feature .hd5 file.
        feature_extractor : reference to a class (maybe object?)
            which implements the FeatureExtractor mixin, defined in feature_extractor.
        run_type : str
            'train' or 'test'
        hyperparameters : dict
            dict keyed by hyperparameter name.
        """
        pass
