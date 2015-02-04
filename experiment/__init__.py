""" This module defines an experiment class, and the classes related to the parts of running an experiment.
"""

import os
import time
import logging
import json, yaml

import feature_extractor
import model
import path_mgr
from sample_factory import SampleFactory

logger = logging.getLogger(__name__)

class Experiment(object):
    def __init__(self, input,
                 feature_path, feature_extractor_class,
                 data_split_mode, run_type,
                 model_class, hyperparameters,
                 max_epochs,
                 output_path,
                 verbosity=None):
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
        feature_extractor_class : reference to a class
            which implements the FeatureExtractor mixin, defined in feature_extractor.
        data_split_mode : str
            "3/2" | "5fold" | "10fold"
        run_type : str
            'train' or 'test'
        model : class/type
            Class which handles training or testing the data.
        hyperparameters : dict
            dict keyed by hyperparameter name.
        max_epochs : int
            maximum number of epochs
        output_path : str
            path to output directory for logs, results, etc.
        verbosity : int
            Logging verbosity
        """
        # Load the input
        self.input_files = self._init_input(input)
        self.paths = self._init_paths(feature_path, output_path)

        self.feature_extractor = feature_extractor()
        self.sample_factory = self._init_factory(data_split_mode, run_type)
        self.run_type = run_type
        self.max_epochs = max_epochs
        self.model = model_class(self.sample_factory, hyperparameters)

    def __call__(self, timeout=None):
        """ Execute the experiment.

        Parameters
        ---------
        timeout : float
            Duration in seconds after which to kill the experiment.
            None disables this.
        """
        t0 = time.time()
        if timeout is not None:
            self.tEnd = t0 + timeout

        if self.run_type == "train":
            for epoch in xrange(self.max_epochs):
                self._check_timeout(epoch)
                logger.info("Experiment - Epoch: {}, time: {}".format(epoch, time.time() - t0))

                self.epoch_results()

        elif self.run_type == "test":
            self.epoch_results()

        self.save_results()

    def _check_timeout(self, epoch=None):
        tNow = time.time()
        if self.tEnd is not None and tNow > self.tEnd:
            raise RuntimeError("[Epoch: {}, time:{}] timeout - exiting".format(epoch, tNow))

    def _init_input(self, input):
        """ Process input to get a list of files to send to the feature extractor. """
        input_list = []
        if type(input) == str:
            if os.path.isdir(input):
                for path in os.listdir(input):
                    input_list.append(os.path.join(input, path))
            elif os.path.isfile(input):
                if os.path.splitext(input)[1] == ".json":
                    with open(input) as fh:
                        input_list = json.load(input)
                elif os.path.splitext(input)[1] == ".yaml":
                    input_list = yaml.load(input)
                else:
                    raise RuntimeError("File {} is not a valid input file type.".format(input))
        elif isinstance(input, (list, tuple, set)):
            for path in input:
                if not os.path.exists(path):
                    logger.warn("Input file [{}] does not exist.".format(path))
                input_list.append(path)
        return input_list

    def _init_paths(self, feature_path, output_path):
        """ Save paths to path dict and make sure they exist."""
        path_dict = {}
        if not os.path.exists(feature_path):
            os.mkdirs(feature_path)
        if not os.path.exists(output_path):
            os.mkdirs(output_path)
        path_dict["feature_path"] = feature_path
        path_dict["output_path"] = output_path
        return path_dict

    def _init_factory(self, data_split_mode, run_type):
        " Initialize the sample factory from the data and the modes. "
        if data_split_mode not in ["3/2", "5fold", "10fold"]:
            raise ValueError("Invalid Test/Train split: {}".format(data_split_mode))
        if run_type not in ["train", "test"]:
            raise ValueError("Invalid run type: {}".format(self.run_type))

        return SampleFactory(self.input_files, data_split_mode, run_type)

    def epoch_results(self):
        " Store the epoch results "
        if self.run_type == "train":
            self.all_train_results = []
            self.all_valid_results = []
            for train_data, valid_data in self.sample_factory:
                training_results = self.model.fit(train_data)
                validation_results = self.model.predict(valid_data)
                self.all_train_results.append(training_results)
                self.all_valid_results.append(validation_results)
        elif self.run_type == "test":
            self.all_test_results = []
            for test_data in self.sample_factory:
                test_results = self.model.predict(test_data)
                self.all_test_results.append(test_results)

    def save_results(self):
        "finally, write the pass results to a file."
        if self.run_type == "train":
            pass
        else:
            pass
