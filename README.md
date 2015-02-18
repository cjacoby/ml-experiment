experiment
=======
Module for managing machine learning experiments.

Modules
-------
experiment (in __init__) - parent class which handles executing the segments of the experiment based on the input parameters.

path_mgr - things related to managing the experiment paths

feature_extractor - utility for managing the feature extraction process. No feature extraction should actually be implemented here;
    this utility essentially just executes the feature extraction given the input data.

sample_factory - module for managing generation of sample data.
    Handles train/test splits, cross validation, "training mode", "validation mode", "test mode"

model - module for executing model code, and getting result outputs.
