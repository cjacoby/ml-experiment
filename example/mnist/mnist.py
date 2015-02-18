import yaml
import logging
import argparse

from experiment import Experiment
from experiment import feature_extractor
from experiment import model

logger = logging.getLogger("mnist_experiment")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

def setup_experiment(run_type, config_file="mnist_config.yaml", verbosity=None):
    experiment_config = {
        "feature_extractor_class" : feature_extractor.FeatureExtractor,
        "run_type" : run_type,
        "model_class" : model.Model,
        "hyperparameters" : None,
        "verbosity" : verbosity
    }
    with open(config_file, 'r') as fh:
        config = yaml.load(fh)
        experiment_config.update(config)

    return Experiment(**(experiment_config))


def train(args):
    experiment = setup_experiment("train", verbosity=args.verbosity)

    # Run training
    experiment(timeout=1000)

def test(args):
    experiment = setup_experiment("test", verbosity=args.verbosity)

    # Run test
    experiment(timeout=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase verbosity: 1=info, 2=debug")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    if args.verbosity == 0:
        ch.setLevel(logging.WARNING)
    elif args.verbosity == 1:
        ch.setLevel(logging.INFO)
    else:# args.verbosity == 2:
        ch.setLevel(logging.DEBUG)

    args.func(args)
