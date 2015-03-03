import os
import numpy as np

class SampleFactory(object):
    def __init__(self, input_files, data_split_mode, run_type, max_epochs=None, batch_size=1):
        """ Initialize the sample factory from the data and the modes. """
        if input_files is None or len(input_files) < 1:
            raise ValueError("Not enough valid input files")
        else:
            for fn in input_files:
                if not os.path.exists(fn):
                    raise IOError("Input file does not exist")
        if data_split_mode not in ["3/2", "5fold", "10fold"]:
            raise ValueError("Invalid Test/Train split: {}".format(data_split_mode))
        if run_type not in ["train", "test"]:
            raise ValueError("Invalid run type: {}".format(run_type))

        self.input_files = input_files
        self.split_mode = data_split_mode
        self.run_type = run_type
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.sample_count = 0
        self.epoch_count = 0
        self.sample_size = None

        self.load_data()

    def __iter__(self):
        return self.__next__()

    # Generator function
    def __next__(self):
        if self.sample_count >= self.sample_size:
            self.epoch_count += 1
        if self.epoch_count >= self.max_epochs:
            raise StopIteration()

        return self.sample_data()

    def load_data(self):
        data = []
        for filename in self.input_files:
            fh = np.load(filename)
            data.append([fh['data'], fh['targets']])
        self.data = np.concatenate([x[0] for x in data])
        self.targets = np.concatenate([x[1] for x in data])

        self.sample_size = len(self.data)

    def sample_data(self):
        # get random index
        i = np.random.randint(self.sample_size)
        return self.data[i], self.targets[i]
