

class SampleFactory(object):
    def __init__(self, input_files, data_split_mode, run_type):
        self.input_files = input_files
        self.split_mode = data_split_mode
        self.run_type = run_type

    # Generator function
    # def
