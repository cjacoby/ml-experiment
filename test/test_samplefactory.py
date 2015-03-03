import os
import numpy as np
import py.test

import experiment.sample_factory as esf

def create_input_files(n_files, n_points):
        base_name = "foodata"
        ext = "npz"

        input_files = []
        for i in xrange(n_files):
            X = np.random.random([n_points, 5])
            Y = np.random.randint(3, size=n_points)

            filename = "{}-{}.{}".format(base_name, i, ext)
            np.savez(filename, data=X, targets=Y)
            input_files.append(filename)

        return input_files

class FakeData(object):
    def __init__(self, n_files, n_points=1000):
        self.input_files = create_input_files(n_files, n_points)

    def destroy_input_files(self):
        import os  # 'cause os went out of scope when this gets called?
        for fn in self.input_files:
            if os.path.exists(fn):
                os.remove(fn)

    def __del__(self):
        self.destroy_input_files()

fakeData = FakeData(5)

def test_fake_data():
    for filename in fakeData.input_files:
        assert os.path.exists(filename)

        d = np.load(filename)
        assert "data" in d.keys()
        assert "targets" in d.keys()


def test_init_sf_input_files():
    with py.test.raises(ValueError):
        esf.SampleFactory(None, "foo", "bar")

    with py.test.raises(ValueError):
        esf.SampleFactory([], "foo", "bar")

    with py.test.raises(IOError):
        esf.SampleFactory(["what.biz", "whodunnit.foo", "foodata-0.npz"], "foo", "bar")

def test_init_sf_split_modes():
    with py.test.raises(ValueError):
        esf.SampleFactory(fakeData.input_files, "foo", "train")

    sf = esf.SampleFactory(fakeData.input_files, "3/2", "train")
    sf = esf.SampleFactory(fakeData.input_files, "5fold", "train")
    sf = esf.SampleFactory(fakeData.input_files, "10fold", "train")


def test_init_sf_run_types():
    with py.test.raises(ValueError):
        esf.SampleFactory(fakeData.input_files, "3/2", "foo")
    sf = esf.SampleFactory(fakeData.input_files, "3/2", "train")
    sf = esf.SampleFactory(fakeData.input_files, "3/2", "test")


def test_sf_generator_batch1():
    sf = esf.SampleFactory(fakeData.input_files, "3/2", "train", max_epochs=10, batch_size=1)

    for X, Y in sf:
        assert X.shape[0] == 1
        assert Y.shape[0] == 1


def test_sf_train():
    assert False


def test_sf_test():
    assert False
