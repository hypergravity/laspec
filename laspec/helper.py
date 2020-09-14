import os


def test():
    print(__package__, __file__,)


def laspec_path():
    return os.path.dirname(__file__)


def stilts_path():
    return laspec_path() + "/stilts"
