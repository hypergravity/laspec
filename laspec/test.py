import os

from conda.exceptions import DirectoryNotFoundError


def get_test_data(relative_file_path: str = "README.md") -> str:
    """Get the absolute path to the test data file.

    Parameters
    ----------
    relative_file_path : str
        The relative path to the test data file.

    Returns
    -------
    str
        The absolute path to the test data file.
    """
    # get test data root
    LASPEC_TEST_DATA_ROOT = os.getenv(
        "LAPSEC_TEST_DATA_ROOT",
        # by default, parallal to laspec package
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data"),
    )
    if not os.path.exists(LASPEC_TEST_DATA_ROOT):
        raise DirectoryNotFoundError(f"{LASPEC_TEST_DATA_ROOT} does not exist.")

    # get absolute file path
    absolute_file_path = os.path.join(LASPEC_TEST_DATA_ROOT, relative_file_path)
    if not os.path.exists(absolute_file_path):
        raise FileNotFoundError(
            f"File does not exist. \n"
            f"absolute_file_path = {absolute_file_path} \n"
            f"LASPEC_TEST_DATA_ROOT = {LASPEC_TEST_DATA_ROOT} \n"
        )
    return absolute_file_path
