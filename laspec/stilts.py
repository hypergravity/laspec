"""
python wrapper of stilts
"""

import os
from laspec.utils import download


def stilts(**kwargs):
    stilts_executable = download("stilts")
    cmd = f"java -jar {stilts_executable}"
    for k, v in kwargs.items():
        cmd += f" {k} {v}"
    return os.system(cmd)
