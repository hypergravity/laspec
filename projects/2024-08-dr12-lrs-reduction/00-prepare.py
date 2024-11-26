"""
cd /nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog
echo 3 > /proc/sys/vm/drop_caches
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/hypergravity/laspec.git --force-reinstall
# pip install git+https://gitee.com/hypergravity/laspec.git --force-reinstall
"""

import os
import joblib
from laspec.ccf import RVM
from astropy import table
import numpy as np
from laspec.lamost_kits import CodeKit
from laspec.mrs import MrsFits

hostname = os.uname()[1]
if hostname in ["alpha", "beta", "gamma"]:
    WORKDIR = "/nfsdata/share/lamost/dr12v0-lrs-reduction"
    SPECDIR11 = "/nfsdata/share/lamost/dr11-v1.0/fits"
    SPECDIR12 = "/nfsdata/share/lamost/dr12-v0/fits"
# elif hostname == "Mac-Studio.local":
#     WORKDIR = "/Users/cham/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog"
#     SPECDIR = "/Users/cham/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
# elif hostname.startswith("MBP"):
#     WORKDIR = "/Users/cham/projects/lamost/dr11-v1.0"
#     SPECDIR = "/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
else:
    raise ValueError(f"Invalid hostname {hostname}")
os.chdir(WORKDIR)
