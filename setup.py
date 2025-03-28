import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as f:
    requirements = [req.strip() for req in f.readlines() if not req.startswith("#")]

setuptools.setup(
    name="laspec",
    version="2024.11.26",
    author="Bo Zhang",
    author_email="bozhang@nao.cas.cn",
    description="Modules for LAMOST spectra.",  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/hypergravity/laspec",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    # package_dir={"laspec": "laspec"},
    include_package_data=False,
    package_data={
        "": ["LICENSE", "README.md"],
        "laspec": ["config/*.toml"],
    },
    #  "laspec": ["data/*",
    #             "data/FERRESun/*",
    #             "data/KIC8098300/*/*",
    #             "data/lamost/*",
    #             "data/phoenix/*",
    #             "data/songmgb/*",
    #             "stilts/*"],
    install_requires=requirements,
)
