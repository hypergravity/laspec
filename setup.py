import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='laspec',
    version='2020.0320.2',
    author='Bo Zhang',
    author_email='bozhang@nao.cas.cn',
    description='Modules for LAMOST spectra.',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/hypergravity/laspec',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    package_dir={'laspec': 'laspec'},
    # include_package_data=True,
    package_data={"laspec": ["data/*",
                             "data/phoenix/*"],
                  "": ["LICENSE"]
                  },
    requires=['numpy', 'scipy', 'astropy', 'lmfit']
)
