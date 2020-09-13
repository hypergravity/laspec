#!/usr/bin/env bash
#python setup.py sdist upload
rm ./dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
