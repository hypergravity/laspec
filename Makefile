all: install clean

install:
	pip install .
	rm -rf build dist *egg-info

upload:
	python setup.py sdist bdist_wheel
	twine upload --verbose dist/*

clean:
	rm -rf build dist *egg-info

test: install clean
	pytest . --import-mode=prepend
