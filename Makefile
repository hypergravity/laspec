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

build:
	docker build -t laspec . --no-cache

sync:
	rsync -e 'ssh -p 2000' -avzr ~/PycharmProjects/laspec cham@159.226.170.52:~/PycharmProjects/
