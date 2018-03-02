SHELL := /bin/bash

TEST = '*Test.py'

all: test clean

test:
	echo "Start testing ..."
	python -m unittest discover -vp $(TEST)

test-light:
	echo "Start (light) testing ..."
	N_RANDOM_FEATURE_SETS=5 N_EXISTING_FEATURE_SETS=5 \
		python -m unittest discover -vp $(TEST)

lint:
	echo "Start linting ..."
	find ./sklearn_porter -name '*.py' -exec pylint {} \;

clean:
	echo "Start cleaning ..."
	rm -rf tmp
	rm -rf build/*
	rm -rf dist/*
