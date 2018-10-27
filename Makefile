BASH := /bin/bash

TEST_N_RANDOM_FEATURE_SETS=25
TEST_N_EXISTING_FEATURE_SETS=25

#
# Requirements
#

install.environment:
	$(info Start [install.environment] ...)
	$(BASH) recipes/install.environment.sh

install.requirements:
	$(info Start [install.requirements] ...)
	$(BASH) recipes/install.requirements.sh

install.requirements.examples: install.requirements
	$(info Start [install.requirements.examples] ...)
	$(BASH) recipes/install.requirements.examples.sh

install.requirements.development: install.requirements.examples
	$(info Start [install.requirements.development] ...)
	$(BASH) recipes/install.requirements.development.sh

#
# Examples
#

link: install.function
install.function:
	$(info Start [install.function (to .bash_profile)] ...)
	$(BASH) recipes/install.function.sh

start.examples: install.requirements.examples examples.pid

examples.pid:
	$(info Start [examples.pid] ...)
	jupyter-lab --notebook-dir='examples' > /dev/null 2>&1 & echo $$! > $@;

stop.examples: examples.pid
	kill `cat $<` && rm $<

.PHONY: start.examples stop.examples

#
# Development
#

all: lint test clean

test: install.requirements.development
	$(info Start [test] ...)
	TEST_N_RANDOM_FEATURE_SETS=$(TEST_N_RANDOM_FEATURE_SETS) \
	TEST_N_EXISTING_FEATURE_SETS=$(TEST_N_EXISTING_FEATURE_SETS) \
		$(BASH) recipes/run.tests.sh

test.sample: install.requirements.development
	$(info Start [test.sample] ...)
	TEST_N_RANDOM_FEATURE_SETS=3 \
	TEST_N_EXISTING_FEATURE_SETS=3 \
		$(BASH) recipes/run.tests.sh

lint: install.requirements.development
	$(info Start [lint] ...)
	find ./sklearn_porter -name '*.py' -exec pylint {} \;

jupytext: install.requirements.development
	$(info Start [jupytext] ...)
	$(BASH) recipes/run.jupytext.sh

clean:
	$(info Start [clean] ...)
	rm -rf tmp
	rm -rf build
	rm -rf dist
