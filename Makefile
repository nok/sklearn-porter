BASH := /bin/bash

PYTHON_FILES := $(shell find -s ./sklearn_porter -name '*.py' | tr '\n' ' ')

TEST_N_RANDOM_FEATURE_SETS=25
TEST_N_EXISTING_FEATURE_SETS=25

#
# Requirements
#

install.environment:
	$(info Start [install.environment] ...)
	$(BASH) scripts/install.environment.sh

install.requirements:
	$(info Start [install.requirements] ...)
	$(BASH) scripts/install.requirements.sh

install.requirements.examples: install.requirements
	$(info Start [install.requirements.examples] ...)
	$(BASH) scripts/install.requirements.examples.sh

install.requirements.development: install.requirements.examples
	$(info Start [install.requirements.development] ...)
	$(BASH) scripts/install.requirements.development.sh

#
# Examples
#

open.examples: install.requirements.examples examples.pid

examples.pid:
	$(info Start [examples.pid] ...)
	jupyter-lab --notebook-dir='examples' > /dev/null 2>&1 & echo $$! > $@;

stop.examples: examples.pid
	kill `cat $<` && rm $<

.PHONY: open.examples stop.examples

#
# Development
#

all: lint test jupytext clean

lint: install.requirements.development
	$(info Start [lint] ...)
	pylint --rcfile=.pylintrc --output-format=text $(PYTHON_FILES) 2>&1 | tee pylint.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p'

test: install.requirements.development
	$(info Start [test] ...)
	TEST_N_RANDOM_FEATURE_SETS=$(TEST_N_RANDOM_FEATURE_SETS) \
	TEST_N_EXISTING_FEATURE_SETS=$(TEST_N_EXISTING_FEATURE_SETS) \
		$(BASH) scripts/run.tests.sh

test.sample: install.requirements.development
	$(info Start [test.sample] ...)
	TEST_N_RANDOM_FEATURE_SETS=3 \
	TEST_N_EXISTING_FEATURE_SETS=3 \
		$(BASH) scripts/run.tests.sh

jupytext: install.requirements.development
	$(info Start [jupytext] ...)
	$(BASH) scripts/run.jupytext.sh

deploy: clean
	$(info Start [deploy.test] ...)
	$(BASH) scripts/run.deployment.sh

clean:
	$(info Start [clean] ...)
	rm -rf tmp
	rm -rf build
	rm -rf dist
